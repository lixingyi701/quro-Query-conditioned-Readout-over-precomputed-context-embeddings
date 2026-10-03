"""Auxiliary supervision must reach the projection without changing generation."""
import copy
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_shared_projector as shared_tests
from src.data import QuROCollator, QuRODataset
from src.model import QuROModel, build_query_encoder
from src.projector import SharedDocumentProjector
from src.support import (annotate_row, balanced_support_loss, manifest_digest,
                         support_scores, support_targets, support_source_digest)
from src.train import evaluate


def annotate_fixture(row, cache, visible=None):
    row = copy.deepcopy(row)
    row["gold_ranks"] = [0]
    ids = row["retrieved_doc_ids"]
    row["support_annotation"] = {
        "version": 1, "doc_ids": ids, "labels": [1]+[0]*(len(ids)-1),
        "visible": [True if visible is None else visible]+[None]*(len(ids)-1),
        "cache_digest": manifest_digest(cache.manifest),
        "source_digest": support_source_digest(row),
        "encoder_rule": "pisco-decoder-encoder-128-plus-3-right",
    }
    return row


class SupportLossTests(unittest.TestCase):
    def test_balancing_ignores_class_count_and_skips_invisible_positives(self):
        logits = torch.tensor([[1., -2., -2., -2.], [float("nan"), 0., 0., 0.]], requires_grad=True)
        labels = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]])
        mask = torch.tensor([[True]*4, [False, True, True, True]])
        loss, n = balanced_support_loss(logits, labels, mask)
        expected = .5*(math.log1p(math.exp(-1))+math.log1p(math.exp(-2)))
        self.assertAlmostEqual(float(loss.detach()), expected, places=6)
        self.assertEqual(int(n), 1)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[1].abs().sum()), 0.)

    def test_all_missing_labels_are_a_differentiable_zero(self):
        logits = torch.full((2, 3), float("nan"), requires_grad=True)
        loss, n = balanced_support_loss(logits, torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.bool))
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(int(n), 0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_two_support_documents_can_both_be_positive(self):
        scores = support_scores([3., -1., 2., 0.], [1, 0, 1, 0], [True]*4)
        self.assertEqual(scores["recall_at_2"], 1.)
        self.assertEqual(scores["both_at_2"], 1.)

    def test_classification_does_not_change_projection_or_need_an_inference_head(self):
        args = dict(hidden_size=8, query_dim=8, memories_per_document=2,
                    hidden_dim=12, attention_dim=8, num_heads=2)
        torch.manual_seed(4)
        plain = SharedDocumentProjector(**args)
        torch.manual_seed(4)
        headed = SharedDocumentProjector(**args, support_head=True)
        z, q = torch.randn(1, 3, 2, 8), torch.randn(1, 5, 8)
        dm = torch.ones(1, 3, dtype=torch.bool)
        a, _ = plain(z, dm, q)
        b, aux = headed(z, dm, q, return_support=True)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(tuple(aux["support_logits"].shape), (1, 3))
        self.assertIsNone(headed(z, dm, q)[1]["support_logits"])
        self.assertEqual(sum(p.numel() for p in headed.parameters())-
                         sum(p.numel() for p in plain.parameters()), 13)


class SupportIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = shared_tests.SharedReaderTests(methodName="runTest")
        self.fixture.setUp()
        self.cfg = copy.deepcopy(self.fixture.cfg)
        self.cfg.readout.support_head = True
        self.cfg.train.support_loss_weight = .1
        self.cfg.revalidate()
        # Both arms use the very same frozen reader, as real published runs do.
        self.stack = self.fixture.stack
        self.model = QuROModel(self.cfg, self.fixture.model.lm, self.stack.tokenizer,
                               build_query_encoder(self.cfg, self.stack), 8, 16)
        self.dataset = QuRODataset(self.cfg.data.train_file, self.stack.tokenizer, self.cfg.data)
        self.dataset.rows = [annotate_fixture(r, self.fixture.cache) for r in self.dataset.rows]
        self.collator = QuROCollator(self.fixture.cache, self.model.pad_id, max_docs=3,
                                     support_policy="visible")
        self.batch = self.collator([self.dataset[0], self.dataset[1]])

    def tearDown(self):
        self.fixture.tearDown()

    def test_auxiliary_gradient_reaches_shared_query_path_at_zero_output_init(self):
        result = self.model.readout_cached(self.batch, return_support=True)
        loss, n = balanced_support_loss(result["aux"]["support_logits"],
                                        self.batch["support_labels"], self.batch["support_loss_mask"])
        self.assertEqual(int(n), 2)
        loss.backward()
        for module in (self.model.readout.to_query, self.model.readout.to_key,
                       self.model.readout.to_value, self.model.readout.context_proj,
                       self.model.readout.memory_proj):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()), 0.)
        self.assertIsNone(self.model.readout.out_proj.weight.grad)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.model.lm.parameters()))

    def test_loss_composition_and_qa_only_control(self):
        output = self.model(self.batch, support_weight=.05)
        self.assertTrue(torch.allclose(output["loss"], output["qa_loss"]+.05*output["support_loss"]))
        output["loss"].backward()
        self.assertTrue(all(p.grad is None for p in self.model.lm.parameters()))
        self.cfg.train.support_loss_weight = 0.
        control = self.model(self.batch)
        self.assertTrue(torch.equal(control["loss"], control["qa_loss"]))
        self.assertNotIn("support_loss", control)

    def test_generation_skips_head_and_labels_cannot_leak_into_prompt(self):
        expected = self.model.generate_answer(self.batch, max_new_tokens=2)
        changed = dict(self.batch)
        changed["support_labels"] = 1-self.batch["support_labels"]
        changed["support_loss_mask"] = ~self.batch["support_loss_mask"]
        with patch.object(self.model.readout.support_classifier, "forward",
                          side_effect=AssertionError("head must not execute during generation")):
            actual = self.model.generate_answer(changed, max_new_tokens=2)
        self.assertEqual(expected, actual)

    def test_real_bf16_mistral_and_both_lora_adapters_stay_frozen_with_auxiliary_loss(self):
        shared_tests.SharedReaderTests.test_real_mistral_and_both_peft_adapters_remain_frozen(self)

    def test_invisible_positive_is_masked_not_relabeled(self):
        self.dataset.rows[0] = annotate_fixture(self.dataset.rows[0], self.fixture.cache, visible=False)
        b = self.collator([self.dataset[0]])
        self.assertEqual(float(b["support_labels"][0, 0]), 1.)
        self.assertFalse(bool(b["support_loss_mask"][0, 0]))
        loss, n = balanced_support_loss(torch.randn_like(b["support_labels"]),
                                        b["support_labels"], b["support_loss_mask"])
        self.assertEqual(int(n), 0)
        self.assertEqual(float(loss), 0.)

    def test_stale_annotation_and_document_order_are_rejected(self):
        row = self.dataset.rows[0]
        with self.assertRaisesRegex(ValueError, "misaligned"):
            support_targets(row, row["retrieved_doc_ids"][::-1], self.collator.support_digest)
        with self.assertRaisesRegex(ValueError, "misaligned"):
            support_targets(row, row["retrieved_doc_ids"], "different-cache")
        raw = dict(row)
        raw.pop("support_annotation")
        with self.assertRaisesRegex(ValueError, "annotate"):
            support_targets(raw, raw["retrieved_doc_ids"], self.collator.support_digest, require=True)

    def test_mismatch_controls_have_correct_targets_and_masks(self):
        item = self.dataset[0]
        item["readout_query"] = "another question"
        b = self.collator([item])
        self.assertTrue(bool(b["support_label_mask"].any()))
        self.assertFalse(bool(b["support_loss_mask"].any()))
        item["retrieved_doc_ids"] = item["retrieved_doc_ids"][::-1]
        b = self.collator([item])
        self.assertFalse(bool(b["support_label_mask"].any()))

    def test_old_checkpoint_requires_explicit_warm_start_and_preserves_logits(self):
        path = os.path.join(self.fixture.tmp.name, "old.pt")
        with torch.no_grad():
            self.fixture.model.readout.out_proj.weight.normal_(std=.03)
        self.fixture.model.save(path)
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["projector_layout"].pop("support_head")  # actual pre-extension format
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "layout"):
            self.model.load(path)
        self.model.load(path, allow_new_support_head=True)
        self.assertTrue(torch.equal(self.fixture.model.readout_cached(self.fixture.batch)["soft_tokens"],
                                    self.model.readout_cached(self.batch)["soft_tokens"]))
        self.assertTrue(torch.equal(self.fixture.model.qa_loss(self.fixture.batch)[0],
                                    self.model.qa_loss(self.batch)[0]))
        headed = os.path.join(self.fixture.tmp.name, "headed.pt")
        self.model.save(headed)
        checkpoint = torch.load(headed, weights_only=False)
        checkpoint["state_dict"].pop("readout.support_classifier.weight")
        torch.save(checkpoint, headed)
        with self.assertRaisesRegex(ValueError, "missing"):
            self.model.load(headed, allow_new_support_head=True)

    def test_trainer_warm_start_and_eval_emit_support_metrics(self):
        from src.train import main
        path = os.path.join(self.fixture.tmp.name, "annotated.jsonl")
        with open(path, "w") as handle:
            for row in self.dataset.rows:
                handle.write(json.dumps(row)+"\n")
        old = os.path.join(self.fixture.tmp.name, "old.pt")
        self.fixture.model.save(old)
        out = os.path.join(self.fixture.tmp.name, "stage2")
        cfg = copy.deepcopy(self.cfg)
        cfg.data.train_file, cfg.data.eval_files = path, {"dev": path}
        argv = ["train", "--preset", "pisco_shared_projector", "--steps", "2", "--batch_size", "2",
                "--grad_accum", "1", "--eval_every", "1", "--eval_every_samples", "2",
                "--eval_max_samples", "2", "--device", "cpu", "--out_dir", out,
                "--resume_from", old, "--warm_start", "--query_control", "--doc_control"]
        with patch("sys.argv", argv), patch("src.train.get_config", return_value=cfg), \
                contextlib.redirect_stdout(io.StringIO()):
            main()
        with open(os.path.join(out, "result.json")) as handle:
            result = json.load(handle)
        normal = result["metrics"]["dev|D0|B=full"]
        self.assertIn("support_recall_at_2", normal)
        with open(os.path.join(out, "train_log.jsonl")) as handle:
            records = [json.loads(line) for line in handle]
        self.assertTrue(any(r.get("support_weight", 0) > 0 for r in records))
        checkpoint = torch.load(os.path.join(out, "checkpoint_last.pt"), weights_only=False)
        self.assertEqual(checkpoint["generator_trainable"], {})
        self.assertIn("readout.support_classifier.weight", checkpoint["state_dict"])
        self.model.load(os.path.join(out, "checkpoint_last.pt"))


class VisibilityTests(unittest.TestCase):
    @staticmethod
    def tokenizer():
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        backend = Tokenizer(models.WordLevel({"[UNK]": 0, "early": 1, "fact": 2}, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                            bos_token="<s>", eos_token="</s>")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<ENC>"]})
        return tokenizer

    def test_real_fast_tokenizer_wrapped_prefix_and_late_sentence(self):
        tokenizer = self.tokenizer()
        row = {"id": "q", "retrieved_doc_ids": ["a", "b", "c"], "gold_ranks": [0, 1],
               "supporting_sentences": [{"doc_rank": 0, "text": "early fact"},
                                         {"doc_rank": 1, "text": "late fact"}]}
        corpus = {"a": "Title: a\nContent: early fact " + " filler"*150,
                  "b": "Title: b\nContent:" + " filler"*150 + " late fact", "c": "other"}
        manifest = {"doc_max_length": 128, "latent_size": 8}
        annotated = annotate_row(row, corpus, tokenizer, manifest)
        self.assertEqual(annotated["support_annotation"]["visible"], [True, False, None])
        ys, lm, sm, vs = support_targets(annotated, row["retrieved_doc_ids"], manifest_digest(manifest))
        self.assertEqual(ys, [1, 1, 0])
        self.assertEqual(sm, [True, False, True])
        original = support_targets(annotated, row["retrieved_doc_ids"], manifest_digest(manifest), "original")
        self.assertEqual(original[2], [True, True, True])

    def test_partial_sentence_is_not_visible(self):
        row = {"id": "q", "retrieved_doc_ids": ["a"], "gold_ranks": [0],
               "supporting_sentences": [{"doc_rank": 0, "text": "near end"}]}
        annotated = annotate_row(row, {"a": "filler "*128+"near end"}, self.tokenizer(),
                                 {"doc_max_length": 128, "latent_size": 8})
        self.assertEqual(annotated["support_annotation"]["visible"], [False])

    def test_offline_annotation_cli_never_loads_an_lm(self):
        from scripts.annotate_support_visibility import main
        with tempfile.TemporaryDirectory() as root:
            base, pisco, cache, out = [os.path.join(root, x) for x in ("base", "pisco", "cache", "out")]
            for directory in (base, pisco, cache):
                os.makedirs(directory)
            self.tokenizer().save_pretrained(base)
            with open(os.path.join(pisco, "config.json"), "w") as handle:
                json.dump({"decoder_model_name": base, "doc_max_length": 128,
                           "compr_rate": 16, "compr_model_name": None}, handle)
            with open(os.path.join(cache, "manifest.json"), "w") as handle:
                json.dump({"compressor": "pisco-mistral:rate16", "doc_max_length": 128,
                           "latent_size": 8, "documents": {"a": {}}}, handle)
            corpus, source = os.path.join(root, "corpus.jsonl"), os.path.join(root, "source.jsonl")
            with open(corpus, "w") as handle:
                handle.write(json.dumps({"doc_id": "a", "text": "early fact"})+"\n")
            with open(source, "w") as handle:
                handle.write(json.dumps({"id": "q", "query": "which fact?", "retrieved_doc_ids": ["a"],
                                         "gold_ranks": [0], "supporting_sentences": [
                                             {"doc_rank": 0, "text": "early fact"}]})+"\n")
            argv = ["annotate", "--input_files", "train="+source, "--corpus", corpus,
                    "--cache_dir", cache, "--pisco_path", pisco, "--out_dir", out]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                    patch("transformers.AutoModelForCausalLM.from_pretrained",
                          side_effect=AssertionError("no LM needed")):
                main()
            with open(os.path.join(out, "train.jsonl")) as handle:
                row = json.loads(handle.readline())
            self.assertEqual(row["support_annotation"]["visible"], [True])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

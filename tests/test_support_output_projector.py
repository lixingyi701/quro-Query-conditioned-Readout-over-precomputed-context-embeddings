"""Output supervision must reach Wo, preserve native E and load explicitly."""
import contextlib
import copy
import io
import json
import os
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_support_projector as support_tests
from config import arm_label
from src.model import QuROModel, build_query_encoder
from src.projector import SharedDocumentProjector
from src.support import balanced_support_loss


class OutputHeadUnitTests(unittest.TestCase):
    def test_head_reads_final_e_preserves_scale_and_masks_nan_padding(self):
        torch.manual_seed(31)
        p = SharedDocumentProjector(8, 8, 2, hidden_dim=12, attention_dim=8,
                                    num_heads=2, support_head=True, support_head_input="output")
        with torch.no_grad():
            p.out_proj.weight.normal_(std=.1)
        z, q = torch.randn(1, 3, 2, 8)*7, torch.randn(1, 5, 8)
        dm = torch.tensor([[True, True, False]])
        z[:, 2] = float("nan")
        ordinary, _ = p(z, dm, q)
        output, aux = p(z, dm, q, return_support=True)
        self.assertTrue(torch.equal(ordinary, output))
        pooled = output.reshape(1, 3, 2, 8).mean(2)
        expected = p.support_classifier(p.support_norm(pooled)).squeeze(-1)
        self.assertTrue(torch.equal(aux["support_logits"][:, :2], expected[:, :2]))
        self.assertEqual(float(aux["support_logits"][0, 2]), 0.)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(sum(x.numel() for x in p.support_classifier.parameters()), 9)
        # Native output is not LN(E); the large input scale survives.
        self.assertGreater(float(output[:, :4].detach().square().mean()), 10.)

    def test_zero_output_updates_wo_then_auxiliary_query_path(self):
        torch.manual_seed(31)
        p = SharedDocumentProjector(8, 8, 2, hidden_dim=12, attention_dim=8,
                                    num_heads=2, support_head=True, support_head_input="output")
        z, q = torch.randn(2, 3, 2, 8, requires_grad=True), torch.randn(2, 5, 8, requires_grad=True)
        dm, labels = torch.ones(2, 3, dtype=torch.bool), torch.tensor([[1, 0, 1], [0, 1, 0]])
        optimizer = torch.optim.AdamW(p.parameters(), lr=.01)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            _, aux = p(z, dm, q, return_support=True)
            loss, _ = balanced_support_loss(aux["support_logits"], labels, dm)
            loss.backward()
            self.assertGreater(float(p.out_proj.weight.grad.abs().sum()), 0.)
            self.assertGreater(float(p.out_proj.bias.grad.abs().sum()), 0.)
            for module in (p.to_query, p.to_key, p.to_value, p.context_proj, p.memory_proj):
                gradient = sum(float(x.grad.abs().sum()) for x in module.parameters())
                self.assertEqual(gradient, 0.) if step == 0 else self.assertGreater(gradient, 0.)
            self.assertIsNone(z.grad)
            self.assertIsNone(q.grad)
            optimizer.step()

    def test_default_source_stays_hidden_and_invalid_sources_fail(self):
        cfg = support_tests.shared_tests.get_config("pisco_shared_projector")
        self.assertEqual(cfg.readout.support_head_input, "hidden")
        cfg.readout.support_head = True
        cfg.readout.support_head_input = "output"
        cfg.revalidate()
        self.assertEqual(arm_label(cfg), "SQ+HeadE")
        cfg.train.support_loss_weight = .1
        self.assertEqual(arm_label(cfg), "SQ+DocE")
        cfg.readout.support_head_input = "invalid"
        with self.assertRaisesRegex(ValueError, "support_head_input"):
            cfg.revalidate()
        with self.assertRaisesRegex(ValueError, "support head input"):
            SharedDocumentProjector(8, 8, 2, support_head_input="invalid")


class OutputHeadIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = support_tests.SupportIntegrationTests(methodName="runTest")
        self.fixture.setUp()
        self.cfg = copy.deepcopy(self.fixture.cfg)
        self.cfg.readout.support_head_input = "output"
        self.cfg.revalidate()
        self.stack = self.fixture.stack
        self.batch = self.fixture.batch
        self.model = QuROModel(self.cfg, self.fixture.model.lm, self.stack.tokenizer,
                               build_query_encoder(self.cfg, self.stack), 8, 16)

    def tearDown(self):
        self.fixture.tearDown()

    def test_warm_started_auxiliary_loss_reaches_wo_and_query_with_frozen_reader(self):
        with torch.no_grad():
            self.model.readout.out_proj.weight.normal_(std=.03)
        result = self.model.readout_cached(self.batch, return_support=True)
        loss, _ = balanced_support_loss(result["aux"]["support_logits"],
                                        self.batch["support_labels"], self.batch["support_loss_mask"])
        loss.backward()
        for module in (self.model.readout.out_proj, self.model.readout.to_query,
                       self.model.readout.to_key, self.model.readout.to_value,
                       self.model.readout.context_proj, self.model.readout.memory_proj):
            grads = [x.grad for x in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.model.lm.parameters()))

    def test_old_sq_warm_start_preserves_e_and_qa_but_hidden_head_is_not_replaced(self):
        plain = self.fixture.fixture.model
        with torch.no_grad():
            plain.readout.out_proj.weight.normal_(std=.03)
        path = os.path.join(self.fixture.fixture.tmp.name, "old-sq.pt")
        plain.save(path)
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["projector_layout"].pop("support_head_input")
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "layout"):
            self.model.load(path)
        optimizer = torch.optim.AdamW(self.model.trainable_parameters())
        with self.assertRaisesRegex(ValueError, "weights-only"):
            self.model.load(path, optimizer=optimizer, allow_new_support_head=True)
        self.model.load(path, allow_new_support_head=True)
        self.assertTrue(torch.equal(plain.readout_cached(self.batch)["soft_tokens"],
                                    self.model.readout_cached(self.batch)["soft_tokens"]))
        self.assertTrue(torch.equal(plain.qa_loss(self.batch)[0], self.model.qa_loss(self.batch)[0]))
        hidden = os.path.join(self.fixture.fixture.tmp.name, "hidden.pt")
        self.fixture.model.save(hidden)
        checkpoint = torch.load(hidden, weights_only=False)
        checkpoint["projector_layout"].pop("support_head_input")
        torch.save(checkpoint, hidden)
        self.fixture.model.load(hidden)  # old headed checkpoints default to hidden
        with self.assertRaisesRegex(ValueError, "layout"):
            self.model.load(hidden, allow_new_support_head=True)

    def test_output_checkpoint_roundtrip_and_missing_head_rejection(self):
        path = os.path.join(self.fixture.fixture.tmp.name, "output.pt")
        with torch.no_grad():
            self.model.readout.out_proj.weight.normal_(std=.03)
        self.model.save(path)
        expected = self.model.readout_cached(self.batch, return_support=True)
        with torch.no_grad():
            self.model.readout.support_classifier.weight.zero_()
        self.model.load(path)
        actual = self.model.readout_cached(self.batch, return_support=True)
        self.assertTrue(torch.equal(expected["aux"]["support_logits"], actual["aux"]["support_logits"]))
        checkpoint = torch.load(path, weights_only=False)
        self.assertEqual(checkpoint["projector_layout"]["support_head_input"], "output")
        checkpoint["state_dict"].pop("readout.support_classifier.weight")
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "missing"):
            self.model.load(path, allow_new_support_head=True)

    def test_generation_and_loss_contracts(self):
        support_tests.SupportIntegrationTests.test_generation_skips_head_and_labels_cannot_leak_into_prompt(self)
        support_tests.SupportIntegrationTests.test_loss_composition_and_qa_only_control(self)

    def test_real_bf16_mistral_and_both_lora_adapters_remain_frozen(self):
        support_tests.shared_tests.SharedReaderTests.test_real_mistral_and_both_peft_adapters_remain_frozen(self)

    def test_matched_trainer_arms_warm_start_and_emit_topk_metrics(self):
        from src.train import main
        root = self.fixture.fixture.tmp.name
        data, old = os.path.join(root, "annotated.jsonl"), os.path.join(root, "plain.pt")
        with open(data, "w") as handle:
            for row in self.fixture.dataset.rows:
                handle.write(json.dumps(row)+"\n")
        self.fixture.fixture.model.save(old)
        starts = []
        for weight in (0., .1):
            cfg = copy.deepcopy(self.cfg)
            cfg.data.train_file, cfg.data.eval_files = data, {"dev": data}
            out = os.path.join(root, "ce" if weight == 0 else "doc")
            argv = ["train", "--preset", "pisco_shared_projector", "--steps", "2", "--batch_size", "2",
                    "--grad_accum", "1", "--eval_every", "1", "--eval_every_samples", "2",
                    "--eval_max_samples", "2", "--device", "cpu", "--out_dir", out,
                    "--resume_from", old, "--warm_start", "--support_head", "--support_head_input", "output",
                    "--support_loss_weight", str(weight)]
            with patch("sys.argv", argv), patch("src.train.get_config", return_value=cfg), \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
            with open(os.path.join(out, "result.json")) as handle:
                result = json.load(handle)
            self.assertEqual(result["support_head_input"], "output")
            self.assertEqual(result["arm"], "SQ+HeadE" if weight == 0 else "SQ+DocE")
            for k in (2, 4, 6):
                self.assertIn(f"support_recall_at_{k}", result["metrics"]["dev|D0|B=full"])
            with open(os.path.join(out, "train_log.jsonl")) as handle:
                records = [json.loads(line) for line in handle]
            starts.append(next(r["validation"]["f1"] for r in records if "validation" in r))
            self.assertTrue(any("output_projection_grad_norm" in r for r in records))
            checkpoint = torch.load(os.path.join(out, "checkpoint_last.pt"), weights_only=False)
            self.assertEqual(checkpoint["generator_trainable"], {})
            self.assertEqual(checkpoint["projector_layout"]["support_head_input"], "output")
            self.model.load(os.path.join(out, "checkpoint_last.pt"))
            with open(os.path.join(out, "predictions_dev_D0_Bfull.json")) as handle:
                rows = json.load(handle)
            self.assertIn("doc_ids", rows[0]["support"])
        self.assertEqual(starts[0], starts[1])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

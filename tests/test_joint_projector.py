"""CPU tests for joint projection, native-reader equivalence and frozen QA training.

Run: python -m unittest discover -s tests -p test_joint_projector.py -v
"""
import copy
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import arm_label, get_config
from src.baselines import PiscoDirectReadout
from src.cache import LatentCache
from src.data import QuROCollator, QuRODataset
from src.model import build_model
from src.projector import JointQueryProjector
from src.prompt import assemble_inputs
from src.train import lr_lambda_factory
from test_shapes import build_workspace


class ProjectorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.p = JointQueryProjector(8, 8, 3, 2, 4, 5)
        self.z = torch.randn(2, 2, 2, 8)
        self.dm = torch.tensor([[True, True], [True, False]])
        self.q = torch.randn(2, 3, 8)
        self.qm = torch.tensor([[True, True, True], [True, True, False]])

    def activate(self):
        with torch.no_grad():
            self.p.out_proj.weight.normal_(std=0.1)
            self.p.out_proj.bias.normal_(std=0.1)

    def test_identity_and_full_budget_with_padding(self):
        out, aux = self.p(self.z, self.dm, self.q, self.qm, budget=6)
        direct, da = PiscoDirectReadout(8, 8)(self.z, self.dm)
        self.assertEqual(tuple(out.shape), (2, 6, 8))
        for i in range(2):
            self.assertTrue(torch.equal(out[i, aux["token_mask"][i]],
                                        direct[i, da["token_mask"][i]]))
        self.assertEqual(aux["token_mask"].sum(1).tolist(), [4, 2])
        self.assertEqual(float(aux["delta_ms"].detach()), 0.0)

    def test_masks_block_document_and_query_padding(self):
        self.activate()
        out, aux = self.p(self.z, self.dm, self.q, self.qm)
        z, q = self.z.clone(), self.q.clone()
        z[1, 1] = float("nan")
        q[1, 2] = float("nan")
        other, _ = self.p(z, self.dm, q, self.qm)
        self.assertTrue(torch.equal(out, other))
        self.assertTrue(torch.isfinite(other).all())
        self.assertEqual(float(other[~aux["token_mask"]].detach().abs().sum()), 0.0)

    def test_dynamic_batch_padding_is_invariant(self):
        self.activate()
        short, _ = self.p(self.z[:1, :1], self.dm[:1, :1], self.q[:1, :2])
        z = torch.cat([self.z[:1, :1], torch.full_like(self.z[:1, :1], float("nan"))], 1)
        q = torch.cat([self.q[:1, :2], torch.full_like(self.q[:1, :1], float("nan"))], 1)
        padded, _ = self.p(z, torch.tensor([[True, False]]), q,
                           torch.tensor([[True, True, False]]))
        self.assertTrue(torch.equal(short, padded))

    def test_query_order_changes_projection(self):
        self.activate()
        a, _ = self.p(self.z[:1], self.dm[:1], self.q[:1])
        b, _ = self.p(self.z[:1], self.dm[:1], self.q[:1].flip(1))
        self.assertGreater(float((a - b).detach().abs().max()), 1e-5)

    def test_one_document_can_change_another_documents_output(self):
        self.activate()
        a, _ = self.p(self.z[:1], self.dm[:1], self.q[:1])
        z = self.z[:1].clone()
        z[:, 1] = torch.randn_like(z[:, 1])
        b, _ = self.p(z, self.dm[:1], self.q[:1])
        self.assertGreater(float((a[:, :2] - b[:, :2]).detach().abs().max()), 1e-5)

    def test_matched_control_is_query_independent(self):
        control = JointQueryProjector(8, 8, 3, 2, 4, 5, "agnostic_matched")
        control.load_state_dict(self.p.state_dict(), strict=False)
        with torch.no_grad():
            control.out_proj.weight.normal_(std=0.1)
        a, _ = control(self.z, self.dm, self.q, self.qm)
        b, _ = control(self.z, self.dm, torch.randn_like(self.q), ~self.qm)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(sum(p.numel() for p in control.parameters()),
                         sum(p.numel() for p in self.p.parameters()))

    def test_feature_sources_are_detached(self):
        self.activate()
        z, q = self.z.requires_grad_(), self.q.requires_grad_()
        out, _ = self.p(z, self.dm, q, self.qm)
        out.square().mean().backward()
        self.assertIsNone(z.grad)
        self.assertIsNone(q.grad)
        self.assertGreater(float(self.p.out_proj.weight.grad.abs().sum()), 0.0)

    def test_incompatible_shapes_budgets_and_empty_inputs_fail(self):
        cases = [
            (self.z, self.dm, self.q, self.qm, 2),
            (self.z[:, :, :1], self.dm, self.q, self.qm, None),
            (self.z, torch.zeros_like(self.dm), self.q, self.qm, None),
            (self.z, self.dm, self.q, torch.zeros_like(self.qm), None),
            (self.z, self.dm, torch.randn(2, 5, 8), None, None),
        ]
        for z, dm, q, qm, budget in cases:
            with self.assertRaises(ValueError):
                self.p(z, dm, q, qm, budget)


class FrozenReaderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.tmp = tempfile.TemporaryDirectory()
        cache_dir, train_file, _, _, _ = build_workspace(self.tmp.name, m=8, h=16)
        cfg = get_config("pisco_joint_projector")
        cfg.generator.kind = "toy"
        cfg.generator.toy_d_model = 16
        cfg.generator.toy_n_layer = 1
        cfg.generator.toy_n_head = 2
        cfg.generator.toy_max_pos = 512
        cfg.data.train_file = train_file
        cfg.data.eval_files = {"dev": train_file}
        cfg.data.cache_dir = cache_dir
        cfg.data.max_docs = 3
        cfg.data.max_query_len = 16
        cfg.readout.max_budget = 24
        cfg.readout.budget_buckets = [24]
        cfg.readout.cache_hidden = 16
        cfg.readout.projector_hidden = 12
        cfg.train.out_dir = self.tmp.name
        cfg.revalidate()
        self.cfg = cfg
        self.stack, self.model = build_model(cfg, 16)
        self.cache = LatentCache(cache_dir)
        self.collator = QuROCollator(self.cache, self.model.pad_id, max_docs=3)
        self.dataset = QuRODataset(train_file, self.stack.tokenizer, cfg.data)
        self.batch = self.collator([self.dataset[0], self.dataset[1]])

    def tearDown(self):
        self.tmp.cleanup()

    def packed(self, result):
        return assemble_inputs(self.model.lm.get_input_embeddings(),
                               self.model.build_prompts(self.batch, result["soft_token_mask"], False),
                               result["soft_tokens"], result["soft_token_mask"],
                               self.batch["target_ids"], self.model.pad_id)

    def test_zero_step_matches_direct_reader_inputs_logits_and_loss(self):
        self.model.eval()
        joint = self.model.readout_cached(self.batch)
        packed = self.packed(joint)
        direct, aux = PiscoDirectReadout(16, 16)(self.batch["cached_latents"],
                                                self.batch["document_mask"])
        direct_packed = self.packed({"soft_tokens": direct, "soft_token_mask": aux["token_mask"]})
        for name in packed:
            self.assertTrue(torch.equal(packed[name], direct_packed[name]), name)
        with torch.no_grad():
            a, b = self.model.lm(**packed), self.model.lm(**direct_packed)
        self.assertTrue(torch.equal(a.logits, b.logits))
        self.assertEqual(float(a.loss), float(b.loss))
        labels = packed["labels"]
        for row in range(labels.size(0)):
            self.assertEqual(labels[row][labels[row] != -100].tolist(), self.batch["target_ids"][row])

    def test_answer_ce_updates_only_projector_and_reaches_first_layer(self):
        before = {name: p.detach().clone() for name, p in self.model.lm.named_parameters()}
        self.model.train()
        self.assertFalse(self.model.lm.training)
        self.assertFalse(self.model.query_encoder.training)
        self.assertTrue(all(name.startswith("readout.") for name, p in self.model.named_parameters()
                            if p.requires_grad))
        optim = torch.optim.AdamW(self.model.trainable_parameters(), lr=1e-2)
        for step in range(3):
            optim.zero_grad(set_to_none=True)
            loss = self.model(self.batch)["loss"]
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(float(self.model.readout.out_proj.weight.grad.abs().sum()), 0.0)
            if step:
                self.assertGreater(float(self.model.readout.memory_proj.weight.grad.abs().sum()), 0.0)
                query_projection = (self.model.readout.context_proj
                                    if hasattr(self.model.readout, "context_proj")
                                    else self.model.readout.query_proj)
                self.assertGreater(float(query_projection.weight.grad.abs().sum()), 0.0)
            self.assertTrue(all(p.grad is None for p in self.model.lm.parameters()))
            optim.step()
        self.assertTrue(all(torch.equal(p, before[name]) for name, p in self.model.lm.named_parameters()))
        generated = self.model.generate_answer(self.batch, max_new_tokens=2)
        self.assertEqual(len(generated), 2)
        self.assertFalse(self.model.lm.training)

    def test_checkpoint_roundtrip_without_reader_weights(self):
        with torch.no_grad():
            self.model.readout.out_proj.weight.normal_(std=0.05)
        path = os.path.join(self.tmp.name, "projector.pt")
        self.model.save(path, step=3)
        ckpt = torch.load(path, weights_only=False)
        self.assertEqual(ckpt["generator_trainable"], {})
        self.assertTrue(all(name.startswith("readout.") for name in ckpt["state_dict"]
                            if name != "budget_selector.buckets"))
        expected = self.model.readout_cached(self.batch)["soft_tokens"].detach().clone()
        with torch.no_grad():
            self.model.readout.out_proj.weight.zero_()
        _, unexpected, step = self.model.load(path)
        self.assertFalse(unexpected)
        self.assertEqual(step, 3)
        self.assertTrue(torch.equal(self.model.readout_cached(self.batch)["soft_tokens"], expected))
        incomplete = copy.deepcopy(ckpt)
        del incomplete["state_dict"]["readout.out_proj.weight"]
        torch.save(incomplete, path)
        with self.assertRaisesRegex(ValueError, "missing trained"):
            self.model.load(path)
        wrong_layout = copy.deepcopy(ckpt)
        wrong_layout["projector_layout"]["max_documents"] = 1
        torch.save(wrong_layout, path)
        with self.assertRaisesRegex(ValueError, "different memory/query layout"):
            self.model.load(path)
        ckpt["generator_trainable"] = {"reader": torch.ones(1)}
        torch.save(ckpt, path)
        with self.assertRaisesRegex(ValueError, "frozen reader"):
            self.model.load(path)

    def test_contextual_option_is_frozen_and_query_only(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.query_encoder.kind = "generator"
        _, model = build_model(cfg, 16)
        encoded = model.encode_query(self.batch["query_ids"], self.batch["query_mask"])
        self.assertFalse(encoded.requires_grad)
        model(self.batch)["loss"].backward()
        self.assertTrue(all(p.grad is None for p in model.lm.parameters()))

    def test_training_entrypoint_validates_step_zero_and_saves_projector(self):
        import src.train as trainer

        cfg = copy.deepcopy(self.cfg)
        cfg.train.steps = 2
        cfg.train.batch_size = 2
        cfg.train.grad_accum = 1
        cfg.train.eval_every = 1
        cfg.train.eval_every_samples = 2
        cfg.train.eval_max_samples = 2
        cfg.train.eval_batch_size = 2
        cfg.train.gen_max_new_tokens = 1
        cfg.train.device = "cpu"
        with patch.object(sys, "argv", ["train", "--preset", "pisco_joint_projector",
                                       "--query_control", "--doc_control"]):
            args = trainer.build_args()
        with patch.object(trainer, "build_args", return_value=args), \
                patch.object(trainer, "get_config", return_value=cfg), \
                contextlib.redirect_stdout(io.StringIO()):
            trainer.main()
        with open(os.path.join(cfg.train.out_dir, "train_log.jsonl")) as f:
            records = [json.loads(line) for line in f]
        self.assertEqual([r["validation"]["step"] for r in records if "validation" in r], [0, 1, 2])
        ckpt = torch.load(os.path.join(cfg.train.out_dir, "checkpoint_last.pt"), weights_only=False)
        self.assertEqual(ckpt["generator_trainable"], {})
        self.assertEqual(len(ckpt["optimizer"]["param_groups"]), 1)
        with open(os.path.join(cfg.train.out_dir, "result.json")) as f:
            result = json.load(f)
        self.assertEqual(result["arm"], arm_label(cfg))
        self.assertEqual(result["query_representation"], self.model.query_encoder.representation)
        label = "full" if cfg.readout.kind == "shared_projector" else "24"
        self.assertIn(f"dev/mismatch-q|D0|B={label}", result["metrics"])

    def test_bad_freeze_and_budget_settings_are_rejected(self):
        for section, name, value in [("generator", "lora_init", "pisco"),
                                     ("train", "budget_dropout", True),
                                     ("data", "prefer_teacher_output", True),
                                     ("decoder", "input_mode", "D2")]:
            cfg = copy.deepcopy(self.cfg)
            setattr(getattr(cfg, section), name, value)
            with self.assertRaises(ValueError):
                cfg.revalidate()
        cfg = copy.deepcopy(self.cfg)
        cfg.readout.max_budget = 8
        cfg.readout.budget_buckets = [8]
        with self.assertRaisesRegex(ValueError, r"K\*m budget"):
            build_model(cfg, 16)

    def test_gold_target_and_query_truncation_are_recorded(self):
        path = os.path.join(self.tmp.name, "teacher-row.jsonl")
        raw = {"query": "What is the code word?", "answers": ["alpha0"],
               "teacher_output": "wrong", "retrieved_doc_ids": self.dataset.rows[0]["retrieved_doc_ids"]}
        with open(path, "w") as f:
            f.write(json.dumps(raw) + "\n")
        gold_dataset = QuRODataset(path, self.stack.tokenizer, self.cfg.data)
        self.assertEqual(gold_dataset.rows[0]["target_source"], "gold")
        self.assertEqual(gold_dataset.rows[0]["target"], "alpha0")
        self.assertEqual(self.dataset[0]["target_ids"][-1], self.stack.tokenizer.eos_token_id)
        self.dataset.cfg.max_query_len = 1
        self.assertTrue(self.dataset[0]["query_truncated"])

    def test_real_mistral_and_both_peft_adapters_remain_frozen(self):
        try:
            from transformers import MistralConfig, MistralForCausalLM
            from peft import LoraConfig, get_peft_model
        except ImportError:
            self.skipTest("requires transformers and peft; no model download is needed")
        from src.generator import _configure_generator_training
        from src.model import FrozenWordEmbeddingQueryEncoder, GeneratorQueryEncoder, QuROModel

        base = MistralForCausalLM(MistralConfig(
            vocab_size=len(self.stack.tokenizer), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=512, pad_token_id=self.model.pad_id))
        base.to(dtype=torch.bfloat16)
        lora = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.1,
                          target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        lm = get_peft_model(base, lora, adapter_name="encoder_adapter")
        lm.add_adapter("decoder_adapter", copy.deepcopy(lora))
        with torch.no_grad():
            for name, p in lm.named_parameters():
                if "lora_B" in name:
                    p.normal_(std=0.05)
        lm.set_adapter("decoder_adapter")
        _configure_generator_training(lm, "frozen", "decoder_adapter")
        encoder = (GeneratorQueryEncoder(lm) if self.cfg.query_encoder.kind == "generator"
                   else FrozenWordEmbeddingQueryEncoder(lm))
        model = QuROModel(self.cfg, lm, self.stack.tokenizer,
                          encoder, 8, 16)
        before = {name: p.detach().clone() for name, p in lm.named_parameters()}
        model.train()
        optim = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
        for _ in range(2):
            optim.zero_grad(set_to_none=True)
            model(self.batch)["loss"].backward()
            self.assertGreater(float(model.readout.out_proj.weight.grad.abs().sum()), 0.0)
            optim.step()
        self.assertEqual(lm.active_adapters, ["decoder_adapter"])
        self.assertFalse(lm.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in lm.parameters()))
        self.assertTrue(all(torch.equal(p, before[name]) for name, p in lm.named_parameters()))


class ConfigTests(unittest.TestCase):
    def test_native_preset_and_control_labels(self):
        cfg = get_config("pisco_joint_projector")
        self.assertEqual(cfg.readout.max_budget, 80)
        self.assertEqual(cfg.train.lr_schedule, "linear")
        self.assertEqual(cfg.generator.lora_init, "frozen")
        self.assertEqual(arm_label(cfg), "JQ")
        cfg.readout.projector_query_mode = "agnostic_matched"
        self.assertEqual(arm_label(cfg), "J0m")

    def test_linear_schedule_and_historical_default(self):
        linear = lr_lambda_factory(100, 0.1, "linear")
        self.assertEqual(linear(0), 0.1)
        self.assertEqual(linear(10), 1.0)
        self.assertAlmostEqual(linear(55), 0.5)
        self.assertEqual(linear(100), 0.0)
        self.assertEqual(lr_lambda_factory(100, 0.1)(55), 0.5)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

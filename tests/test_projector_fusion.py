"""Matched gamma fusion: identity starts, answer gradients and strict reloads."""
import contextlib
import copy
import io
import json
import math
import os
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_support_output_projector as output_tests
from config import arm_label, get_config
from src.fusion_metrics import add_fusion_stats, fusion_metrics
from src.model import QuROModel, build_query_encoder
from src.projector import SharedDocumentProjector


class FusionUnitTests(unittest.TestCase):
    def make(self, fusion="none", **kwargs):
        torch.manual_seed(42)
        return SharedDocumentProjector(8, 8, 2, hidden_dim=12, attention_dim=8,
                                       num_heads=2, support_head=True,
                                       support_head_input="output", fusion=fusion, **kwargs)

    def test_legacy_init_and_head_rng_preserved_with_equal_arm_capacity(self):
        old, add, film = self.make(), self.make("additive"), self.make("film")
        self.assertIsNone(old.gamma_proj)
        for name, tensor in old.state_dict().items():
            self.assertTrue(torch.equal(tensor, add.state_dict()[name]), name)
            self.assertTrue(torch.equal(tensor, film.state_dict()[name]), name)
        self.assertEqual(set(old.state_dict()), set(add.state_dict()) -
                         {"gamma_proj.weight", "gamma_proj.bias"})
        self.assertEqual(sum(p.numel() for p in add.parameters()),
                         sum(p.numel() for p in film.parameters()))
        self.assertEqual(sum(p.numel() for p in add.parameters())-
                         sum(p.numel() for p in old.parameters()), 12*12+12)

    def test_zero_gamma_matches_nonzero_sq_output_and_classifier_exactly(self):
        old = self.make()
        with torch.no_grad():
            old.out_proj.weight.normal_(std=.2)
            old.out_proj.bias.normal_(std=.1)
        z, q = torch.randn(2, 3, 2, 8), torch.randn(2, 7, 8)
        dm = torch.tensor([[True, True, False], [True, True, True]])
        expected, ea = old(z, dm, q, return_support=True)
        for mode in ("additive", "film"):
            p = self.make(mode)
            p.load_state_dict(old.state_dict(), strict=False)
            actual, aa = p(z, dm, q, return_support=True)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(aa["support_logits"], ea["support_logits"]))
            self.assertEqual(fusion_metrics(aa["fusion_stats"])["fusion_gamma_rms"], 0.)

    def test_active_gamma_respects_padding_permutation_and_variable_lengths(self):
        for mode in ("additive", "film"):
            p = self.make(mode)
            with torch.no_grad():
                p.out_proj.weight.normal_(std=.1)
                p.gamma_proj.weight.normal_(std=.1)
                p.gamma_proj.bias.fill_(.03)
            for k, t in ((2, 7), (10, 80)):
                z, q = torch.randn(1, k, 2, 8), torch.randn(1, t, 8)
                dm = torch.ones(1, k, dtype=torch.bool)
                a, aux = p(z, dm, q)
                self.assertEqual(tuple(a.shape), (1, 2*k, 8))
                perm = torch.arange(k-1, -1, -1)
                b, _ = p(z[:, perm], dm, q)
                self.assertTrue(torch.allclose(b.reshape(1, k, 2, 8),
                                              a.reshape(1, k, 2, 8)[:, perm], atol=1e-6))
                padded_z = torch.cat([z, torch.full_like(z[:, :1], float("nan"))], 1)
                padded_q = torch.cat([q, torch.full_like(q[:, :2], float("nan"))], 1)
                padded_dm = torch.cat([dm, torch.zeros(1, 1, dtype=torch.bool)], 1)
                qm = torch.cat([torch.ones(1, t, dtype=torch.bool),
                                torch.zeros(1, 2, dtype=torch.bool)], 1)
                c, ca = p(padded_z, padded_dm, padded_q, qm)
                self.assertTrue(torch.allclose(c[:, :2*k], a, atol=1e-6))
                self.assertEqual(float(c[:, 2*k:].detach().abs().sum()), 0.)
                for key, value in aux["fusion_stats"].items():
                    self.assertTrue(torch.allclose(value, ca["fusion_stats"][key], atol=1e-5))
                self.assertTrue(all(not v.requires_grad for v in ca["fusion_stats"].values()))

    def test_rms_statistics_pool_elements_not_microbatch_rms_and_zero_ratio(self):
        totals = {}
        for n, h, b in ((1, 1., 0.), (3, 3., 0.)):
            add_fusion_stats(totals, {"elements": n, "h_sum_sq": n*h*h,
                                    "b_sum_sq": b, "gamma_sum_sq": 0.,
                                    "product_sum_sq": 0., "update_sum_sq": 0.})
        values = fusion_metrics(totals)
        self.assertAlmostEqual(values["fusion_h_rms"], math.sqrt(7.))
        self.assertIsNone(values["fusion_product_over_b_rms"])
        json.dumps(values, allow_nan=False)
        with self.assertRaises(ValueError):
            fusion_metrics({**totals, "elements": 0})

    def test_config_cli_and_arm_labels_distinguish_legacy_from_additive(self):
        from src.train import apply_overrides, build_args
        self.assertEqual(get_config("pisco_shared_projector").readout.projector_fusion, "none")
        for mode, suffix in (("additive", "+AddG"), ("film", "+FiLM")):
            with patch("sys.argv", ["train", "--projector_fusion", mode,
                                    "--support_head_input", "output", "--support_loss_weight", ".1"]):
                cfg = apply_overrides(get_config("pisco_shared_projector"), build_args())
            self.assertEqual(arm_label(cfg), "SQ+DocE"+suffix)
            self.assertIn("fusion="+mode, cfg.summary())
        cfg = get_config("pisco_joint_projector")
        cfg.readout.projector_fusion = "film"
        with self.assertRaisesRegex(ValueError, "shared_projector"):
            cfg.revalidate()
        with self.assertRaisesRegex(ValueError, "fusion"):
            self.make("invalid")


class FusionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = output_tests.OutputHeadIntegrationTests(methodName="runTest")
        self.fixture.setUp()
        self.cfg = copy.deepcopy(self.fixture.cfg)
        self.stack, self.batch = self.fixture.stack, self.fixture.batch
        self.root = self.fixture.fixture.fixture.tmp.name
        self.sq = self.fixture.fixture.fixture.model
        with torch.no_grad():
            self.sq.readout.out_proj.weight.normal_(std=.03)
        self.old = os.path.join(self.root, "sq.pt")
        self.sq.save(self.old)

    def tearDown(self):
        self.fixture.tearDown()

    def make(self, mode):
        cfg = copy.deepcopy(self.cfg)
        cfg.readout.projector_fusion = mode
        cfg.revalidate()
        torch.manual_seed(42)
        return QuROModel(cfg, self.sq.lm, self.stack.tokenizer,
                         build_query_encoder(cfg, self.stack), 8, 16)

    def warm(self, model, path=None):
        model.load(path or self.old, allow_new_support_head=True,
                   allow_new_projector_fusion=True)

    def test_legacy_warm_start_e_and_ce_identity_even_after_gamma_was_modified(self):
        ckpt = torch.load(self.old, weights_only=False)
        ckpt["config"]["readout"].pop("projector_fusion")
        ckpt["projector_layout"].pop("fusion")
        torch.save(ckpt, self.old)
        for mode in ("additive", "film"):
            model = self.make(mode)
            with torch.no_grad():
                model.readout.gamma_proj.weight.fill_(.2)
                model.readout.gamma_proj.bias.fill_(.2)
            self.warm(model)
            self.assertTrue(torch.equal(self.sq.readout_cached(self.batch)["soft_tokens"],
                                        model.readout_cached(self.batch)["soft_tokens"]))
            self.assertTrue(torch.equal(self.sq.qa_loss(self.batch)[0], model.qa_loss(self.batch)[0]))

    def test_answer_ce_updates_zero_gamma_and_keeps_source_and_reader_frozen(self):
        for mode in ("additive", "film"):
            model = self.make(mode)
            self.warm(model)
            batch = dict(self.batch)
            batch["cached_latents"] = batch["cached_latents"].clone().requires_grad_()
            before = model.readout.gamma_proj.weight.detach().clone()
            optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=2e-5)
            loss, _ = model.qa_loss(batch)
            loss.backward()
            for name, p in model.readout.gamma_proj.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(float(p.grad.abs().sum()), 0.)
            self.assertIsNone(batch["cached_latents"].grad)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.lm.parameters()))
            optimizer.step()
            self.assertFalse(torch.equal(before, model.readout.gamma_proj.weight))
            self.assertTrue(all(p.requires_grad for p in model.readout.parameters()))

    def test_new_fusion_requires_explicit_weights_only_warm_start(self):
        model = self.make("film")
        with self.assertRaisesRegex(ValueError, "layout"):
            model.load(self.old, allow_new_support_head=True)
        with self.assertRaisesRegex(ValueError, "weights-only"):
            self.warm_with_optimizer(model)
        with self.assertRaisesRegex(ValueError, "weights-only"):
            model.load(self.old, scheduler=object(), allow_new_projector_fusion=True)
        self.warm(model)

    def warm_with_optimizer(self, model):
        optimizer = torch.optim.AdamW(model.trainable_parameters())
        model.load(self.old, optimizer=optimizer, allow_new_support_head=True,
                   allow_new_projector_fusion=True)

    def test_trained_fusion_roundtrip_resume_and_partial_missing_weights_rejected(self):
        for mode in ("additive", "film"):
            model = self.make(mode)
            self.warm(model)
            optimizer = torch.optim.AdamW(model.trainable_parameters())
            model(self.batch)["loss"].backward()
            optimizer.step()
            path = os.path.join(self.root, mode+".pt")
            model.save(path, optimizer=optimizer, step=1)
            expected = model.readout_cached(self.batch)["soft_tokens"]
            rebuilt = self.make(mode)
            reoptim = torch.optim.AdamW(rebuilt.trainable_parameters())
            _, unexpected, step = rebuilt.load(path, optimizer=reoptim)
            self.assertEqual(step, 1)
            self.assertFalse(unexpected)
            self.assertTrue(torch.equal(expected, rebuilt.readout_cached(self.batch)["soft_tokens"]))
            ckpt = torch.load(path, weights_only=False)
            ckpt["state_dict"].pop("readout.gamma_proj.bias")
            torch.save(ckpt, path)
            with self.assertRaisesRegex(ValueError, "missing"):
                self.warm(rebuilt, path)

    def test_modes_cannot_be_silently_switched_or_gamma_disguised_as_legacy(self):
        add, film = self.make("additive"), self.make("film")
        self.warm(add)
        path = os.path.join(self.root, "add.pt")
        add.save(path)
        with self.assertRaisesRegex(ValueError, "layout"):
            self.warm(film, path)
        ckpt = torch.load(path, weights_only=False)
        ckpt["config"]["readout"]["projector_fusion"] = "none"
        ckpt["projector_layout"]["fusion"] = "none"
        torch.save(ckpt, path)
        with self.assertRaisesRegex(ValueError, "unexpectedly contains gamma"):
            self.warm(film, path)

    def test_frozen_bf16_mistral_and_both_peft_adapters_with_each_fusion(self):
        for mode in ("additive", "film"):
            self.fixture.cfg.readout.projector_fusion = mode
            output_tests.OutputHeadIntegrationTests.test_real_bf16_mistral_and_both_lora_adapters_remain_frozen(
                self.fixture)

    def test_trainer_startup_diagnostics_controls_and_saved_fusion(self):
        from src.train import main
        data = os.path.join(self.root, "annotated.jsonl")
        with open(data, "w") as handle:
            for row in self.fixture.fixture.dataset.rows:
                handle.write(json.dumps(row)+"\n")
        starts = []
        for mode in ("additive", "film"):
            cfg = copy.deepcopy(self.cfg)
            cfg.data.train_file, cfg.data.eval_files = data, {"dev": data}
            cfg.train.eval_batch_size, cfg.train.gen_max_new_tokens = 2, 1
            cfg.train.log_every = 1
            out = os.path.join(self.root, mode)
            argv = ["train", "--preset", "pisco_shared_projector", "--projector_fusion", mode,
                    "--steps", "2", "--batch_size", "2", "--grad_accum", "2", "--lr", "2e-5",
                    "--eval_every", "1", "--eval_every_samples", "2", "--eval_max_samples", "2",
                    "--device", "cpu", "--out_dir", out, "--resume_from", self.old, "--warm_start",
                    "--support_head", "--support_head_input", "output", "--support_loss_weight", ".1",
                    "--query_control"]
            with patch("sys.argv", argv), patch("src.train.get_config", return_value=cfg), \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
            with open(os.path.join(out, "result.json")) as handle:
                result = json.load(handle)
            self.assertEqual(result["projector_fusion"], mode)
            self.assertIn("dev/mismatch-q|D0|B=full", result["metrics"])
            self.assertIn("fusion_gamma_rms", result["metrics"]["dev|D0|B=full"])
            with open(os.path.join(out, "train_log.jsonl")) as handle:
                records = [json.loads(line) for line in handle]
            startup = next(r["validation"] for r in records if "validation" in r)
            starts.append(startup)
            self.assertEqual(startup["fusion_gamma_rms"], 0.)
            steps = [r for r in records if "validation" not in r]
            self.assertGreater(steps[0]["gamma_projection_grad_norm"], 0.)
            self.assertGreater(steps[-1]["fusion_gamma_rms"], 0.)
            ckpt = torch.load(os.path.join(out, "checkpoint_last.pt"), weights_only=False)
            self.assertEqual(ckpt["projector_layout"]["fusion"], mode)
            self.assertEqual(ckpt["generator_trainable"], {})
        self.assertEqual(starts[0]["f1"], starts[1]["f1"])
        self.assertEqual(starts[0]["fusion_h_rms"], starts[1]["fusion_h_rms"])
        self.assertEqual(starts[0]["fusion_b_rms"], starts[1]["fusion_b_rms"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

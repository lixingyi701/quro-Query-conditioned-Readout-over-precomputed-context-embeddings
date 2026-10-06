"""Frozen-gamma control arm and evaluation-only gamma interventions."""
import contextlib
import copy
import io
import json
import os
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_projector_fusion as fusion_tests
from config import arm_label, get_config
from src.fusion_metrics import add_fusion_stats, paired_change_metrics, paired_change_stats
from src.projector import SharedDocumentProjector


class GammaInterventionUnitTests(unittest.TestCase):
    def make(self, fusion):
        torch.manual_seed(42)
        p = SharedDocumentProjector(8, 8, 2, hidden_dim=12, attention_dim=8, num_heads=2,
                                    support_head=True, support_head_input="output", fusion=fusion)
        if p.gamma_proj is not None:
            with torch.no_grad():
                p.out_proj.weight.normal_(std=.2)
                p.gamma_proj.weight.normal_(std=.3)
                p.gamma_proj.bias.fill_(.05)
        return p

    def inputs(self):
        torch.manual_seed(0)
        z = torch.randn(2, 3, 2, 8)
        dm = torch.tensor([[True, True, False], [True, True, True]])
        q, other = torch.randn(2, 5, 8), torch.randn(2, 7, 8)
        other_mask = torch.ones(2, 7, dtype=torch.bool)
        other_mask[0, 5:] = False
        return z, dm, q, other, other_mask

    def test_zero_gamma_equals_zeroed_weights_and_same_question_swap_is_identity(self):
        z, dm, q, other, om = self.inputs()
        for mode in ("additive", "film"):
            p = self.make(mode)
            normal, _ = p(z, dm, q)
            same, _ = p(z, dm, q, gamma_query_emb=q)
            self.assertTrue(torch.allclose(normal, same, atol=1e-6))
            zero, za = p(z, dm, q, gamma_zero=True)
            self.assertEqual(float(za["gamma"].abs().sum()), 0.)
            reference = copy.deepcopy(p)
            with torch.no_grad():
                reference.gamma_proj.weight.zero_()
                reference.gamma_proj.bias.zero_()
            self.assertTrue(torch.allclose(zero, reference(z, dm, q)[0], atol=1e-6))
            self.assertFalse(torch.allclose(zero, normal, atol=1e-4))

    def test_swap_takes_gamma_from_other_question_but_keeps_h_and_b(self):
        z, dm, q, other, om = self.inputs()
        for mode in ("additive", "film"):
            p = self.make(mode)
            swapped, sa = p(z, dm, q, gamma_query_emb=other, gamma_query_mask=om)
            _, oa = p(z, dm, other, om)
            self.assertTrue(torch.allclose(sa["gamma"], oa["gamma"], atol=1e-6))
            memory = torch.where(dm[:, :, None, None], z, 0.)
            zn = p.memory_norm(memory)
            h = p.memory_proj(zn.flatten(2))
            b = p._query_context(zn, dm, q, None)[0]
            update = oa["gamma"] * h if mode == "film" else oa["gamma"]
            delta = p.out_proj(F.gelu(h + b + update)).reshape(2, 3, 2, 8)
            expected = (memory + torch.where(dm[:, :, None, None], delta, 0.)).reshape(2, 6, 8)
            self.assertTrue(torch.allclose(swapped, expected, atol=1e-5))

    def test_invalid_interventions_fail(self):
        z, dm, q, other, om = self.inputs()
        with self.assertRaisesRegex(ValueError, "additive/film"):
            self.make("none")(z, dm, q, gamma_zero=True)
        with self.assertRaisesRegex(ValueError, "either"):
            self.make("film")(z, dm, q, gamma_query_emb=other, gamma_zero=True)

    def test_paired_change_metrics_pool_and_null_reference(self):
        z, dm, q, other, om = self.inputs()
        p = self.make("film")
        _, normal = p(z, dm, q)
        _, zero = p(z, dm, q, gamma_zero=True)
        memory = torch.where(dm[:, :, None, None], z, 0.)
        same = paired_change_metrics(paired_change_stats(normal, normal, memory, dm))
        self.assertEqual(same["change_gamma_rel"], 0.)
        self.assertEqual(same["change_e_over_delta_ref"], 0.)
        self.assertAlmostEqual(same["change_gamma_cosine"], 1., places=5)
        totals = {}
        for _ in range(2):
            add_fusion_stats(totals, paired_change_stats(zero, normal, memory, dm))
        values = paired_change_metrics(totals)
        self.assertAlmostEqual(values["change_gamma_rel"], 1., places=6)
        self.assertIsNone(values["change_gamma_cosine"])
        self.assertEqual(values["change_documents"], 2*int(dm.sum()))
        self.assertGreater(values["change_e_over_memory"], 0.)
        reverse = paired_change_metrics(paired_change_stats(normal, zero, memory, dm))
        self.assertIsNone(reverse["change_gamma_rel"])
        json.dumps(values, allow_nan=False)

    def test_config_label_and_frozen_requires_gamma_module(self):
        from src.train import apply_overrides, build_args
        with patch("sys.argv", ["train", "--projector_fusion", "additive", "--projector_gamma_frozen",
                                "--support_head", "--support_head_input", "output"]):
            cfg = apply_overrides(get_config("pisco_shared_projector"), build_args())
        self.assertTrue(cfg.readout.projector_gamma_frozen)
        self.assertEqual(arm_label(cfg), "SQ+HeadE+G0")
        self.assertIn("fusion=additive(frozen)", cfg.summary())
        cfg = get_config("pisco_shared_projector")
        cfg.readout.projector_gamma_frozen = True
        with self.assertRaisesRegex(ValueError, "gamma module"):
            cfg.revalidate()
        with patch("sys.argv", ["train", "--gamma_control"]), self.assertRaisesRegex(ValueError, "fusion"):
            apply_overrides(get_config("pisco_shared_projector"), build_args())

    def test_drop_frozen_gradients_clears_grads_and_detects_drift(self):
        from src.train import drop_frozen_gamma_gradients
        p = self.make("additive")
        with torch.no_grad():
            p.gamma_proj.weight.zero_()
            p.gamma_proj.bias.zero_()
        z, dm, q, _, _ = self.inputs()
        p(z, dm, q)[0].square().sum().backward()
        self.assertIsNotNone(p.gamma_proj.weight.grad)
        drop_frozen_gamma_gradients(p)
        self.assertTrue(all(x.grad is None for x in p.gamma_proj.parameters()))
        with torch.no_grad():
            p.gamma_proj.bias.fill_(1e-3)
        with self.assertRaisesRegex(RuntimeError, "nonzero"):
            drop_frozen_gamma_gradients(p)


class GammaControlTrainerTests(unittest.TestCase):
    """Reuses the fusion fixture: tiny frozen reader, old SQ checkpoint, annotated rows."""

    def setUp(self):
        self.base = fusion_tests.FusionIntegrationTests(methodName="runTest")
        self.base.setUp()
        self.fixture, self.cfg = self.base.fixture, self.base.cfg
        self.root, self.old = self.base.root, self.base.old

    def tearDown(self):
        self.base.tearDown()

    def run_arm(self, name, extra):
        from src.train import main
        data = os.path.join(self.root, "annotated.jsonl")
        with open(data, "w") as handle:
            for row in self.fixture.fixture.dataset.rows:
                handle.write(json.dumps(row)+"\n")
        cfg = copy.deepcopy(self.cfg)
        cfg.data.train_file, cfg.data.eval_files = data, {"dev": data}
        cfg.train.eval_batch_size, cfg.train.gen_max_new_tokens = 2, 1
        cfg.train.log_every = 1
        out = os.path.join(self.root, name)
        argv = ["train", "--preset", "pisco_shared_projector", "--projector_fusion", "additive",
                "--steps", "3", "--batch_size", "2", "--grad_accum", "2", "--lr", "1e-2",
                "--eval_every", "100", "--eval_every_samples", "2", "--eval_max_samples", "2",
                "--device", "cpu", "--out_dir", out, "--resume_from", self.old, "--warm_start",
                "--support_head", "--support_head_input", "output", "--support_loss_weight", "0",
                "--query_control", "--gamma_control", *extra]
        with patch("sys.argv", argv), patch("src.train.get_config", return_value=cfg), \
                contextlib.redirect_stdout(io.StringIO()):
            main()
        with open(os.path.join(out, "result.json")) as handle:
            result = json.load(handle)
        with open(os.path.join(out, "train_log.jsonl")) as handle:
            steps = [json.loads(line) for line in handle if "validation" not in line]
        ckpt = torch.load(os.path.join(out, "checkpoint_last.pt"), weights_only=False)
        return result, steps, ckpt

    def test_frozen_arm_matches_data_order_keeps_gamma_zero_and_emits_interventions(self):
        frozen, frozen_steps, frozen_ckpt = self.run_arm("frozen", ["--projector_gamma_frozen"])
        active, active_steps, active_ckpt = self.run_arm("active", [])
        # Identical first forward: same RNG use, same rows, gamma zero in both.
        self.assertEqual(frozen_steps[0]["qa_loss"], active_steps[0]["qa_loss"])
        self.assertTrue(frozen["projector_gamma_frozen"])
        self.assertFalse(active["projector_gamma_frozen"])
        for key in ("readout.gamma_proj.weight", "readout.gamma_proj.bias"):
            self.assertEqual(float(frozen_ckpt["state_dict"][key].abs().sum()), 0.)
        self.assertGreater(float(active_ckpt["state_dict"]["readout.gamma_proj.weight"].abs().sum()), 0.)
        self.assertTrue(all(s["gamma_projection_grad_norm"] is None for s in frozen_steps))
        self.assertTrue(all(s["fusion_gamma_rms"] == 0. for s in frozen_steps))
        for result in (frozen, active):
            metrics = result["metrics"]
            for name in ("dev/gamma-zero", "dev/gamma-swap", "dev/mismatch-q"):
                self.assertIn("change_e_over_delta_ref", metrics[name+"|D0|B=full"])
            self.assertNotIn("change_documents", metrics["dev|D0|B=full"])
        # With gamma frozen at zero, zeroing it changes nothing.
        zero = frozen["metrics"]["dev/gamma-zero|D0|B=full"]
        self.assertEqual(zero["change_e_over_delta_ref"], 0.)
        self.assertEqual(zero["f1"], frozen["metrics"]["dev|D0|B=full"]["f1"])
        active_zero = active["metrics"]["dev/gamma-zero|D0|B=full"]
        self.assertAlmostEqual(active_zero["change_gamma_rel"], 1., places=6)
        self.assertGreater(active_zero["change_e_over_delta_ref"], 0.)
        with open(os.path.join(self.root, "active", "predictions_dev_gamma-swap_D0_Bfull.json")) as h:
            rows = json.load(h)
        self.assertTrue(all("gamma_query" in r and r["gamma_query"] != r["query"] for r in rows))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

"""S0: a separately trainable document MLP, with no retained condition branch."""
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
import test_shared_projector as shared_tests
import test_support_projector as support_tests
from config import arm_label, get_config
from src.model import QuROModel, build_query_encoder
from src.projector import SharedDocumentProjector


class NoQueryProjectionTests(unittest.TestCase):
    def make(self, mode="none", **kwargs):
        return SharedDocumentProjector(8, 8, 2, hidden_dim=12, attention_dim=8,
                                       num_heads=2, query_mode=mode, **kwargs)

    def test_shared_initial_weights_and_subsequent_rng_match_sq(self):
        for cross in (False, True):
            for head_input in ("hidden", "output"):
                models, next_draws = [], []
                for mode in ("conditioned", "none"):
                    torch.manual_seed(42)
                    models.append(self.make(mode, cross_document=cross, support_head=True,
                                            support_head_input=head_input))
                    next_draws.append(torch.randn(7))
                sq, s0 = models
                for name, value in s0.state_dict().items():
                    self.assertTrue(torch.equal(value, sq.state_dict()[name]), name)
                self.assertTrue(torch.equal(*next_draws))
                self.assertFalse(s0.needs_query)
                self.assertIsNone(s0.fixed_query)
                for name in ("to_query", "to_key", "to_value", "context_proj",
                             "query_norm", "context_norm", "gamma_proj"):
                    self.assertIsNone(getattr(s0, name), name)

    def test_only_document_mlp_parameters_remain_and_counts_are_exact(self):
        p = self.make()
        self.assertEqual(set(dict(p.named_parameters())),
                         {"memory_proj.weight", "memory_proj.bias", "out_proj.weight", "out_proj.bias"})
        self.assertEqual(sum(x.numel() for x in p.parameters()), 412)
        # Check the published configuration without allocating 38M real weights.
        with torch.device("meta"):
            sq = SharedDocumentProjector(4096, 4096, 8)
            s0 = SharedDocumentProjector(4096, 4096, 8, query_mode="none")
        self.assertEqual(sum(x.numel() for x in sq.parameters()), 37_782_016)
        self.assertEqual(sum(x.numel() for x in s0.parameters()), 33_587_712)

    def test_active_projection_matches_document_only_formula_and_padding(self):
        p = self.make()
        with torch.no_grad():
            p.out_proj.weight.normal_(std=.1)
            p.out_proj.bias.normal_(std=.1)
        z = torch.randn(2, 3, 2, 8)
        dm = torch.tensor([[True, True, False], [False, True, True]])
        z[~dm] = float("nan")
        memory = torch.where(dm[:, :, None, None], z, 0.)
        delta = p.out_proj(F.gelu(p.memory_proj(p.memory_norm(memory).flatten(2)))).reshape_as(z)
        expected = (memory + torch.where(dm[:, :, None, None], delta, 0.)).flatten(1, 2)
        with patch.object(p, "_query_context", side_effect=AssertionError("query path executed")):
            actual, aux = p(z, dm, return_attn=True)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertIsNone(aux["query_attention"])
        self.assertEqual(float(aux["fusion_stats"]["b_sum_sq"]), 0.)

    def test_query_content_length_and_masks_are_ignored_after_training(self):
        p = self.make(query_position=True, support_head=True)
        with torch.no_grad():
            p.out_proj.weight.normal_(std=.1)
        z, dm = torch.randn(2, 3, 2, 8), torch.ones(2, 3, dtype=torch.bool)
        a, aa = p(z, dm, return_support=True)
        b, ba = p(z, dm, torch.full((2, 80, 8), float("nan")),
                  torch.zeros(2, 80, dtype=torch.bool), return_support=True)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(aa["support_logits"], ba["support_logits"]))

    def test_variable_k_and_document_permutation_with_optional_mixing(self):
        for cross in (False, True):
            p = self.make(cross_document=cross)
            with torch.no_grad():
                p.out_proj.weight.normal_(std=.1)
            for k in (2, 10):
                z, dm = torch.randn(1, k, 2, 8), torch.ones(1, k, dtype=torch.bool)
                a, _ = p(z, dm)
                b, _ = p(z.flip(1), dm)
                self.assertEqual(tuple(a.shape), (1, k*2, 8))
                self.assertTrue(torch.allclose(b.reshape_as(z), a.reshape_as(z).flip(1), atol=1e-6))

    def test_cli_labels_and_invalid_combinations(self):
        from src.train import apply_overrides, build_args
        with patch("sys.argv", ["train", "--preset", "pisco_shared_projector",
                                "--projector_query_mode", "none"]):
            cfg = apply_overrides(get_config("pisco_shared_projector"), build_args())
        self.assertEqual(arm_label(cfg), "S0")
        cfg.readout.projector_cross_document = True
        self.assertEqual(arm_label(cfg), "S0X")
        for fusion in ("additive", "film"):
            with self.assertRaisesRegex(ValueError, "gamma branches"):
                self.make(fusion=fusion)
            bad = copy.deepcopy(cfg)
            bad.readout.projector_fusion = fusion
            with self.assertRaisesRegex(ValueError, "gamma branches"):
                bad.revalidate()
        bad = get_config("pisco_joint_projector")
        bad.readout.projector_query_mode = "none"
        with self.assertRaisesRegex(ValueError, "shared_projector"):
            bad.revalidate()


class NoQueryReaderTests(unittest.TestCase):
    def setUp(self):
        self.fixture = shared_tests.SharedReaderTests(methodName="runTest")
        self.fixture.setUp()
        self.cfg = copy.deepcopy(self.fixture.cfg)
        self.cfg.readout.projector_query_mode = "none"
        self.cfg.revalidate()
        self.model = QuROModel(self.cfg, self.fixture.model.lm, self.fixture.stack.tokenizer,
                               build_query_encoder(self.cfg, self.fixture.stack), 8, 16)
        self.batch = self.fixture.batch
        self.root = self.fixture.tmp.name

    def tearDown(self):
        self.fixture.tearDown()

    def test_zero_step_matches_sq_inputs_and_qa_without_query_encoder_forward(self):
        with patch.object(self.model.query_encoder, "forward",
                          side_effect=AssertionError("query encoder executed")):
            a = self.model.readout_cached(self.batch)
            loss, _ = self.model.qa_loss(self.batch)
        b = self.fixture.model.readout_cached(self.batch)
        self.assertTrue(torch.equal(a["soft_tokens"], b["soft_tokens"]))
        self.assertTrue(torch.equal(loss, self.fixture.model.qa_loss(self.batch)[0]))
        # The decoder keeps the same correct question and the same memory slots.
        self.assertEqual(self.model.build_prompts(self.batch, a["soft_token_mask"], False),
                         self.fixture.model.build_prompts(self.batch, b["soft_token_mask"], False))

    def test_ce_trains_document_mlp_with_frozen_cache_and_reader(self):
        batch = dict(self.batch)
        batch["cached_latents"] = batch["cached_latents"].clone().requires_grad_()
        before = {name: p.detach().clone() for name, p in self.model.lm.named_parameters()}
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.trainable_parameters(), lr=1e-2)
        with patch.object(self.model.query_encoder, "forward",
                          side_effect=AssertionError("query encoder executed")):
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss, _ = self.model.qa_loss(batch)
                loss.backward()
                for module in (self.model.readout.out_proj, self.model.readout.memory_proj):
                    if step or module is self.model.readout.out_proj:
                        self.assertGreater(float(module.weight.grad.abs().sum()), 0.)
                optimizer.step()
        self.assertIsNone(batch["cached_latents"].grad)
        self.assertTrue(all(not p.requires_grad and p.grad is None and torch.equal(p, before[name])
                            for name, p in self.model.lm.named_parameters()))
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(optimizer_ids, {id(p) for p in self.model.readout.parameters()})

    def test_checkpoint_roundtrip_and_resume_keep_branch_absent(self):
        optimizer = torch.optim.AdamW(self.model.trainable_parameters(), lr=1e-2)
        self.model.qa_loss(self.batch)[0].backward()
        optimizer.step()
        path = os.path.join(self.root, "s0.pt")
        self.model.save(path, optimizer=optimizer, step=1)
        expected = self.model.readout_cached(self.batch)["soft_tokens"].detach().clone()
        with torch.no_grad():
            self.model.readout.out_proj.weight.zero_()
        missing, unexpected, step = self.model.load(path, optimizer=optimizer)
        # The disabled budget classifier is frozen and intentionally not saved.
        self.assertFalse([name for name in missing if name.startswith("readout.")])
        self.assertFalse(unexpected)
        self.assertEqual(step, 1)
        self.assertTrue(torch.equal(expected, self.model.readout_cached(self.batch)["soft_tokens"]))
        checkpoint = torch.load(path, weights_only=False)
        self.assertEqual(checkpoint["projector_layout"]["query_mode"], "none")
        self.assertEqual(checkpoint["generator_trainable"], {})
        self.assertFalse(any("context_proj" in name or "to_value" in name or "fixed_query" in name
                             for name in checkpoint["state_dict"]))

    def test_cross_mode_checkpoint_loads_are_rejected_even_for_warm_start(self):
        for mode in ("conditioned", "agnostic_matched"):
            cfg = copy.deepcopy(self.cfg)
            cfg.readout.projector_query_mode = mode
            other = QuROModel(cfg, self.model.lm, self.fixture.stack.tokenizer,
                              build_query_encoder(cfg, self.fixture.stack), 8, 16)
            path = os.path.join(self.root, mode+".pt")
            other.save(path)
            with self.assertRaisesRegex(ValueError, "configuration"):
                self.model.load(path, allow_new_support_head=True, allow_new_projector_fusion=True)
            self.model.save(path)
            with self.assertRaisesRegex(ValueError, "configuration"):
                other.load(path, allow_new_support_head=True, allow_new_projector_fusion=True)

    def test_incomplete_or_forged_document_only_checkpoint_is_rejected(self):
        path = os.path.join(self.root, "bad.pt")
        self.model.save(path)
        checkpoint = torch.load(path, weights_only=False)
        missing = copy.deepcopy(checkpoint)
        missing["state_dict"].pop("readout.memory_proj.weight")
        torch.save(missing, path)
        with self.assertRaisesRegex(ValueError, "missing"):
            self.model.load(path)
        checkpoint["state_dict"]["readout.to_value.weight"] = torch.zeros(8, 16)
        torch.save(checkpoint, path)
        with self.assertRaisesRegex(ValueError, "unexpectedly contains query"):
            self.model.load(path)

    def test_old_sq_layout_remains_loadable(self):
        path = os.path.join(self.root, "old_sq.pt")
        self.fixture.model.save(path)
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["projector_layout"].pop("query_mode")
        torch.save(checkpoint, path)
        self.fixture.model.load(path)

    def test_bf16_mistral_and_both_lora_adapters_remain_frozen(self):
        self.fixture.cfg.readout.projector_query_mode = "none"
        shared_tests.SharedReaderTests.test_real_mistral_and_both_peft_adapters_remain_frozen(self.fixture)

    def test_matched_trainer_startup_controls_support_logging_and_saved_counts(self):
        from src.train import main
        data = os.path.join(self.root, "annotated.jsonl")
        with open(data, "w") as handle:
            for row in self.fixture.dataset.rows:
                handle.write(json.dumps(support_tests.annotate_fixture(row, self.fixture.cache))+"\n")
        starts, first_losses = [], []
        for mode in ("conditioned", "none"):
            cfg = copy.deepcopy(self.fixture.cfg)
            cfg.data.train_file, cfg.data.eval_files = data, {"dev": data}
            cfg.train.eval_batch_size, cfg.train.gen_max_new_tokens = 2, 1
            cfg.train.log_every = 1
            out = os.path.join(self.root, mode)
            argv = ["train", "--preset", "pisco_shared_projector", "--projector_query_mode", mode,
                    "--steps", "2", "--batch_size", "2", "--grad_accum", "2", "--device", "cpu",
                    "--eval_every", "1", "--eval_every_samples", "2", "--eval_max_samples", "2",
                    "--out_dir", out, "--support_head", "--support_head_input", "output",
                    "--support_loss_weight", ".1", "--query_control"]
            with patch("sys.argv", argv), patch("src.train.get_config", return_value=cfg), \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
            with open(os.path.join(out, "result.json")) as handle:
                result = json.load(handle)
            self.assertEqual(result["arm"], "S0+DocE" if mode == "none" else "SQ+DocE")
            self.assertEqual(result["projector_uses_query"], mode == "conditioned")
            self.assertEqual(result["parameters"]["total"], result["parameters"]["readout"])
            with open(os.path.join(out, "train_log.jsonl")) as handle:
                records = [json.loads(line) for line in handle]
            starts.append(next(r["validation"]["f1"] for r in records if "validation" in r))
            first_losses.append(next(r["qa_loss"] for r in records if "qa_loss" in r))
            if mode == "none":
                gradients = [r["query_condition_grad_norm"] for r in records if "query_condition_grad_norm" in r]
                self.assertTrue(gradients and all(g is None for g in gradients))
                with open(os.path.join(out, "predictions_dev_D0_Bfull.json")) as handle:
                    normal = json.load(handle)
                with open(os.path.join(out, "predictions_dev_mismatch-q_D0_Bfull.json")) as handle:
                    swapped = json.load(handle)
                self.assertEqual([r["pred"] for r in normal], [r["pred"] for r in swapped])
        self.assertEqual(starts[0], starts[1])
        self.assertEqual(first_losses[0], first_losses[1])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

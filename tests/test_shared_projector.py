"""Shared projection contracts plus frozen-reader/training integration."""
import copy
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_joint_projector as joint_tests
from config import arm_label, get_config
from src.model import QuROModel, build_model, build_query_encoder
from src.projector import SharedDocumentProjector


class SharedProjectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.z = torch.randn(2, 3, 2, 8)
        self.dm = torch.tensor([[True, True, False], [True, False, True]])
        self.q = torch.randn(2, 5, 8)
        self.qm = torch.tensor([[True, True, True, False, False],
                                [True, True, True, True, True]])

    def make(self, **kwargs):
        return SharedDocumentProjector(8, 8, 2, hidden_dim=12,
                                        attention_dim=8, num_heads=2, **kwargs)

    def activate(self, p):
        with torch.no_grad():
            p.out_proj.weight.normal_(std=0.1)
            p.out_proj.bias.normal_(std=0.1)

    def test_identity_and_actual_full_budget(self):
        for cross_document in (False, True):
            p = self.make(cross_document=cross_document)
            out, aux = p(self.z, self.dm, self.q, self.qm)
            expected = torch.where(self.dm[:, :, None, None], self.z, 0.0).flatten(1, 2)
            self.assertTrue(torch.equal(out, expected))
            self.assertEqual(aux["token_mask"].sum(1).tolist(), [4, 4])
            with self.assertRaises(ValueError):
                p(self.z, self.dm, self.q, self.qm, budget=6)

    def test_document_permutation_equivariance_with_and_without_mixing(self):
        permutation = torch.tensor([2, 0, 1])
        for cross_document in (False, True):
            p = self.make(cross_document=cross_document)
            self.activate(p)
            a, aa = p(self.z, self.dm, self.q, self.qm)
            b, bb = p(self.z[:, permutation], self.dm[:, permutation], self.q, self.qm)
            self.assertTrue(torch.allclose(b.reshape(2, 3, 2, 8),
                                          a.reshape(2, 3, 2, 8)[:, permutation], atol=1e-6))
            self.assertTrue(torch.equal(bb["token_mask"].reshape(2, 3, 2),
                                        aa["token_mask"].reshape(2, 3, 2)[:, permutation]))

    def test_same_weights_accept_k2_k10_and_query_longer_than_64(self):
        p = self.make()
        for k, t in [(2, 7), (10, 80)]:
            out, aux = p(torch.randn(1, k, 2, 8), torch.ones(1, k, dtype=torch.bool),
                         torch.randn(1, t, 8))
            self.assertEqual(tuple(out.shape), (1, k*2, 8))
            self.assertEqual(int(aux["token_mask"].sum()), k*2)

    def test_padding_is_inert_even_when_values_are_nan(self):
        for cross_document in (False, True):
            p = self.make(cross_document=cross_document)
            self.activate(p)
            a, _ = p(self.z, self.dm, self.q, self.qm)
            z, q = self.z.clone(), self.q.clone()
            z[~self.dm] = float("nan")
            q[~self.qm] = float("nan")
            b, _ = p(z, self.dm, q, self.qm)
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.isfinite(b).all())
            padded_z = torch.cat([z, torch.full_like(z[:, :1], float("nan"))], 1)
            padded_dm = torch.cat([self.dm, torch.zeros(2, 1, dtype=torch.bool)], 1)
            c, _ = p(padded_z, padded_dm, q, self.qm)
            self.assertTrue(torch.allclose(a, c[:, :6], atol=1e-6))

    def test_query_attention_masks_and_fused_path_agree(self):
        p = self.make()
        self.activate(p)
        a, _ = p(self.z, self.dm, self.q, self.qm)
        b, aux = p(self.z, self.dm, self.q, self.qm, return_attn=True)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))
        weights = aux["query_attention"]
        self.assertEqual(tuple(weights.shape), (2, 2, 6, 5))
        self.assertEqual(float(weights[0, :, :, 3:].detach().abs().sum()), 0.0)
        valid = aux["token_mask"][:, None, :].expand(-1, 2, -1)
        self.assertTrue(torch.allclose(weights.sum(-1)[valid], torch.ones_like(weights.sum(-1)[valid])))
        self.assertEqual(float(weights.sum(-1)[~valid].detach().abs().sum()), 0.0)

    def test_query_changes_output_and_repetition_does_not_amplify_readout(self):
        p = self.make()
        self.activate(p)
        a, _ = p(self.z, self.dm, self.q, self.qm)
        b, _ = p(self.z, self.dm, torch.randn_like(self.q), self.qm)
        self.assertGreater(float((a-b).detach().abs().max()), 1e-5)
        repeated, _ = p(self.z, self.dm, self.q.repeat(1, 3, 1), self.qm.repeat(1, 3))
        self.assertTrue(torch.allclose(a, repeated, atol=1e-6))

    def test_cross_document_mixing_is_optional_and_effective(self):
        z = self.z[:1, :2].clone()
        dm = torch.ones(1, 2, dtype=torch.bool)
        changed = z.clone()
        changed[:, 1] = torch.randn_like(changed[:, 1])
        for cross_document in (False, True):
            p = self.make(cross_document=cross_document)
            self.activate(p)
            a, _ = p(z, dm, self.q[:1], self.qm[:1])
            b, _ = p(changed, dm, self.q[:1], self.qm[:1])
            if cross_document:
                self.assertGreater(float((a[:, :2]-b[:, :2]).detach().abs().max()), 1e-6)
            else:
                self.assertTrue(torch.equal(a[:, :2], b[:, :2]))

    def test_last_valid_token_control_handles_left_and_right_padding(self):
        p = self.make(conditioning="last")
        self.activate(p)
        valid = self.q[:1, :3]
        pad = torch.full((1, 2, 8), float("nan"))
        a, _ = p(self.z[:1], self.dm[:1], torch.cat([valid, pad], 1), self.qm[:1])
        b, _ = p(self.z[:1], self.dm[:1], torch.cat([pad, valid], 1),
                 torch.tensor([[False, False, True, True, True]]))
        self.assertTrue(torch.equal(a, b))
        changed = valid.clone()
        changed[:, :2] = torch.randn_like(changed[:, :2])
        c, _ = p(self.z[:1], self.dm[:1], changed)
        self.assertTrue(torch.equal(a, c))

    def test_word_features_retain_order_via_fixed_positions(self):
        p = self.make(query_position=True)
        self.activate(p)
        q = self.q[:1, :3]
        a, _ = p(self.z[:1], self.dm[:1], q)
        b, _ = p(self.z[:1], self.dm[:1], q.flip(1))
        self.assertGreater(float((a-b).detach().abs().max()), 1e-6)

    def test_matched_control_ignores_query_content_and_length(self):
        for cross_document in (False, True):
            p = self.make(query_mode="agnostic_matched", cross_document=cross_document)
            self.activate(p)
            a, _ = p(self.z, self.dm, self.q, self.qm)
            b, _ = p(self.z, self.dm, torch.randn(2, 70, 8))
            self.assertTrue(torch.equal(a, b))
            conditioned = self.make(cross_document=cross_document)
            self.assertEqual(sum(x.numel() for x in p.parameters()),
                             sum(x.numel() for x in conditioned.parameters()))

    def test_degenerate_context_normalisation_is_finite_not_constant_norm(self):
        p = self.make()
        context = p.context_norm(torch.zeros(1, 8))
        self.assertTrue(torch.isfinite(context).all())
        self.assertEqual(float(context.square().sum()), 0.0)


class SharedReaderTests(joint_tests.FrozenReaderTests):
    """Exercise the same CE, freezing, checkpoint and trainer contracts for SQ."""

    def setUp(self):
        super().setUp()
        self.cfg.readout.kind = "shared_projector"
        self.cfg.readout.projector_attention_dim = 8
        self.cfg.readout.projector_heads = 2
        self.cfg.query_encoder.kind = "generator"
        self.cfg.revalidate()
        self.stack, self.model = build_model(self.cfg, 16)

    def test_bad_freeze_and_budget_settings_are_rejected(self):
        for section, name, value in [("generator", "lora_init", "pisco"),
                                     ("train", "budget_dropout", True),
                                     ("data", "prefer_teacher_output", True),
                                     ("decoder", "input_mode", "D2")]:
            cfg = copy.deepcopy(self.cfg)
            setattr(getattr(cfg, section), name, value)
            with self.assertRaises(ValueError):
                cfg.revalidate()
        with self.assertRaisesRegex(ValueError, "explicit budget"):
            self.model.readout_cached(self.batch, budget=8)

    def test_checkpoint_can_change_document_and_query_caps(self):
        path = os.path.join(self.tmp.name, "shared.pt")
        with torch.no_grad():
            self.model.readout.out_proj.weight.normal_(std=0.03)
        self.model.save(path)
        cfg = copy.deepcopy(self.cfg)
        cfg.data.max_docs = 10
        cfg.data.max_query_len = 80
        rebuilt = QuROModel(cfg, self.model.lm, self.stack.tokenizer,
                            build_query_encoder(cfg, self.stack), 8, 16)
        rebuilt.load(path)
        a = self.model.readout_cached(self.batch)["soft_tokens"]
        b = rebuilt.readout_cached(self.batch)["soft_tokens"]
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(tuple(b.shape), (2, 16, 16))

    def test_answer_ce_trains_query_and_document_attention(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.readout.projector_cross_document = True
        model = QuROModel(cfg, self.model.lm, self.stack.tokenizer,
                          build_query_encoder(cfg, self.stack), 8, 16)
        model.train()
        optim = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
        for step in range(3):
            optim.zero_grad(set_to_none=True)
            model(self.batch)["loss"].backward()
            if step:
                for module in (model.readout.to_query, model.readout.to_key,
                               model.readout.to_value, model.readout.document_attention):
                    gradients = [p.grad for p in module.parameters()]
                    self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                    self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.0)
            optim.step()
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.lm.parameters()))

    def test_main_and_control_presets(self):
        cfg = get_config("pisco_shared_projector")
        self.assertEqual(cfg.query_encoder.kind, "generator")
        self.assertEqual(arm_label(cfg), "SQ")
        cfg.readout.projector_cross_document = True
        self.assertEqual(arm_label(cfg), "SQX")
        cfg.readout.projector_conditioning = "last"
        self.assertEqual(arm_label(cfg), "SLX")
        cfg.readout.projector_query_mode = "agnostic_matched"
        self.assertEqual(arm_label(cfg), "S0mX")
        cfg.readout.projector_heads = 3
        with self.assertRaisesRegex(ValueError, "divisible"):
            cfg.revalidate()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

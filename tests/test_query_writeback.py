"""RQ contracts, including inherited trained-P and frozen-checkpoint integration."""
import copy
import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_refinement
from config import get_config, apply_arm, arm_label
from src.model import build_model
from src.query_writeback import PiscoQueryWritebackReadout


class QueryWritebackContracts(test_refinement.ResidualContracts):
    readout_class = PiscoQueryWritebackReadout
    arm = "RQ"

    def test_attention_direction_and_padding_invariance(self):
        torch.manual_seed(4)
        module = self.readout_class(16, 16, 12, d_readout=16, num_heads=4)
        # Nonzero bridge makes these checks meaningful beyond the identity bypass.
        torch.nn.init.normal_(module.out_proj.weight, std=0.1)
        z, q = torch.randn(1, 2, 3, 16), torch.randn(1, 5, 12)
        dm, qm = torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 5, dtype=torch.bool)
        out, aux = module(z, dm, q, qm, budget=1, return_attn=True)
        a = aux['attention']
        self.assertEqual(a.shape, (1, 4, 5, 6))  # Q=query, KV=latent, not the reverse
        torch.testing.assert_close(a.sum(-1), torch.ones(1, 4, 5))
        # Add padded queries AND documents, whose values may be NaN. Valid output
        # and attention must be independent of batch padding length and contents.
        zp = torch.cat([z, torch.full((1, 1, 3, 16), float('nan'))], dim=1)
        qp = torch.cat([q, torch.full((1, 7, 12), float('nan'))], dim=1)
        dmp = torch.tensor([[True, True, False]])
        qmp = torch.tensor([[True] * 5 + [False] * 7])
        padded, more = module(zp, dmp, qp, qmp, return_attn=True)
        torch.testing.assert_close(padded[:, :6], out)
        torch.testing.assert_close(more['attention'][:, :, :5, :6], a)
        self.assertEqual(more['attention'][:, :, 5:, :].count_nonzero(), 0)
        self.assertEqual(more['attention'][:, :, :, 6:].count_nonzero(), 0)
        padded[:, :6].square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in module.parameters()))
        for proj in (module.query_proj[1], module.key_proj, module.value_proj):
            self.assertGreater(proj.weight.grad.abs().sum().item(), 0)
        # Reordering the latent axis reorders outputs, without changing content.
        flat = z.flatten(1, 2)
        perm = torch.tensor([3, 1, 5, 0, 2, 4])
        reordered, _ = module(flat[:, perm].reshape_as(z), dm, q, qm)
        torch.testing.assert_close(reordered, out[:, perm])

    def test_no_column_renormalisation(self):
        # One query, two orthogonal memories: explicit nonuniform probabilities.
        # Identity norms/projections isolate the read/write operator. Renormalising
        # A^T would make both outputs equal and fail this hand-computed case.
        module = self.readout_class(2, 2, 2, d_readout=2, num_heads=1)
        module.query_proj = torch.nn.Identity()
        module.memory_norm = torch.nn.Identity()
        module.key_proj = torch.nn.Identity()
        module.value_proj = torch.nn.Identity()
        module.activation = torch.nn.Identity()
        with torch.no_grad():
            module.out_proj.weight.copy_(torch.eye(2))
        z = torch.eye(2).reshape(1, 1, 2, 2)
        q = torch.tensor([[[2.0 ** 0.5, 0.0]]])
        out, aux = module(z, torch.ones(1, 1, dtype=torch.bool), q, return_attn=True)
        a = torch.tensor([1.0, 0.0]).softmax(0)
        expected_delta = torch.outer(a, a)
        torch.testing.assert_close(out[0], torch.eye(2) + expected_delta)

    def test_empty_rows_and_unsupported_depth(self):
        module = self.readout_class(8, 8, 8, d_readout=8, num_heads=2)
        z, q = torch.randn(1, 1, 2, 8), torch.randn(1, 3, 8)
        dm = torch.ones(1, 1, dtype=torch.bool)
        with self.assertRaises(ValueError):
            module(z, ~dm, q)
        with self.assertRaises(ValueError):
            module(z, dm, q, torch.zeros(1, 3, dtype=torch.bool))
        with self.assertRaises(ValueError):
            self.readout_class(8, 8, 8, num_blocks=2)

    def test_amp_identity_and_backward(self):
        module = self.readout_class(16, 16, 12, d_readout=16, num_heads=4, dropout=0.1)
        z, q = torch.randn(2, 2, 3, 16), torch.randn(2, 5, 12)
        dm = torch.ones(2, 2, dtype=torch.bool)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out, _ = module(z, dm, q)
            loss = (out - 1).square().mean()
        self.assertTrue(torch.equal(out, z.flatten(1, 2)))
        loss.backward()
        self.assertGreater(module.out_proj.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in module.parameters()))

    def test_arm_config_and_checkpoint_separation(self):
        rq_cfg = get_config('toy')
        apply_arm(rq_cfg, 'RQ')
        self.assertEqual(arm_label(rq_cfg), 'RQ')
        bad = copy.deepcopy(rq_cfg)
        bad.readout.adaptive_budget = True
        with self.assertRaises(ValueError):
            bad.revalidate()
        r_cfg = copy.deepcopy(rq_cfg)
        apply_arm(r_cfg, 'R')
        with tempfile.TemporaryDirectory() as root:
            # The explicit kind check must reject before loading any tensors,
            # even when load(strict=False) would otherwise ignore missing keys.
            for saved_kind, cfg in [('pisco_residual', rq_cfg), ('pisco_query_writeback', r_cfg)]:
                path = os.path.join(root, 'wrong.pt')
                torch.save({'config': {'readout': {'kind': saved_kind}}, 'state_dict': {}}, path)
                _, model = build_model(cfg, cache_hidden=64)
                with self.assertRaisesRegex(ValueError, 'readout kind mismatch'):
                    model.load(path)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

"""CPU contracts: identity, trainability, masks and trained-P checkpoint fidelity."""
import copy
import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, apply_arm, arm_label
from src.refinement import PiscoResidualReadout, initialize_from_pisco, verify_pisco_identity
from src.model import build_model
from src.cache import LatentCache
from src.data import QuROCollator, QuRODataset
from test_shapes import build_workspace


class ResidualContracts(unittest.TestCase):
    def test_identity_gradients_query_and_padding(self):
        torch.manual_seed(17)
        module = PiscoResidualReadout(16, 16, 12, d_readout=16, num_heads=4)
        z = torch.randn(2, 3, 2, 16)
        mask = torch.tensor([[True, True, True], [True, False, True]])
        q = torch.randn(2, 4, 12)
        qm = torch.tensor([[True, True, True, True], [True, True, False, False]])
        out, aux = module(z, mask, q, qm, budget=1)
        self.assertTrue(torch.equal(out, z.flatten(1, 2)))
        self.assertEqual(aux['token_mask'].sum(-1).tolist(), [6, 4])
        target = torch.randn_like(out)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        for step in range(2):
            optimizer.zero_grad()
            out, _ = module(z, mask, q, qm)
            ((out - target)[aux['token_mask']].square().mean()).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in module.parameters()
                                if p.grad is not None))
            self.assertGreater(module.out_proj.weight.grad.abs().sum().item(), 0)
            if step == 1:
                self.assertGreater(module.query_proj[1].weight.grad.abs().sum().item(), 0)
            optimizer.step()
        module.eval()
        out, _ = module(z, mask, q, qm)
        other, _ = module(z, mask, q + torch.randn_like(q), qm)
        self.assertFalse(torch.allclose(out[aux['token_mask']], other[aux['token_mask']]))
        # Perturb padding only: no effect on real latent outputs.
        zbad, qbad = z.clone(), q.clone()
        zbad[1, 1] = float('nan')
        qbad[1, 2:] = float('nan')
        padded, _ = module(zbad, mask, qbad, qm)
        self.assertTrue(torch.equal(out[aux['token_mask']], padded[aux['token_mask']]))

    def test_trained_p_identity_and_frozen_decoder_reload(self):
        with tempfile.TemporaryDirectory() as root:
            cache_dir, train_path, _, m, h = build_workspace(root)
            cfg = get_config('toy')
            cfg.train.out_dir = root
            cfg.data.train_file = train_path
            cfg.data.cache_dir = cache_dir
            cfg.data.max_docs = 2
            cfg.readout.cache_hidden = h
            cfg.readout.d_readout = 32
            cfg.decoder.query_text_dropout = 0
            apply_arm(cfg, 'P')
            torch.manual_seed(3)
            stack, p = build_model(cfg, cache_hidden=h)
            collator = QuROCollator(LatentCache(cache_dir), pad_id=p.pad_id, max_docs=2)
            dataset = QuRODataset(train_path, stack.tokenizer, cfg.data)
            batch = collator([dataset[0], dataset[1]])
            # Change P before saving, so reusing the original random decoder
            # cannot accidentally pass the identity or checkpoint test.
            opt = torch.optim.SGD(p.trainable_parameters(), lr=0.05)
            p(batch)['loss'].backward()
            opt.step()
            source = os.path.join(root, 'p.pt')
            p.save(source, step=1)
            p.eval()
            with torch.no_grad():
                _, p_result = p.qa_loss(batch, return_logits=True)

            rcfg = copy.deepcopy(cfg)
            apply_arm(rcfg, 'R')
            rcfg.generator.lora_init = 'frozen'
            rcfg.revalidate()
            torch.manual_seed(99)
            _, r = build_model(rcfg, cache_hidden=h)
            initialize_from_pisco(r, source)
            self.assertEqual(arm_label(rcfg), 'R')
            self.assertTrue(verify_pisco_identity(r, batch)['identity'])
            r.eval()
            with torch.no_grad():
                _, result = r.qa_loss(batch, return_logits=True)
            self.assertTrue(torch.equal(result['answer_logits'], p_result['answer_logits']))
            decoder_before = {n: v.clone() for n, v in r.lm.state_dict().items()}
            ropt = torch.optim.AdamW(r.trainable_parameters(), lr=1e-3)
            r.train()
            for _ in range(2):
                ropt.zero_grad()
                loss = r(batch)['loss']
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertGreater(r.readout.out_proj.weight.grad.abs().sum().item(), 0)
                ropt.step()
            self.assertTrue(all(p.grad is None for p in r.lm.parameters()))
            self.assertTrue(all(torch.equal(v, decoder_before[n])
                                for n, v in r.lm.state_dict().items()))
            r.eval()
            _, trained = r.qa_loss(batch, return_logits=True)
            dest = os.path.join(root, 'r.pt')
            r.save(dest, step=2)
            torch.manual_seed(111)
            _, restored = build_model(rcfg, cache_hidden=h)
            restored.load(dest)
            restored.eval()
            _, actual = restored.qa_loss(batch, return_logits=True)
            self.assertTrue(torch.equal(trained['answer_logits'], actual['answer_logits']))
            self.assertEqual(restored.baseline_initialization['step'], 1)

    def test_training_data_moves_only_when_asked_and_is_recorded(self):
        """Which rows R trains on is a variable; it may not move silently.

        The residual results put the training data next in line, so the guard
        has to let it move -- but a run whose comparison to P's 54.50 rests on
        an unrecorded data swap is worth nothing, and a cache that encodes
        differently is not a replacement at all.
        """
        with tempfile.TemporaryDirectory() as root:
            cache_dir, train_path, _, m, h = build_workspace(root)
            cfg = get_config('toy')
            cfg.train.out_dir = root
            cfg.data.train_file = train_path
            cfg.data.cache_dir = cache_dir
            cfg.data.max_docs = 2
            cfg.readout.cache_hidden = h
            cfg.readout.d_readout = 32
            cfg.decoder.query_text_dropout = 0
            apply_arm(cfg, 'P')
            torch.manual_seed(3)
            _, p = build_model(cfg, cache_hidden=h)
            source = os.path.join(root, 'p.pt')
            p.save(source, step=1)

            def fresh(train_file, cache):
                rcfg = copy.deepcopy(cfg)
                apply_arm(rcfg, 'R')
                rcfg.generator.lora_init = 'frozen'
                rcfg.data.train_file = train_file
                rcfg.data.cache_dir = cache
                rcfg.revalidate()
                torch.manual_seed(99)
                return rcfg, build_model(rcfg, cache_hidden=h)[1]

            # Same rows at a different path: the toy tokenizer derives its vocab
            # from the training file, and a different vocab would fail the
            # decoder shape check before the data guard was ever reached.
            other_train = os.path.join(root, 'held_out.jsonl')
            with open(train_path, encoding='utf-8') as src, \
                    open(other_train, 'w', encoding='utf-8') as dst:
                dst.write(src.read())

            # Silence is the failure mode: without the flag, a moved train file stops the run.
            _, r = fresh(other_train, cache_dir)
            with self.assertRaises(ValueError):
                initialize_from_pisco(r, source)
            # Asked for explicitly, it proceeds and says what it changed.
            _, r = fresh(other_train, cache_dir)
            record = initialize_from_pisco(r, source, allow_data_change=True)
            self.assertEqual(record['data_deviation']['train_file']['run'], other_train)
            self.assertNotIn('cache_dir', record['data_deviation'])

            # A replacement cache must cover the baseline's documents ...
            import json as _json

            def write_manifest(directory, manifest):
                os.makedirs(directory)
                with open(os.path.join(directory, 'manifest.json'), 'w', encoding='utf-8') as f:
                    _json.dump(manifest, f)

            with open(os.path.join(cache_dir, 'manifest.json'), encoding='utf-8') as f:
                manifest = _json.load(f)
            short = os.path.join(root, 'short-cache')
            dropped = dict(manifest)
            dropped['documents'] = {k: v for k, v in list(manifest['documents'].items())[1:]}
            write_manifest(short, dropped)
            _, r = fresh(train_path, short)
            with self.assertRaisesRegex(ValueError, 'superset'):
                initialize_from_pisco(r, source, allow_data_change=True)

            # ... and encode them the same way.
            recoded = os.path.join(root, 'recoded-cache')
            changed = dict(manifest)
            changed['compr_rate'] = (manifest.get('compr_rate') or 0) + 1
            write_manifest(recoded, changed)
            _, r = fresh(train_path, recoded)
            with self.assertRaisesRegex(ValueError, 'encodes differently'):
                initialize_from_pisco(r, source, allow_data_change=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

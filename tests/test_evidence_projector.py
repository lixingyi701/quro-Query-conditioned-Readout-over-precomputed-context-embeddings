"""CPU checks for the new architecture, auxiliary gradients and offline boundary.

These tests do not measure published PISCO/HotpotQA performance.
"""
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config
from scripts.audit_evidence_pools import audit
from scripts.run_evidence_experiments import build_commands
from scripts.build_evidence_pool_pairs import build_pairs
from scripts.analyze_evidence_comparison import compare
from src.cache import LatentCache
from src.data import QuROCollator, QuRODataset, read_jsonl
from src.evidence import (EVIDENCE_INSTRUCTION, ENCODER_RULE, annotate_evidence,
                          digest, evidence_target, source_digest)
from src.evidence_projector import QueryGuidedEvidenceProjector
from src.model import QuROModel, build_model, build_query_encoder
from src import train
from test_shapes import build_workspace


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.z = torch.randn(2, 3, 2, 8)
        self.dm = torch.tensor([[True, False, True], [True, True, False]])
        self.q = torch.randn(2, 4, 8)
        self.qm = torch.tensor([[True, True, False, False], [True, True, True, True]])

    def make(self, **kwargs):
        return QueryGuidedEvidenceProjector(8, 8, 2, 12, 8, 2, **kwargs)

    def test_zero_step_and_masks_exclude_padding_from_both_attention_sides(self):
        p = self.make()
        with torch.no_grad():
            p.base.out_proj.weight.normal_(std=.1)
        initial, _ = p(self.z, self.dm, self.q, self.qm)
        base, _ = p.base(self.z, self.dm)
        self.assertTrue(torch.equal(initial, base))
        with torch.no_grad():
            p.out_proj.weight.normal_(std=.1)
        a, aux = p(self.z, self.dm, self.q, self.qm, return_attn=True)
        z, q = self.z.clone(), self.q.clone()
        z[~self.dm], q[~self.qm] = float('nan'), float('nan')
        b, _ = p(z, self.dm, q, self.qm, return_attn=True)
        self.assertTrue(torch.equal(a, b))
        fused, _ = p(z, self.dm, q, self.qm)
        self.assertTrue(torch.allclose(b, fused, atol=1e-6))
        self.assertTrue(torch.isfinite(b).all())
        self.assertEqual(float(b[~aux['token_mask']].abs().sum()), 0)
        self.assertEqual(tuple(aux['content_attention'].shape), (4, 2, 2, 2))
        self.assertEqual(aux['valid_document_indices'].tolist(), self.dm.nonzero().tolist())
        with self.assertRaises(ValueError):
            p(z, self.dm, q, torch.zeros_like(self.qm))

    def test_query_effect_document_isolation_and_fixed_query_control(self):
        for mode in ('conditioned', 'agnostic_matched'):
            p = self.make(query_mode=mode)
            with torch.no_grad():
                p.out_proj.weight.normal_(std=.1)
            a, _ = p(self.z, self.dm, self.q, self.qm)
            changed, _ = p(self.z, self.dm, torch.randn_like(self.q), self.qm)
            if mode == 'conditioned':
                self.assertGreater(float((a-changed).abs().max()), 1e-6)
            else:
                self.assertTrue(torch.equal(a, changed))
            z = self.z.clone()
            z[:, 2] = torch.randn_like(z[:, 2])
            b, _ = p(z, self.dm, self.q, self.qm)
            self.assertTrue(torch.equal(a[:, :2], b[:, :2]))
            perm = torch.tensor([2, 0, 1])
            c, _ = p(self.z[:, perm], self.dm[:, perm], self.q, self.qm)
            self.assertTrue(torch.allclose(c.reshape(2, 3, 2, 8),
                                          a.reshape(2, 3, 2, 8)[:, perm], atol=1e-6))

    def test_ablation_initialisation_and_slotwise_content_mask(self):
        modules = {}
        for stage in ('full', 'first_only', 'slotwise'):
            torch.manual_seed(31)
            modules[stage] = self.make(stage=stage)
        for name, value in modules['full'].state_dict().items():
            for p in modules.values():
                self.assertTrue(torch.equal(value, p.state_dict()[name]), name)
        p = modules['slotwise']
        _, aux = p(self.z, self.dm, self.q, self.qm, return_attn=True)
        self.assertTrue(torch.equal(aux['content_attention'], torch.eye(2).expand(4, 2, 2, 2)))
        self.assertFalse(any(p.requires_grad for p in modules['first_only'].content_attention.parameters()))
        self.assertEqual(sum(p.numel() for p in modules['full'].parameters()),
                         sum(p.numel() for p in modules['slotwise'].parameters()))


def add_target(row, manifest):
    annotation = {'version': 1, 'cache_digest': digest(manifest), 'source_digest': source_digest(row),
                  'doc_ids': row['retrieved_doc_ids'], 'encoder_rule': ENCODER_RULE,
                  'sentences': [{'doc_id': row['retrieved_doc_ids'][0], 'text': 'The code word is alpha0.'}],
                  'fact_visible': [True], 'all_facts_visible': True, 'text': 'The code word is alpha0.'}
    annotation['target_digest'] = digest(annotation)
    return {**row, 'evidence_annotation': annotation}


class ReaderIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cache_dir, train_file, _, _, _ = build_workspace(self.tmp.name, m=8, h=16)
        cfg = get_config('pisco_shared_projector')
        cfg.generator.kind = 'toy'
        cfg.generator.toy_d_model = 16
        cfg.generator.toy_n_layer = 1
        cfg.generator.toy_n_head = 2
        cfg.generator.toy_max_pos = 512
        cfg.data.train_file, cfg.data.cache_dir = train_file, cache_dir
        cfg.data.eval_files = {'dev': train_file}
        cfg.data.max_docs, cfg.data.max_query_len = 3, 32
        cfg.readout.projector_hidden = 12
        cfg.readout.projector_attention_dim, cfg.readout.projector_heads = 8, 2
        cfg.readout.projector_query_mode = 'none'
        cfg.train.out_dir = self.tmp.name
        self.stack, s0 = build_model(cfg, 16)
        with torch.no_grad():
            s0.readout.out_proj.weight.normal_(std=.03)
        self.s0_path = str(Path(self.tmp.name)/'s0.pt')
        s0.save(self.s0_path, step=9)
        self.s0 = s0
        cfg = copy.deepcopy(cfg)
        cfg.readout.kind, cfg.readout.projector_query_mode = 'evidence_projector', 'conditioned'
        cfg.train.evidence_loss_weight = .1
        cfg.train.s0_checkpoint = self.s0_path
        cfg.revalidate()
        self.cfg = cfg
        self.model = QuROModel(cfg, self.stack.lm, self.stack.tokenizer,
                              build_query_encoder(cfg, self.stack), 8, 16)
        self.model.initialize_s0(self.s0_path)
        self.cache = LatentCache(cache_dir)
        self.collator = QuROCollator(self.cache, self.model.pad_id, max_docs=3,
                                     evidence_tokenizer=self.stack.tokenizer)
        rows = [add_target(r, self.cache.manifest) for r in read_jsonl(train_file)]
        Path(train_file).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.dataset = QuRODataset(train_file, self.stack.tokenizer, cfg.data)
        self.batch = self.collator([self.dataset[0], self.dataset[1]])

    def tearDown(self):
        self.tmp.cleanup()

    def test_auxiliary_only_gradients_frozen_reader_and_joint_base(self):
        for joint in (False, True):
            model = self.model if not joint else QuROModel(
                self.joint_config(), self.stack.lm, self.stack.tokenizer,
                build_query_encoder(self.cfg, self.stack), 8, 16)
            if joint:
                model.initialize_s0(self.s0_path)
            model.train()
            # At step zero only Wout should move. Activate it to check upstream
            # gradients separately; zero-initialised paths legitimately block them.
            with torch.no_grad():
                model.readout.out_proj.weight.normal_(std=.05)
            result = model.readout_cached(self.batch)
            calls = []
            original = model.prompt_builders['D0'].build
            def record(query, budget, *args, **kwargs):
                calls.append(query)
                return original(query, budget, *args, **kwargs)
            with patch.object(model.prompt_builders['D0'], 'build', side_effect=record):
                loss, count = model.evidence_loss(self.batch, result)
            self.assertEqual(calls, [EVIDENCE_INSTRUCTION]*2)
            self.assertEqual(count, 2)
            loss.backward()
            for module in (model.readout.condition_attention, model.readout.content_attention,
                           model.readout.out_proj):
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()
                                       if p.grad is not None), 0)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.lm.parameters()))
            if joint:
                self.assertGreater(float(model.readout.base.out_proj.weight.grad.abs().sum()), 0)
            else:
                self.assertTrue(all(p.grad is None for p in model.readout.base.parameters()))

    def joint_config(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.readout.evidence_base_trainable = True
        return cfg

    def test_bf16_mistral_qa_and_evidence_keep_both_adapters_frozen(self):
        # Shared HF/PEFT fixture, now exercising this readout and evidence CE.
        # Random small Mistral architecture; no published weights or GPU.
        from test_joint_projector import FrozenReaderTests
        FrozenReaderTests.test_real_mistral_and_both_peft_adapters_remain_frozen(self)

    def test_s0_equivalence_and_frozen_base_checkpoint_roundtrip(self):
        self.model.eval()
        self.s0.eval()
        a = self.model.readout_cached(self.batch)
        b = self.s0.readout_cached(self.batch)
        self.assertTrue(torch.equal(a['soft_tokens'], b['soft_tokens']))
        self.assertEqual(float(self.model.qa_loss(self.batch)[0]), float(self.s0.qa_loss(self.batch)[0]))
        path = str(Path(self.tmp.name)/'new.pt')
        self.model.data_provenance = dict(train_sha256='train', cache_digest='cache', rows=8)
        self.model.save(path, step=2)
        rebuilt = QuROModel(self.cfg, self.stack.lm, self.stack.tokenizer,
                            build_query_encoder(self.cfg, self.stack), 8, 16)
        rebuilt.load(path)
        self.assertTrue(torch.equal(a['soft_tokens'], rebuilt.readout_cached(self.batch)['soft_tokens']))
        self.assertEqual(rebuilt.s0_provenance['step'], 9)
        rebuilt.data_provenance = dict(train_sha256='changed', cache_digest='cache', rows=8)
        optim = torch.optim.AdamW(rebuilt.trainable_parameters())
        with self.assertRaisesRegex(ValueError, 'changed training data'):
            rebuilt.load(path, optimizer=optim)
        ckpt = torch.load(path, weights_only=False)
        del ckpt['state_dict']['readout.base.memory_proj.weight']
        torch.save(ckpt, path)
        with self.assertRaisesRegex(ValueError, 'including frozen S0'):
            rebuilt.load(path)
        with self.assertRaises(ValueError):
            rebuilt.initialize_s0(path)

    def test_collation_masks_mismatches_and_overlong_targets(self):
        swapped = QuRODataset(self.cfg.data.train_file, self.stack.tokenizer, self.cfg.data,
                               readout_query_shift=1)
        self.assertEqual(self.collator([swapped[0]])['evidence_target_ids'], [[]])
        self.collator.evidence_max_len = 2
        out = self.collator([self.dataset[0]])
        self.assertEqual(out['evidence_target_ids'], [[]])
        self.assertEqual(out['evidence_status'], ['overlong'])
        loss, count = self.model.evidence_loss(out, self.model.readout_cached(out))
        self.assertEqual(count, 0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_trainer_two_steps_and_reload(self):
        dev = Path(self.tmp.name)/'dev.jsonl'
        dev.write_text(''.join(json.dumps({**r, 'id': 'dev-'+r['id']})+'\n'
                                for r in self.dataset.rows[:2]))
        cfg = copy.deepcopy(self.cfg)
        cfg.data.eval_files = {'dev': str(dev)}
        cfg.train.batch_size, cfg.train.grad_accum = 2, 1
        cfg.train.gen_max_new_tokens, cfg.train.eval_batch_size = 1, 2
        output = str(Path(self.tmp.name)/'run')
        argv = ['train.py', '--preset', 'pisco_evidence_projector', '--s0_checkpoint', self.s0_path,
                '--steps', '2', '--eval_every', '1', '--eval_every_samples', '2',
                '--eval_max_samples', '2', '--out_dir', output]
        # Remove annotations from dev: evaluation must work without evidence labels.
        dev_rows = read_jsonl(str(dev))
        dev.write_text(''.join(json.dumps({k: v for k, v in r.items() if k != 'evidence_annotation'})+'\n'
                                for r in dev_rows))
        with patch('sys.argv', argv), patch.object(train, 'get_config', return_value=cfg), contextlib.redirect_stdout(io.StringIO()):
            train.main()
        result = json.loads((Path(output)/'result.json').read_text())
        self.assertEqual(result['budget_policy'], 'all_cached')
        self.assertIsNotNone(result['train_order_digest'])
        rebuilt = QuROModel(cfg, self.stack.lm, self.stack.tokenizer,
                            build_query_encoder(cfg, self.stack), 8, 16)
        rebuilt.load(str(Path(output)/'checkpoint_last.pt'))
        self.assertEqual(len(rebuilt.generate_answer(self.batch, max_new_tokens=1)), 2)


class OffsetTokenizer:
    is_fast, truncation_side = True, 'right'
    bos_token, eos_token = '<bos>', '<eos>'
    def get_vocab(self):
        return {'<ENC>': 1}
    def __call__(self, text, **kwargs):
        # Character tokens make the prefix boundary exact in this contract test.
        return {'offset_mapping': [(i, i+1) for i in range(min(len(text), kwargs['max_length']))]}


class OfflineTests(unittest.TestCase):
    def test_paired_analysis_requires_identical_ids_and_training_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/name for name in ('A', 'B')]
            metadata = dict(s0_provenance={'sha256': 's0'}, train_order_digest='order',
                            data_provenance={'train_sha256': 'train', 'cache_digest': 'cache'})
            for path in paths:
                path.mkdir()
                (path/'result.json').write_text(json.dumps(metadata))
                rows = [{'id': 'q', 'golds': ['a'], 'f1': .5, 'em': 0}]
                (path/'predictions_dev_D0_Bfull.json').write_text(json.dumps(rows))
            report = compare(*paths, resamples=100)
            self.assertEqual(report['metrics']['f1']['delta_pp'], 0)
            (paths[1]/'predictions_dev_D0_Bfull.json').write_text(json.dumps([
                {'id': 'other', 'golds': ['a'], 'f1': .5, 'em': 0}]))
            with self.assertRaisesRegex(ValueError, 'ID sets'):
                compare(*paths, resamples=100)

    def test_natural_pair_augmentation_preserves_facts_and_command_protocol(self):
        rows = [{'id': label, 'query': label+'?', 'answers': [label],
                 'retrieved_doc_ids': ['x']+[str(i) for i in range(9)], 'gold_ranks': [0],
                 'supporting_sentences': [{'doc_rank': 0, 'sent_id': 0, 'text': label+' fact'}],
                 'evidence_annotation': {'stale': True}} for label in ('a', 'b')]
        pairs = build_pairs(rows, max_pairs=1)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]['retrieved_doc_ids'], pairs[1]['retrieved_doc_ids'])
        source = {r['id']: r for r in rows}
        for paired in pairs:
            original = source[paired['original_id']]
            self.assertEqual(original['answers'], paired['answers'])
            self.assertEqual(paired['retrieved_doc_ids'][paired['gold_ranks'][0]], 'x')
            self.assertEqual(paired['supporting_sentences'][0]['text'], original['supporting_sentences'][0]['text'])
            self.assertNotIn('evidence_annotation', paired)
        self.assertEqual(audit(pairs)['ordered_pool']['multi_question_pools_with_distinct_targets'], 1)
        from argparse import Namespace
        args = Namespace(s0=['42=s0.pt'], phase='formal', base='frozen', steps=10,
                         evidence_weight=.1, train_file='train.jsonl', dev_file='dev.jsonl',
                         cache_dir='cache', pisco_path='pisco', out_dir='runs', evidence_max_len=128)
        commands = build_commands(args)
        self.assertEqual([c['arm'] for c in commands], ['A', 'B', 'C'])
        for item in commands:
            with patch('sys.argv', ['train.py']+item['command'][2:]):
                cfg = train.apply_overrides(get_config('pisco_evidence_projector'), train.build_args())
            self.assertEqual(cfg.data.train_file, 'train.jsonl')
            self.assertEqual(cfg.data.eval_files, {'dev': 'dev.jsonl'})
            self.assertEqual(cfg.train.s0_checkpoint, 's0.pt')

    def test_sentence_visibility_and_stale_target_rejection(self):
        row = {'id': 'q', 'query': 'which?', 'retrieved_doc_ids': ['d1'], 'gold_ranks': [0],
               'supporting_sentences': [{'doc_rank': 0, 'sent_id': 0, 'text': 'Visible fact.'},
                                       {'doc_rank': 0, 'sent_id': 1, 'text': 'Hidden fact.'}]}
        manifest = {'latent_size': 8, 'doc_max_length': 128, 'compressor': 'pisco', 'documents': {'d1': {}}}
        corpus = {'d1': 'Visible fact.'+'x'*140+'Hidden fact.'}
        annotated = annotate_evidence(row, corpus, OffsetTokenizer(), manifest)
        self.assertEqual(annotated['evidence_annotation']['fact_visible'], [True, False])
        self.assertEqual(evidence_target(annotated, ['d1'], digest(manifest)), 'Visible fact.')
        self.assertEqual(evidence_target(annotated, [], digest(manifest)), '')
        annotated['query'] = 'changed question'
        with self.assertRaisesRegex(ValueError, 'stale'):
            evidence_target(annotated, ['d1'], digest(manifest))

    def test_pool_audit_distinguishes_ordered_unordered_and_document_reuse(self):
        rows = [{'id': 'a', 'query': 'A?', 'retrieved_doc_ids': ['x', 'y'], 'supporting_sentences': [{'text': 'A'}]},
                {'id': 'b', 'query': 'B?', 'retrieved_doc_ids': ['y', 'x'], 'supporting_sentences': [{'text': 'B'}]},
                {'id': 'c', 'query': 'C?', 'retrieved_doc_ids': ['x', 'z']}]
        report = audit(rows)
        self.assertEqual(report['ordered_pool']['multi_question_pools'], 0)
        self.assertEqual(report['unordered_pool']['multi_question_pools_with_distinct_targets'], 1)
        self.assertEqual(report['document_reuse']['multi_question_pools'], 2)
        with self.assertRaises(ValueError):
            audit(rows+rows[:1])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

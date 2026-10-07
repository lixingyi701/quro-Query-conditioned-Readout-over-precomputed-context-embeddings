"""Input interventions, causal answer/EOS scores and real residual hooks."""
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import analyze_projector_inputs as analysis
from scripts import eval_projector_inputs as evaluator
from src.model import build_model
from src.projector_diagnostics import (MODES, DecoderTrace, ScalarDistributions, answer_scores,
    generation_record, geometry_values, input_variants, question_positions, summarize_trace)
from src.prompt import BuiltPrompt, assemble_inputs
import test_shared_projector as shared_tests


class DiagnosticsUnitTests(unittest.TestCase):
    def test_factorization_preserves_intended_norms_and_directions_masks_nan(self):
        z = torch.tensor([[[3., 4.], [0., 2.], [float("nan"), float("nan")]]])
        e = torch.tensor([[[0., 10.], [-6., 0.], [float("nan"), float("nan")]]])
        mask = torch.tensor([[True, True, False]])
        modes, counts = input_variants(z, e, mask)
        self.assertEqual(counts["valid_tokens"], 2)
        self.assertEqual(counts["original_near_zero_tokens"], 0)
        for x in modes.values():
            self.assertTrue(torch.isfinite(x).all())
            self.assertEqual(float(x[~mask].abs().sum()), 0.)
        self.assertTrue(torch.allclose(modes["scale_only"][mask].norm(dim=-1), e[mask].norm(dim=-1)))
        self.assertTrue(torch.allclose(modes["direction_only"][mask].norm(dim=-1), z[mask].norm(dim=-1)))
        self.assertTrue(torch.allclose(F.cosine_similarity(modes["scale_only"][mask], z[mask]), torch.ones(2)))
        self.assertTrue(torch.allclose(F.cosine_similarity(modes["direction_only"][mask], e[mask]), torch.ones(2)))
        self.assertTrue(torch.equal(modes["original"][mask], z[mask]))
        self.assertTrue(torch.equal(modes["full"][mask], e[mask]))
        with self.assertRaises(ValueError):
            input_variants(z, e, torch.ones_like(mask))

    def test_zero_directions_are_explicit_and_geometry_radial_decomposition(self):
        z, e = torch.tensor([[[2., 0.], [0., 0.], [1., 0.]]]), torch.tensor([[[4., 3.], [0., 2.], [0., 0.]]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        modes, counts = input_variants(z, e, mask)
        self.assertEqual(counts["original_near_zero_tokens"], 1)
        self.assertEqual(counts["full_near_zero_tokens"], 1)
        self.assertTrue(torch.equal(modes["scale_only"][0, 1], z[0, 1]))
        self.assertTrue(torch.equal(modes["direction_only"][0, 2], e[0, 2]))
        values, _ = geometry_values(z[:, :1], e[:, :1], mask[:, :1])
        self.assertAlmostEqual(float(values["radial_delta_energy_fraction"][0]), 4/13, places=6)
        self.assertAlmostEqual(float(values["signed_radial_over_z"][0]), 1.)
        self.assertAlmostEqual(float(values["tangential_delta_rms"][0]), 3/(2**.5), places=6)
        stats = ScalarDistributions()
        stats.add({"x": torch.tensor([1.])})
        stats.add({"x": torch.tensor([3., 3., 3.])})
        self.assertAlmostEqual(stats.summary()["x"]["pooled_rms"], 7**.5)
        json.dumps(stats.summary(), allow_nan=False)

    def test_gold_scores_use_causal_shift_exclude_prompt_padding_and_eos(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 7, 6)
        labels = torch.tensor([[-100, -100, 3, 4, 2, -100, -100], [-100, -100, -100, 5, 2, -100, -100]])
        values = answer_scores(logits, labels, eos_id=2)
        for i, gold_pos in enumerate(([2, 3, 4], [3, 4])):
            ref = torch.stack([F.log_softmax(logits[i, p-1], -1)[labels[i, p]] for p in gold_pos])
            self.assertAlmostEqual(values[i]["content_logprob_mean"], float(ref[:-1].mean()))
            self.assertAlmostEqual(values[i]["eos_after_gold_logprob"], float(ref[-1]))
        total = sum(v["content_logprob_sum"]+v["eos_after_gold_logprob"] for v in values)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, 6), labels[:, 1:].reshape(-1), ignore_index=-100)
        self.assertAlmostEqual(-total/5, float(ce), places=6)
        with self.assertRaises(ValueError):
            answer_scores(logits, labels.masked_fill(labels == 2, 3), eos_id=2)

    def test_generation_eos_pad_alias_and_question_prompt_boundary(self):
        class Tokenizer:
            eos_token_id = pad_token_id = 2
            def decode(self, ids, skip_special_tokens=True):
                return " ".join(map(str, ids))
        row = generation_record([3, 2, 2, 2], Tokenizer(), 4)
        self.assertEqual(row["prediction"], "3")
        self.assertEqual(row["generated_content_tokens"], 1)
        self.assertEqual(row["generated_token_ids"], [3, 2])
        self.assertTrue(row["eos_reached"])
        self.assertFalse(row["hit_generation_cap"])
        self.assertTrue(generation_record([3, 4], Tokenizer(), 2)["hit_generation_cap"])
        self.assertEqual(question_positions(BuiltPrompt([1, 7, 8, 9], []), BuiltPrompt([1, 6, 9], [])), [1, 2])

    def test_bootstrap_interaction_and_incomplete_margin_rejection(self):
        row = {"modes": {mode: {**{name: 0. for name in analysis.METRICS}, "f1": value}
                          for mode, value in zip(MODES, (.2, .3, .5, .7))}}
        out = analysis.analyze_rows([row], bootstrap=10)
        self.assertAlmostEqual(out["interaction"]["f1"]["difference"], .1)
        self.assertAlmostEqual(out["full_minus_original"]["f1"]["ci95"][0], .5)
        row["modes"]["full"]["content_candidate_margin"] = 1.
        with self.assertRaises(ValueError):
            analysis.analyze_rows([row], bootstrap=10)


class DiagnosticsIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = shared_tests.SharedReaderTests()
        self.fixture.setUp()
        self.model, self.batch = self.fixture.model, self.fixture.batch
        with torch.no_grad():
            self.model.readout.out_proj.weight.normal_(std=.02)
        self.model.eval()

    def tearDown(self):
        self.fixture.tearDown()

    def test_full_arm_matches_existing_reader_and_scores_without_mutation(self):
        model, batch = self.model, self.batch
        before = {name: p.detach().clone() for name, p in model.readout.named_parameters()}
        old = model.generate_answer(batch, max_new_tokens=2)
        stats, traces = ScalarDistributions(), {}
        candidates = {str(key): "Document" for key in batch["ids"]}
        with patch.object(model, "readout_cached", wraps=model.readout_cached) as readout:
            rows, counts = evaluator.evaluate_batch(model, batch, 2, stats, traces, True, candidates)
        self.assertEqual(readout.call_count, 1)
        self.assertEqual([r["modes"]["full"]["prediction"] for r in rows], old)
        with torch.no_grad():
            result = model.readout_cached(batch)
            packed = assemble_inputs(model.lm.get_input_embeddings(), model.build_prompts(batch, result["soft_token_mask"], False),
                result["soft_tokens"], result["soft_token_mask"], batch["target_ids"], model.pad_id)
            output = model.lm(**packed)
        scores = answer_scores(output.logits, packed["labels"], model.tok.eos_token_id)
        self.assertEqual(rows[0]["modes"]["full"]["content_logprob_mean"], scores[0]["content_logprob_mean"])
        self.assertEqual(counts["valid_tokens"], 32)
        self.assertEqual(set(traces), set(MODES))
        self.assertIn("ln_f/final_norm_input", traces["full"])
        self.assertIn("update_over_residual_rms", summarize_trace(traces["full"])["blocks.0/mlp_update"]["memory"])
        self.assertTrue(all(torch.equal(p, before[name]) for name, p in model.readout.named_parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.lm.modules()))
        json.dumps(rows, allow_nan=False)

    def test_cli_restores_checkpoint_config_finishes_analysis_and_refuses_overwrite(self):
        cfg = copy.deepcopy(self.fixture.cfg)
        cfg.train.seed = 123
        torch.manual_seed(cfg.train.seed)
        _, model = build_model(cfg, 16)
        with torch.no_grad():
            model.readout.out_proj.weight.fill_(.01)
        ckpt = os.path.join(self.fixture.tmp.name, "trained.pt")
        model.save(ckpt, step=5)
        out = os.path.join(self.fixture.tmp.name, "attribution")
        digest = evaluator.file_digest(ckpt)
        argv = ["--checkpoint", ckpt, "--eval_file", cfg.data.train_file, "--out_dir", out,
                "--max_samples", "2", "--max_new_tokens", "2", "--trace_decoder", "--device", "cpu"]
        report = evaluator.main(argv)
        self.assertEqual(report["checkpoint_config"], asdict(cfg))
        self.assertTrue(report["toy_smoke_only"])
        self.assertEqual(report["checkpoint_step"], 5)
        self.assertEqual(evaluator.file_digest(ckpt), digest)
        self.assertEqual(report["qa"]["full"]["n"], 2)
        self.assertIn("full_decoder_input_rms", report["geometry"])
        paired = analysis.main(["--run", out, "--bootstrap", "10"])
        self.assertEqual(paired["overall"]["full_minus_original"]["f1"]["n"], 2)
        with self.assertRaises(FileExistsError):
            evaluator.main(argv)
        payload = torch.load(ckpt, weights_only=False)
        payload["config"]["readout"]["projector_hidden_typo"] = 8
        with self.assertRaises(ValueError):
            evaluator.checkpoint_config(payload)

    def test_full_gold_not_truncated_and_candidate_alias_collision_rejected(self):
        self.model.cfg.data.max_answer_len = 1
        self.batch["raw"][0]["target"] = "Document number"
        rows, _ = evaluator.evaluate_batch(self.model, self.batch, 1, ScalarDistributions(), {})
        self.assertEqual(rows[0]["modes"]["full"]["content_tokens"], 2)
        self.assertTrue(rows[0]["training_target_would_truncate"])
        path = os.path.join(self.fixture.tmp.name, "candidates.jsonl")
        with open(path, "w") as handle:
            for row in self.fixture.dataset.rows:
                handle.write(json.dumps({"id": row["id"], "wrong_answer": row["answers"][0]})+"\n")
        with self.assertRaises(ValueError):
            evaluator.load_candidates(path, self.fixture.dataset)

    def test_mistral_hooks_measure_actual_residual_and_final_norm_separately(self):
        from transformers import MistralConfig, MistralForCausalLM
        torch.manual_seed(2)
        lm = MistralForCausalLM(MistralConfig(hidden_size=16, intermediate_size=24, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, vocab_size=30, max_position_embeddings=64,
            attention_dropout=0., attn_implementation="eager"))
        lm.eval()
        x = torch.randn(1, 5, 16)*3
        mask = torch.tensor([[False, True, True, False, False]])
        captured = {}
        handles = [lm.model.norm.register_forward_pre_hook(lambda m, a: captured.update(norm_input=a[0].detach())),
                   lm.model.norm.register_forward_hook(lambda m, a, o: captured.update(norm_output=o.detach()))]
        with torch.no_grad(), DecoderTrace(lm, {"memory": mask}) as trace:
            lm(inputs_embeds=x, attention_mask=torch.ones(1, 5, dtype=torch.long))
        for handle in handles:
            handle.remove()
        summary = summarize_trace(trace.stats)
        for field in ("input", "output"):
            expected = float(captured["norm_"+field][mask].square().mean().sqrt())
            self.assertAlmostEqual(summary["model.norm/final_norm_"+field]["memory"]["rms"], expected, places=6)
        self.assertTrue(torch.allclose(captured["norm_input"][mask].square().sum(),
            torch.tensor(trace.stats["model.layers.1/block_output"]["memory"]["sum_sq"]), rtol=1e-5))
        # Exercise the real HF inputs_embeds generation return contract.
        ids = lm.generate(inputs_embeds=x, attention_mask=torch.ones(1, 5, dtype=torch.long),
                          max_new_tokens=2, do_sample=False, eos_token_id=2, pad_token_id=2)
        self.assertLessEqual(ids.shape[1], 2)


if __name__ == "__main__":
    unittest.main()

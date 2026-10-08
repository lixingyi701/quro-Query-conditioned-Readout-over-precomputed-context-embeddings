"""Prevent sample-order and paired-analysis confounds in SQX validation."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import analyze_cross_document_validation as analysis
import test_shared_projector as shared_tests


class CrossDocumentAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root/"dev.jsonl"
        rows = [{"id": str(i), "query": f"Question {i}", "answers": ["a"],
                 "hop_type": "bridge" if i < 2 else "comparison"} for i in range(4)]
        self.data.write_text("".join(json.dumps(r)+"\n" for r in rows))
        self.runs = {}
        for seed in (42, 43):
            self.runs[seed] = {}
            for arm, (mode, cross) in analysis.ARMS.items():
                root = self.root/f"{seed}_{arm}"
                root.mkdir()
                self.runs[seed][arm] = str(root)
                record = {key: 0 for key in ("offline_m", "offline_compressor", "offline_compr_rate",
                          "query_text_dropout", "projector_hidden", "projector_conditioning",
                          "projector_attention_dim", "projector_heads", "max_query_len")}
                record.update(readout="shared_projector", query_encoder_kind="generator",
                              query_representation="fixed_adapter", projector_query_mode=mode,
                              projector_cross_document=cross, projector_fusion="none", support_head=False,
                              support_loss_weight=0, train_decoder_input_mode="D0", generator_lora_init="frozen")
                record["data_order_provenance"] = dict(seed=seed, data_order_seed=seed, train_sha256="train",
                    cache_manifest_sha256="cache", protocol=dict(steps=3, batch_size=2, grad_accum=2, gen_max_new_tokens=32),
                    data_limits=dict(max_docs=10, max_query_len=256), generator_path="model", fresh_start=True, completed_steps=3,
                    microbatches=6, examples=12, order_sha256=f"order{seed}")
                record["evaluation_protocol"] = dict(cache_manifest_sha256="cache", generator_path="model",
                                                      max_docs=10, max_query_len=256, gen_max_new_tokens=32)
                (root/"result.json").write_text(json.dumps(record))
                values = {"S0": [.50]*4, "SQ": [.51]*4, "S0X": [.52]*4,
                          "SQX": [.54, .54, .50, .50]}[arm]
                predictions = [{"id": str(i), "query": f"Question {i}", "golds": ["a"], "pred": "a",
                                "f1": v, "em": 1., "substring": 1.} for i, v in enumerate(values)]
                (root/"predictions_dev_D0_Bfull.json").write_text(json.dumps(predictions))

    def tearDown(self):
        self.tmp.cleanup()

    def report(self, **kwargs):
        return analysis.analyze(self.runs, self.data, resamples=100, **kwargs)

    def change_record(self, seed, arm, change):
        path = Path(self.runs[seed][arm])/"result.json"
        record = json.loads(path.read_text())
        change(record)
        path.write_text(json.dumps(record))

    def test_interaction_is_paired_per_question_and_seeds_are_not_extra_questions(self):
        report = self.report()
        bridge = report["groups"]["bridge"]["seed_mean"]["contrasts"]
        self.assertAlmostEqual(bridge["SQX-SQ"]["f1"]["delta"], .03)
        self.assertAlmostEqual(bridge["query_crossdoc_interaction"]["f1"]["delta"], .01)
        self.assertEqual(bridge["SQX-SQ"]["f1"]["n"], 2)
        self.assertAlmostEqual(report["groups"]["comparison"]["seed_mean"]["contrasts"]["SQX-SQ"]["f1"]["delta"], -.01)

    def test_order_mismatch_is_rejected_and_legacy_is_explicit(self):
        self.change_record(42, "SQX", lambda r: r["data_order_provenance"].update(order_sha256="different"))
        with self.assertRaisesRegex(ValueError, "training order"):
            self.report()
        report = self.report(legacy=True)
        self.assertFalse(report["audit"]["42"]["matched_training"])
        self.assertTrue(report["legacy_exploratory"])

    def test_wrong_architecture_is_rejected_even_for_legacy(self):
        self.change_record(42, "SQX", lambda r: r.update(projector_cross_document=False))
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.report(legacy=True)

    def test_actual_evaluation_cache_and_generation_settings_are_checked(self):
        self.change_record(42, "SQX", lambda r: r["evaluation_protocol"].update(gen_max_new_tokens=64))
        with self.assertRaisesRegex(ValueError, "evaluation protocol"):
            self.report()

    def test_best_or_partial_training_cannot_be_presented_as_last(self):
        self.change_record(43, "SQX", lambda r: r["data_order_provenance"].update(completed_steps=2))
        with self.assertRaisesRegex(ValueError, "training order"):
            self.report()

    def test_duplicate_ids_missing_ids_and_changed_question_are_rejected(self):
        path = Path(self.runs[42]["SQX"])/"predictions_dev_D0_Bfull.json"
        original = json.loads(path.read_text())
        for rows, message in ((original+[original[0]], "duplicate"), (original[:-1], "ID sets"),
                              ([{**r, "query": "wrong"} for r in original], "mismatch")):
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, message):
                self.report()

    def test_two_arm_historical_comparison_needs_no_new_training(self):
        self.runs = {42: {arm: self.runs[42][arm] for arm in ("SQ", "SQX")}}
        for arm in self.runs[42]:
            def remove_new_fields(record):
                for key in ("data_order_provenance", "projector_fusion", "support_head", "support_loss_weight"):
                    record.pop(key)
            self.change_record(42, arm, remove_new_fields)
        report = self.report(legacy=True)
        self.assertEqual(set(report["groups"]["bridge"]["seed_mean"]["contrasts"]), {"SQX-SQ"})
        self.assertFalse(report["audit"]["42"]["matched_training"])


class CrossDocumentTrainerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = shared_tests.SharedReaderTests(methodName="runTest")
        self.fixture.setUp()
        self.root = Path(self.fixture.tmp.name)

    def tearDown(self):
        self.fixture.tearDown()

    def run_arm(self, arm, evaluation_checkpoint=None):
        from src.train import main
        data = self.root/"train.jsonl"
        data.write_text("".join(json.dumps({**r, "hop_type": "bridge" if i % 2 == 0 else "comparison"})+"\n"
                                for i, r in enumerate(self.fixture.dataset.rows)))
        cfg = copy.deepcopy(self.fixture.cfg)
        cfg.data.train_file, cfg.data.eval_files = str(data), {"dev": str(data)}
        cfg.train.eval_batch_size, cfg.train.gen_max_new_tokens, cfg.train.log_every = 2, 1, 1
        out = self.root/(arm + ("_eval" if evaluation_checkpoint else ""))
        mode, cross = analysis.ARMS[arm]
        argv = ["train", "--preset", "pisco_shared_projector", "--projector_query_mode", mode,
                "--seed", "42", "--data_order_seed", "42", "--steps", "3", "--batch_size", "2",
                "--grad_accum", "2", "--lr", "1e-2", "--eval_every", "1", "--eval_every_samples", "2",
                "--eval_max_samples", "2", "--device", "cpu", "--out_dir", str(out)]
        if cross:
            argv.append("--projector_cross_document")
        if evaluation_checkpoint:
            argv.extend(["--eval_only", "--resume_from", str(evaluation_checkpoint)])
        with patch("sys.argv", argv), patch("src.train.get_config", return_value=cfg), contextlib.redirect_stdout(io.StringIO()):
            main()
        return json.loads((out/"result.json").read_text()), out

    def test_all_four_arms_share_batch_order_and_checkpoint_restores_provenance(self):
        results, directories = {}, {}
        for arm in analysis.ARMS:
            results[arm], directories[arm] = self.run_arm(arm)
        records = [r["data_order_provenance"] for r in results.values()]
        self.assertEqual(len({r["order_sha256"] for r in records}), 1)
        self.assertEqual({r["examples"] for r in records}, {12})
        self.assertEqual({r["completed_steps"] for r in records}, {3})
        self.assertEqual({r["microbatches"] for r in records}, {6})
        # Existing SQ and SQX remain identical native-reader inputs at step 0.
        starts = []
        for root in directories.values():
            logs = [json.loads(line) for line in (root/"train_log.jsonl").read_text().splitlines()]
            starts.append(logs[0]["validation"]["f1"])
        self.assertEqual(len(set(starts)), 1)
        reloaded, _ = self.run_arm("SQX", directories["SQX"]/"checkpoint_last.pt")
        self.assertEqual(reloaded["data_order_provenance"], results["SQX"]["data_order_provenance"])
        checkpoint = torch.load(directories["SQX"]/"checkpoint_last.pt", weights_only=False)
        self.assertEqual(checkpoint["generator_trainable"], {})
        self.assertTrue(any("document_attention" in key for key in checkpoint["state_dict"]))
        report = analysis.analyze({42: {a: str(p) for a, p in directories.items()}},
                                  self.root/"train.jsonl", resamples=100)
        self.assertTrue(report["audit"]["42"]["matched_training"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

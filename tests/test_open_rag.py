"""Evidence identity, fixed compute, stale retrieval and bounded search regressions."""
from argparse import Namespace
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_public_qa as public
import prepare_open_rag as prep
import run_open_rag as run
import analyze_open_rag as analysis
import build_splade_retrieval as retrieval
from src import metrics


class OpenRAGTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, rows):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        public.write_jsonl(path, rows)
        return path

    def quiet(self, function, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(args)

    def eval_fixture(self):
        native = self.write("native.jsonl", [{"id": "h1", "query": "Who won?", "answers": ["A", "Alias"],
                                               "hop_type": "bridge", "documents": ["native evidence"],
                                               "gold_ranks": [0, 1], "supporting_sentences": ["secret"]}])
        out = self.root / "questions"
        self.quiet(prep.eval_queries, Namespace(inputs=[f"hotpot_dev={native}"], out_dir=str(out)))
        documents = [{"doc_id": f"wiki{i}", "text": f"Wikipedia chunk {i}"} for i in range(10)]
        retrieved = self.write("retrieval.jsonl", [{"id": "hotpot_dev::h1", "query": "Who won?",
                                                     "documents": documents, "teacher_output": "wrong"}])
        return out, retrieved

    def test_eval_removes_native_support_and_preserves_id_labels_and_k_prefix(self):
        queries, retrieved = self.eval_fixture()
        question = list(public.read_jsonl(queries / "eval.queries.jsonl"))[0]
        self.assertNotIn("documents", question)
        self.assertNotIn("gold_ranks", question)
        self.assertNotIn("supporting_sentences", question)
        ready = self.root / "ready"
        self.quiet(prep.attach_eval, Namespace(queries_dir=str(queries), retrieval_jsonl=[str(retrieved)],
                                               ks="5,10", out_dir=str(ready)))
        five = list(public.read_jsonl(ready / "k5/hotpot_dev.jsonl"))[0]
        ten = list(public.read_jsonl(ready / "k10/hotpot_dev.jsonl"))[0]
        self.assertEqual(five["id"], "h1")
        self.assertEqual(five["answers"], ["A", "Alias"])
        self.assertEqual(five["hop_type"], "bridge")
        self.assertEqual(five["retrieved_doc_ids"], ten["retrieved_doc_ids"][:5])
        self.assertNotIn("teacher_output", five)

    def test_duplicate_question_ids_are_namespaced_between_benchmarks(self):
        path = self.write("a.jsonl", [{"id": "1", "query": "Q?", "answers": ["a"]}])
        out = self.root / "questions"
        self.quiet(prep.eval_queries, Namespace(inputs=[f"nq={path}", f"trivia={path}"], out_dir=str(out)))
        self.assertEqual({r["id"] for r in public.read_jsonl(out / "eval.queries.jsonl")}, {"nq::1", "trivia::1"})

    def test_incomplete_eval_does_not_emit_partial_benchmark(self):
        queries, retrieved = self.eval_fixture()
        rows = list(public.read_jsonl(retrieved))
        rows[0]["documents"] = rows[0]["documents"][:5]
        public.write_jsonl(retrieved, rows)
        out = self.root / "ready"
        with self.assertRaisesRegex(ValueError, "10 distinct"):
            prep.attach_eval(Namespace(queries_dir=str(queries), retrieval_jsonl=[str(retrieved)], ks="5,10", out_dir=str(out)))
        self.assertFalse(out.exists())

    def test_only_missing_questions_are_retrieved_and_incomplete_existing_rows_fail(self):
        queries = self.root / "full"
        rows = [{"id": f"nq_open{i}", "source_id": f"nq_open{i}", "query": f"Q{i}?", "answers": ["a"]} for i in range(2)]
        self.write("full/train.queries.jsonl", rows)
        self.write("full/dev.queries.jsonl", [])
        retrieved = self.write("old.jsonl", [{"id": "nq_open0", "query": "Q0?", "documents": [f"d{i}" for i in range(5)]}])
        args = Namespace(queries_dir=str(queries), retrieval_jsonl=[str(retrieved)], max_docs=5, out_dir=str(self.root / "pending"))
        self.quiet(prep.missing_queries, args)
        self.assertEqual([r["id"] for r in public.read_jsonl(self.root / "pending/train.queries.jsonl")], ["nq_open1"])

    def training_fixture(self):
        generator = self.root / "generator"
        generator.mkdir()
        rows = [{"id": str(i), "query": f"Question {i}", "answers": ["answer"], "source": "hotpotqa",
                 "retrieved_doc_ids": [f"d{j}" for j in range(5)]} for i in range(6)]
        full = self.write("full/train.jsonl", rows).parent
        small = self.write("small/train.jsonl", rows[:2]).parent
        self.write("full/dev.jsonl", [{**rows[0], "id": "dev", "query": "Dev question"}])
        (small / "dev.jsonl").write_bytes((full / "dev.jsonl").read_bytes())
        cache = self.root / "cache"
        cache.mkdir()
        (cache / "manifest.json").write_text(json.dumps({"latent_size": 8, "hidden_size": 4096,
                                                        "doc_max_length": 128, "compr_rate": 16,
                                                        "checkpoint": str(generator),
                                                        "documents": {f"d{i}": {} for i in range(10)}}))
        return Namespace(full_ready=str(full), small_ready=str(small), cache_dir=str(cache), generator_path=str(generator),
                         batch_size=2, grad_accum=2, steps=None, data_order_seed=42, seed=42, out_dir=str(self.root / "runs"))

    def test_scale_grid_uses_equal_steps_schedule_and_dev_for_all_four_jobs(self):
        args = self.training_fixture()
        plan = run.training_plan(args)
        self.assertEqual(plan["steps"], 2)
        self.assertEqual(len(plan["jobs"]), 4)
        for job in plan["jobs"]:
            command = job["command"]
            self.assertEqual(command[command.index("--steps") + 1], "2")
            self.assertEqual(command[command.index("--select_metric") + 1], "f1")
            self.assertIn("--data_order_seed", command)
            self.assertNotIn("--resume_from", command)
            self.assertNotIn("--projector_cross_document", command)
        self.assertEqual(len({j["command"][j["command"].index("--eval_files") + 1] for j in plan["jobs"]}), 1)

    def test_scale_grid_refuses_changed_dev_and_changed_subset_evidence(self):
        args = self.training_fixture()
        small = Path(args.small_ready)
        (small / "dev.jsonl").write_text("[]\n")
        with self.assertRaisesRegex(ValueError, "byte-identical"):
            run.training_plan(args)
        (small / "dev.jsonl").write_bytes((Path(args.full_ready) / "dev.jsonl").read_bytes())
        rows = list(public.read_jsonl(small / "train.jsonl"))
        rows[0]["answers"] = ["changed gold"]
        public.write_jsonl(small / "train.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "identical labelled"):
            run.training_plan(args)

    def test_pisco_eval_keeps_all_k_times_eight_memories(self):
        args = self.training_fixture()
        args.eval_files = [f"dev={Path(args.full_ready) / 'dev.jsonl'}"]
        args.checkpoints, args.k, args.max_samples = [], 5, 0
        command = run.evaluation_plan(args)["jobs"][0]["command"]
        self.assertEqual(command[command.index("--budget") + 1], "40")
        self.assertEqual(command[command.index("--readout") + 1], "pisco_direct")
        self.assertIn("--eval_only", command)

    def test_paired_analysis_rejects_intersections_and_evidence_changes(self):
        row = {"id": "1", "query": "Q?", "golds": ["Paris"], "pred": "Paris", "retrieved_doc_ids": ["d1"], "hop_type": "bridge"}
        left, right = {"1": row}, {"1": {**row, "pred": "London"}}
        result = analysis.comparison(left, right, iterations=20)
        self.assertEqual(result["groups"]["all"]["f1"]["delta_pp"], 100)
        self.assertEqual(result["groups"]["hop_type:bridge"]["n"], 1)
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            analysis.comparison(left, {**right, "2": row}, iterations=20)
        right["1"]["retrieved_doc_ids"] = ["d2"]
        with self.assertRaisesRegex(ValueError, "evidence differs"):
            analysis.comparison(left, right, iterations=20)
        self.assertTrue(analysis.comparison(left, right, iterations=20, allow_evidence_change=True)["evidence_change_allowed"])

    def test_stale_query_metadata_cannot_be_reused(self):
        path = self.root / "search_metadata.json"
        retrieval.bind_metadata(path, {"query_sha": "old", "depth": 50})
        retrieval.bind_metadata(path, {"query_sha": "old", "depth": 50})
        with self.assertRaisesRegex(ValueError, "stale/incompatible"):
            retrieval.bind_metadata(path, {"query_sha": "new", "depth": 50})

    def test_grouped_validation_keeps_sample_counts(self):
        rows = [{"source": "hotpotqa", **metrics.score("yes", ["yes"])},
                {"source": "nq_open", **metrics.score("wrong", ["gold"])}]
        result = metrics.aggregate_groups(rows, "source")
        self.assertEqual(result["hotpotqa"]["n"], 1)
        self.assertEqual(result["hotpotqa"]["f1"], 1)

    def test_failed_cache_worker_prevents_packing(self):
        # Exercise the actual shell process handling; an unqualified `wait`
        # can return success even when a background compression slice failed.
        stub_dir = self.root / "bin"
        stub_dir.mkdir()
        stub = stub_dir / "python"
        marker = self.root / "packed"
        stub.write_text('#!/bin/bash\nif [[ "$1" == *pack_latent_cache.py ]]; then touch "$PACK_MARKER"; fi\nif [[ "$*" == *part1* ]]; then exit 7; fi\nexit 0\n')
        stub.chmod(0o755)
        corpus = self.write("corpus.jsonl", [{"doc_id": str(i), "text": "x"} for i in range(4)])
        env = {**os.environ, "PATH": f"{stub_dir}:{os.environ['PATH']}", "CORPUS": str(corpus),
               "CACHE": str(self.root / "cached"), "GPUS": "0,1", "PACK_MARKER": str(marker)}
        completed = subprocess.run(["bash", "scripts/build_hotpot_cache.sh"], cwd=run.REPO, env=env,
                                   capture_output=True, text=True, timeout=15)
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(marker.exists())


class TrainerIntegrationTests(unittest.TestCase):
    def test_saved_budget_snapshots_are_evaluable_and_validation_has_source_counts(self):
        import torch
        from test_shared_projector import SharedReaderTests
        from src.train import main
        torch.set_num_threads(1)
        fixture = SharedReaderTests(methodName="runTest")
        fixture.setUp()
        try:
            root = Path(fixture.tmp.name)
            data = root / "labelled.jsonl"
            public.write_jsonl(data, ({**row, "source": "hotpotqa" if i % 2 == 0 else "nq_open"}
                                     for i, row in enumerate(fixture.dataset.rows)))
            cfg = copy.deepcopy(fixture.cfg)
            cfg.data.train_file, cfg.data.eval_files = str(data), {"dev": str(data)}
            cfg.train.eval_batch_size, cfg.train.gen_max_new_tokens = 2, 1
            out = root / "run"
            argv = ["train", "--preset", "pisco_shared_projector", "--seed", "42", "--data_order_seed", "42",
                    "--steps", "3", "--batch_size", "2", "--grad_accum", "1", "--lr", "1e-2",
                    "--eval_every", "3", "--eval_every_samples", "4", "--eval_max_samples", "4",
                    "--checkpoint_steps", "1,2", "--device", "cpu", "--out_dir", str(out)]
            with patch("sys.argv", argv), patch("src.train.get_config", return_value=cfg), contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertTrue((out / "checkpoint_step1.pt").is_file())
            self.assertTrue((out / "checkpoint_step2.pt").is_file())
            logs = [json.loads(line)["validation"] for line in (out / "train_log.jsonl").read_text().splitlines()
                    if "validation" in json.loads(line)]
            self.assertEqual([row["step"] for row in logs], [0, 1, 2, 3])
            self.assertEqual(set(logs[-1]["by_source"]), {"hotpotqa", "nq_open"})
            self.assertEqual(sum(v["n"] for v in logs[-1]["by_source"].values()), 4)
            cfg = copy.deepcopy(cfg)
            eval_dir = root / "snapshot_eval"
            argv[argv.index("--out_dir") + 1] = str(eval_dir)
            argv += ["--eval_only", "--resume_from", str(out / "checkpoint_step1.pt")]
            with patch("sys.argv", argv), patch("src.train.get_config", return_value=cfg), contextlib.redirect_stdout(io.StringIO()):
                main()
            rows = json.loads((eval_dir / "predictions_dev_D0_Bfull.json").read_text())
            self.assertEqual(len(rows), 4)
            self.assertIn("retrieved_doc_ids", rows[0])
            self.assertIn("source", rows[0])
        finally:
            fixture.tearDown()


class SparseSearchTests(unittest.TestCase):
    def test_blocked_search_matches_exhaustive_multi_shard_topk(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU torch not installed")
        import numpy as np
        import scipy.sparse as sp
        torch.set_num_threads(1)
        rng = np.random.default_rng(71)
        q = rng.uniform(.1, 1, (7, 8)).astype(np.float32)
        d = rng.uniform(.1, 1, (9, 8)).astype(np.float32)
        expected = q @ d.T
        order = np.argsort(-expected, axis=1)[:, :4]
        for block in (1, 3, 20):
            scores, indices, n = retrieval.search_sparse(sp.csr_matrix(q),
                [("small", sp.csr_matrix(d[:2])), ("rest", sp.csr_matrix(d[2:]))], 4, block, device="cpu")
            self.assertEqual(n, 9)
            np.testing.assert_array_equal(indices, order)
            np.testing.assert_allclose(scores, np.take_along_axis(expected, order, axis=1), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

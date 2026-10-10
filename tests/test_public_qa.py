"""Real data-preparation risks: leakage, changing supervision, and partial joins."""
from argparse import Namespace
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_public_qa as prep


class PublicQATests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, rows):
        path = self.root / name
        prep.write_jsonl(path, rows)
        return path

    def test_preserves_all_aliases_and_the_public_first_target(self):
        raw = [{"id": "triviaqa1", "content": "A question?", "label": ["QUEENSTOWN", "Queenstown", "Cobh"]},
               {"id": "asqa1", "content": "An ambiguous question?", "label": ["first", "second"]}]
        train, dev, stats = prep.export_rows(raw, dev_fraction=0)
        self.assertEqual(dev, [])
        rows = {r["id"]: r for r in train}
        self.assertEqual(rows["triviaqa1"]["answers"], raw[0]["label"])
        self.assertEqual(rows["asqa1"]["answers"], raw[1]["label"])
        self.assertNotIn("teacher_output", rows["triviaqa1"])
        self.assertEqual(stats["train_by_source"], {"triviaqa": 1, "asqa": 1})

    def test_excludes_by_native_id_and_normalized_question_across_sources(self):
        excluded = self.write("eval.jsonl", [{"id": "nq_open0", "question": "different"},
                                            {"id": "native-hotpot-id", "query": "WHO won?"}])
        raw = [{"id": "nq_open0", "content": "keep?", "label": ["a"]},
               {"id": "asqa0", "content": "Who won！", "label": ["b"]},
               {"id": "hotpotqa0", "content": "Another", "label": ["c"]}]
        train, _, stats = prep.export_rows(raw, dev_fraction=0, exclude_paths=[excluded])
        self.assertEqual([r["id"] for r in train], ["hotpotqa0"])
        self.assertEqual(stats["counts"]["evaluation_overlap_removed"], 2)
        self.assertTrue(stats["evaluation_exclusion_checked"])

    def test_split_and_sampling_are_independent_of_input_order(self):
        raw = [{"id": f"nq_open{i}", "content": f"Question {i}?", "label": [str(i)]} for i in range(100)]
        raw.append({"id": "asqa1", "content": "question 1！", "label": ["other"]})
        a = prep.export_rows(raw, seed=7, dev_fraction=.3, limit=20)
        b = prep.export_rows(list(reversed(raw)), seed=7, dev_fraction=.3, limit=20)
        self.assertEqual(a, b)
        self.assertEqual(len(a[0]), 20)
        train_queries = {prep.normalized_question(r["query"]) for r in a[0]}
        dev_queries = {prep.normalized_question(r["query"]) for r in a[1]}
        self.assertFalse(train_queries & dev_queries)

    def test_rejects_duplicate_ids_and_unknown_source(self):
        value = {"id": "nq_open1", "content": "q", "label": ["a"]}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prep.export_rows([value, value])
        with self.assertRaisesRegex(ValueError, "unknown public source"):
            prep.export_rows([{**value, "id": "popqa1"}])
        train, _, stats = prep.export_rows([{**value, "content": None}, {**value, "id": "nq_open2", "label": []}])
        self.assertFalse(train)
        self.assertEqual(stats["counts"]["invalid_rows"], 2)

    def setup_attach(self, retrieval):
        queries = self.root / "questions"
        queries.mkdir()
        self.rows = [{"id": f"nq_open{i}", "source_id": f"nq_open{i}", "query": f"Q{i}?",
                      "answers": ["gold", "alias"], "source": "nq_open", "source_split": "train"} for i in range(2)]
        prep.write_jsonl(queries / "train.queries.jsonl", self.rows)
        prep.write_jsonl(queries / "dev.queries.jsonl", [])
        return Namespace(queries_dir=str(queries), retrieval_jsonl=[str(self.write("retrieval.jsonl", retrieval))],
                         corpus_jsonl=[], cache_manifest=None, out_dir=str(self.root / "ready"), max_docs=2)

    def attach(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            prep.attach_command(args)

    def test_attach_preserves_ranks_and_gold_despite_teacher_or_labels_in_retrieval(self):
        args = self.setup_attach([
            {"id": "nq_open0", "query": "Q0?", "documents": ["doc 2", "doc 1", "doc 3"], "teacher_output": "different", "answers": ["wrong"]},
            {"id": "server-nq-1", "question": "q1！", "documents": ["doc 2", {"title": "T", "text": "body"}]}])
        self.attach(args)
        out = Path(args.out_dir)
        rows = list(prep.read_jsonl(out / "train.jsonl"))
        self.assertEqual(rows[0]["retrieved_doc_ids"], [prep.content_id("doc 2"), prep.content_id("doc 1")])
        self.assertEqual(rows[0]["answers"], ["gold", "alias"])
        self.assertNotIn("teacher_output", rows[0])
        self.assertEqual(rows[1]["retrieval_join"], "normalized_question")
        self.assertEqual(len(list(prep.read_jsonl(out / "corpus.jsonl"))), 3)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertTrue(manifest["training_ready"])
        self.assertEqual(manifest["documents_to_compress"], 3)

    def test_incomplete_retrieval_cannot_silently_change_the_training_mixture(self):
        args = self.setup_attach([{"id": "nq_open0", "documents": ["a", "b"]}])
        with self.assertRaises(SystemExit):
            self.attach(args)
        out = Path(args.out_dir)
        self.assertFalse((out / "train.jsonl").exists())
        missing = list(prep.read_jsonl(out / "missing_retrieval.jsonl"))
        self.assertEqual(missing[0]["source_id"], "nq_open1")
        self.assertEqual(missing[0]["missing_reason"], "no_retrieval_row")

    def test_id_only_rows_need_a_real_corpus_or_cache(self):
        args = self.setup_attach([{"id": f"nq_open{i}", "retrieved_doc_ids": ["d1", "d2"]} for i in range(2)])
        with self.assertRaises(SystemExit):
            self.attach(args)
        self.assertEqual({r["doc_id"] for r in prep.read_jsonl(Path(args.out_dir) / "missing_documents.jsonl")}, {"d1", "d2"})
        args.out_dir = str(self.root / "cached")
        manifest = self.root / "cache.json"
        manifest.write_text(json.dumps({"documents": {"d1": {}, "d2": {}}}))
        args.cache_manifest = str(manifest)
        self.attach(args)
        stats = json.loads((Path(args.out_dir) / "manifest.json").read_text())
        self.assertEqual(stats["documents_to_compress"], 0)

    def test_external_corpus_ids_are_preserved(self):
        args = self.setup_attach([{"id": f"nq_open{i}", "doc_ids": ["d1", "d2"]} for i in range(2)])
        args.corpus_jsonl = [str(self.write("existing-corpus.jsonl", [{"doc_id": "d2", "text": "two"}, {"doc_id": "d1", "text": "one"}]))]
        self.attach(args)
        rows = list(prep.read_jsonl(Path(args.out_dir) / "train.jsonl"))
        self.assertEqual(rows[0]["retrieved_doc_ids"], ["d1", "d2"])

    def test_conflicting_id_or_ambiguous_question_match_is_rejected(self):
        values = [{"id": "nq_open0", "question": "wrong", "documents": ["a"]}]
        index = prep.RetrievalIndex([self.write("r1.jsonl", values)])
        row = {"id": "nq_open0", "source_id": "nq_open0", "query": "right"}
        with self.assertRaisesRegex(ValueError, "question differs"):
            index.get(row)
        index = prep.RetrievalIndex([self.write("r2.jsonl", [{"id": "x", "query": "right"}, {"id": "y", "query": "right?"}])])
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            index.get(row)
        same = [{"id": "x", "query": "right", "doc_ids": ["d1", "d2"]},
                {"id": "y", "query": "right?", "doc_ids": ["d1", "d2"]}]
        index = prep.RetrievalIndex([self.write("r3.jsonl", same)])
        self.assertEqual(index.get(row)[1], "normalized_question")
        index = prep.RetrievalIndex([self.write("r4.jsonl", [same[0], {**same[1], "doc_ids": ["d2", "d1"]}])])
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            index.get(row)

    def test_duplicate_references_preserve_ranks_and_text_id_conflicts_are_rejected(self):
        repeated_corpus = {}
        ids = prep.retrieve_documents({"documents": ["same", "same"]}, 2, repeated_corpus)
        self.assertEqual(ids, [prep.content_id("same")] * 2)
        self.assertEqual(len(repeated_corpus), 1)
        corpus = {}
        prep.retrieve_documents({"documents": [{"doc_id": "d1", "text": "first"}]}, 1, corpus)
        with self.assertRaisesRegex(ValueError, "conflicting text"):
            prep.retrieve_documents({"documents": [{"doc_id": "d1", "text": "other"}]}, 1, corpus)
        with self.assertRaisesRegex(ValueError, "counts disagree"):
            prep.retrieve_documents({"documents": ["a"], "retrieved_doc_ids": ["x", "y"]}, 1, {})


@unittest.skipUnless(importlib.util.find_spec("torch"), "training-interface checks require torch")
class TrainingInterfaceTests(unittest.TestCase):
    def test_caps_are_shared_by_query_and_no_query_arms(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from config import get_config
        from src.train import apply_overrides, build_args
        for mode in ("conditioned", "none"):
            with patch.object(sys, "argv", ["train", "--preset", "pisco_shared_projector", "--projector_query_mode", mode,
                                           "--max_answer_len", "64", "--gen_max_new_tokens", "128"]):
                cfg = apply_overrides(get_config("pisco_shared_projector"), build_args())
            self.assertEqual(cfg.data.max_answer_len, 64)
            self.assertEqual(cfg.train.gen_max_new_tokens, 128)
            self.assertFalse(cfg.data.prefer_teacher_output)

    def test_existing_defaults_survive_and_invalid_caps_are_rejected(self):
        from config import get_config
        from src.train import apply_overrides, build_args
        with patch.object(sys, "argv", ["train"]):
            cfg = apply_overrides(get_config("pisco_shared_projector"), build_args())
        self.assertEqual(cfg.data.max_answer_len, 48)
        self.assertEqual(cfg.train.gen_max_new_tokens, 32)
        for flag in ("--max_answer_len", "--gen_max_new_tokens"):
            with patch.object(sys, "argv", ["train", flag, "0"]):
                with self.assertRaisesRegex(ValueError, "caps must be positive"):
                    apply_overrides(get_config("pisco_shared_projector"), build_args())

    def test_exported_answers_reach_the_existing_training_target(self):
        from types import SimpleNamespace
        from src.data import adapt_row
        raw = [{"id": "triviaqa1", "content": "Q?", "label": ["canonical", "alias"]}]
        train, _, _ = prep.export_rows(raw, dev_fraction=0)
        row = adapt_row({**train[0], "retrieved_doc_ids": ["d1", "d2"]},
                        SimpleNamespace(max_docs=5, prefer_teacher_output=False), 0)
        self.assertEqual(row["answers"], ["canonical", "alias"])
        self.assertEqual(row["target"], "canonical")
        self.assertEqual(row["target_source"], "gold")


if __name__ == "__main__":
    unittest.main()

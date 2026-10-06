"""Evidence coverage, paired-set semantics and a truly model-free CLI."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_support_topk import analyze, load_predictions
from src.support_metrics import support_scores


def prediction(question, visible=True):
    return {"id": question, "support": {
        "logits": [10., 9., 8., 7., 6., 5., 4., 3., 2., 8.5],
        "labels": [1]+[0]*8+[1], "label_mask": [True]*10,
        "visible": [visible]+[None]*8+[visible],
        "doc_ids": [f"doc-{i}" for i in range(10)]}}


class SupportTopKTests(unittest.TestCase):
    def test_two_gold_coverage_and_hypothetical_memory_budget(self):
        rows = {"q": prediction("q")}
        report = analyze(rows, resamples=0)
        table = report["normal"]["all"]["topk"]
        self.assertEqual(table["2"]["recall"], .5)
        self.assertEqual(table["2"]["both"], 0.)
        self.assertEqual(table["4"]["recall"], 1.)
        self.assertEqual(table["4"]["both"], 1.)
        self.assertEqual(table["6"]["mean_selected_memory_tokens"], 48.)

    def test_visibility_filters_questions_not_original_gold_labels(self):
        rows = {"a": prediction("a"), "b": prediction("b", False)}
        # Ignoring an invisible gold during training must not inflate eval recall.
        rows["b"]["support"]["loss_mask"] = [False]+[True]*9
        report = analyze(rows, resamples=0)
        self.assertEqual(report["normal"]["all"]["questions"], 2)
        self.assertEqual(report["normal"]["all_gold_visible"]["questions"], 1)
        self.assertEqual(report["normal"]["all"]["topk"]["2"]["recall"], .5)

    def test_query_swap_compares_sets_not_ranked_lists(self):
        normal = {"q": prediction("q")}
        mismatch = copy.deepcopy(normal)
        mismatch["q"]["support"]["logits"][0:2] = [9., 10.]
        report = analyze(normal, mismatch, resamples=20)
        swap = report["query_swap"]["all"]["2"]
        self.assertEqual(swap["same_selected_set_fraction"], 1.)
        self.assertEqual(swap["recall_mismatch_minus_normal"]["ci95"], [0., 0.])

    def test_padding_ties_variable_k_and_non_two_gold_denominator(self):
        scores = support_scores([2., 2., float("nan")], [1, 0, 0], [True, True, False])
        self.assertEqual(scores["recall_at_2"], 1.)
        self.assertIsNone(scores["both_at_4"])
        report = analyze({"q": {"id": "q", "support": {
            "logits": [2., 2., float("nan")], "labels": [1, 0, 0],
            "label_mask": [True, True, False]}}}, resamples=0)
        self.assertEqual(report["normal"]["all"]["both_questions"], 0)
        self.assertEqual(report["normal"]["all"]["topk"]["6"]["mean_selected_memory_tokens"], 16.)
        tied = support_scores([1., 1., 1.], [0, 0, 1], [True]*3, topks=(2,))
        self.assertEqual(tied["recall_at_2"], 0.)

    def test_missing_gold_is_skipped_and_empty_groups_are_defined(self):
        rows = {"a": {"id": "a"}, "b": {"id": "b", "support": {
            "logits": [1.], "labels": [0]}}}
        report = analyze(rows, resamples=0)
        self.assertEqual(report["skipped_questions"], 2)
        self.assertIsNone(report["normal"]["all"]["topk"]["4"]["recall"])

    def test_misaligned_pairs_and_nonfinite_valid_logits_are_rejected(self):
        normal = {"q": prediction("q")}
        for field, value in [("labels", [0]*10), ("doc_ids", list(reversed(normal["q"]["support"]["doc_ids"]))),
                             ("logits", [float("nan")]*10)]:
            mismatch = copy.deepcopy(normal)
            mismatch["q"]["support"][field] = value
            with self.assertRaises(ValueError):
                analyze(normal, mismatch, resamples=0)
        with self.assertRaisesRegex(ValueError, "id sets"):
            analyze(normal, {}, resamples=0)
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            support_scores([1.], [1, 0], [True])

    def test_duplicate_prediction_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"duplicate.json"
            path.write_text(json.dumps([prediction("q"), prediction("q")]))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_predictions(path)

    def test_cli_runs_without_site_packages_and_does_not_overwrite_inputs(self):
        script = Path(__file__).resolve().parents[1]/"scripts/analyze_support_topk.py"
        with tempfile.TemporaryDirectory() as root:
            source, target = Path(root)/"pred.json", Path(root)/"report.json"
            original = json.dumps([prediction("q")])
            source.write_text(original)
            cmd = [sys.executable, "-S", str(script), "--predictions", str(source),
                   "--bootstrap_resamples", "0", "--output_json", str(target)]
            result = subprocess.run(cmd, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(target.read_text())["normal"]["all"]["topk"]["4"]["both"], 1.)
            result = subprocess.run(cmd[:-1]+[str(source)], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(source.read_text(), original)


if __name__ == "__main__":
    unittest.main()

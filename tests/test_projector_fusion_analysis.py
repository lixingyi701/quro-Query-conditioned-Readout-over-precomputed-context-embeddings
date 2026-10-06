"""Paired report must preserve direction, prompts and full example sets."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analyze_projector_fusion import analyze, paired_summary


class FusionAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.roots = {mode: Path(self.tmp.name)/mode for mode in ("additive", "film")}
        for mode, root in self.roots.items():
            root.mkdir()
            rows = [{"id": str(i), "query": "correct question", "golds": ["answer"],
                     "em": 0., "f1": .2+.2*i+(.1 if mode == "film" else 0.), "substring": 0.,
                     "support": {"doc_ids": ["a", "b"], "labels": [1, 0],
                                 "label_mask": [True, True], "visible": [True, None]}}
                    for i in range(2)]
            self.write(root/"predictions_dev_D0_Bfull.json", rows)
            swapped = copy.deepcopy(rows)
            for row in swapped:
                row["readout_query"] = "wrong question"
                row["f1"] -= .2 if mode == "film" else .05
            self.write(root/"predictions_dev_mismatch-q_D0_Bfull.json", swapped)
            self.write(root/"result.json", {"projector_fusion": mode,
                                            "metrics": {"dev|D0|B=full": {"fusion_gamma_rms": .1}}})

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, path, content):
        path.write_text(json.dumps(content), encoding="utf-8")

    def run_analysis(self):
        return analyze(self.roots["additive"], self.roots["film"], bootstrap=30)

    def test_paired_direction_and_drop_difference(self):
        report = self.run_analysis()
        self.assertAlmostEqual(report["film_minus_additive"]["f1"]["difference"], .1)
        for bound in report["film_minus_additive"]["f1"]["ci95"]:
            self.assertAlmostEqual(bound, .1)
        self.assertAlmostEqual(report["query_swap"]["film_minus_additive_drop"]["difference"], .15)
        self.assertEqual(report["fusion_diagnostics"]["film"]["fusion_gamma_rms"], .1)
        self.assertEqual(paired_summary([0., 0.], bootstrap=30)["ci95"], [0., 0.])

    def test_missing_duplicate_or_mismatched_prompts_rejected(self):
        path = self.roots["film"]/"predictions_dev_D0_Bfull.json"
        original = json.loads(path.read_text())
        for altered, message in ((original[:1], "ID sets"),
                                 (original+original[:1], "duplicate")):
            self.write(path, altered)
            with self.assertRaisesRegex(ValueError, message):
                self.run_analysis()
        altered = copy.deepcopy(original)
        altered[0]["query"] = "different decoder question"
        self.write(path, altered)
        with self.assertRaisesRegex(ValueError, "query"):
            self.run_analysis()

    def test_query_control_must_match_and_keep_decoder_prompt_and_documents(self):
        path = self.roots["film"]/"predictions_dev_mismatch-q_D0_Bfull.json"
        original = json.loads(path.read_text())
        for field, value, message in (("readout_query", "another wrong question", "readout query"),
                                      ("decoder_query", "also changed decoder", "decoder_query")):
            altered = copy.deepcopy(original)
            altered[0][field] = value
            self.write(path, altered)
            with self.assertRaisesRegex(ValueError, message):
                self.run_analysis()
        altered = copy.deepcopy(original)
        altered[0]["support"]["doc_ids"] = ["b", "a"]
        self.write(path, altered)
        with self.assertRaisesRegex(ValueError, "doc_ids"):
            self.run_analysis()

    def test_cli_runs_without_site_packages_and_cannot_overwrite_predictions(self):
        script = Path(__file__).resolve().parents[1]/"scripts/analyze_projector_fusion.py"
        output = Path(self.tmp.name)/"report.json"
        args = [sys.executable, "-S", str(script), "--additive_run", str(self.roots["additive"]),
                "--film_run", str(self.roots["film"]), "--bootstrap", "30", "--output_json", str(output)]
        subprocess.run(args, check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(output.read_text())["film_minus_additive"]["f1"]["n"], 2)
        target = self.roots["film"]/"result.json"
        before = target.read_bytes()
        result = subprocess.run(args[:-1]+[str(target)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(target.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

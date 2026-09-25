"""Deterministic report routing, layout and evaluation checks."""
import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from zemi.reporting import DefaultReportRenderer, ReportWriter, _replace_report
from zemi.dataset import table_evaluator
from zemi import env


class ReportingTests(unittest.TestCase):
    def setUp(self):
        env.path.tmp.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=env.path.tmp)
        self.addCleanup(self.tmp.cleanup)
        self.writer = ReportWriter(Path(self.tmp.name) / "run260924-120000")
        self.renderer = DefaultReportRenderer()

    def test_fragment_order_replacement_and_atomic_file(self):
        writer = self.writer
        ref = writer.register_module("first", filename="one.ipynb")
        writer.write_module_errors("first", "## Errors\n\nFirst error")
        writer.write_module_results("first", "## Results\n\nOld")
        writer.write_module_results("first", "## Results\n\nNew")
        doc = (writer.root / ref.path).read_text(encoding="utf-8")
        self.assertEqual(doc.count("## Results"), 1)
        self.assertNotIn("Old", doc)
        self.assertLess(doc.index("## Results"), doc.index("## Errors"))
        self.assertFalse((writer.root / f".{ref.path}.tmp").exists())

    def test_unchanged_fragment_does_not_replace_file_again(self):
        writer = self.writer
        writer.register_module("m")
        writer.write_module_header("m", "Header")
        with patch("zemi.reporting.os.replace", wraps=os.replace) as replace:
            writer.write_module_header("m", "Header")
        replace.assert_not_called()

    def test_transient_windows_lock_retries_atomic_replace(self):
        target = Path(self.tmp.name) / "report.md"
        tmp = Path(self.tmp.name) / ".report.md.tmp"
        target.write_text("old", encoding="utf-8")
        tmp.write_text("new", encoding="utf-8")
        calls = 0
        def transient(source, destination):
            nonlocal calls
            calls += 1
            if calls <= 2:
                error = PermissionError(13, "File is temporarily locked")
                error.winerror = 5
                raise error
            os_replace(source, destination)
        os_replace = os.replace
        with patch("zemi.reporting.os.replace", side_effect=transient), patch("zemi.reporting.time.sleep"):
            _replace_report(tmp, target)
        self.assertEqual(calls, 3)
        self.assertEqual(target.read_text(encoding="utf-8"), "new")
        self.assertFalse(tmp.exists())

    def test_persistent_windows_lock_keeps_previous_report_intact(self):
        target = Path(self.tmp.name) / "report.md"
        tmp = Path(self.tmp.name) / ".report.md.tmp"
        target.write_text("old", encoding="utf-8")
        tmp.write_text("new", encoding="utf-8")
        error = PermissionError(13, "File is locked")
        error.winerror = 32
        with patch("zemi.reporting.os.replace", side_effect=error), patch("zemi.reporting.time.sleep"):
            with self.assertRaises(PermissionError):
                _replace_report(tmp, target)
        self.assertEqual(target.read_text(encoding="utf-8"), "old")

    def test_safe_colliding_names_and_registered_relative_links(self):
        writer = self.writer
        first = writer.register_module("index", filename="a.ipynb")
        second = writer.register_module("index!", filename="b.ipynb")
        self.assertNotEqual(first.path.casefold(), second.path.casefold())
        self.assertNotEqual(first.path, "index.md")
        for ref in (first, second):
            self.assertTrue((writer.root / ref.path).is_file())
            self.assertNotIn("..", ref.path)
        sample = writer.register_sample("index", "../same")
        run = writer.register_run("index", "../same", sample_id="../same")
        self.assertTrue((writer.root / sample.path).is_file())
        self.assertTrue((writer.root / run.path).is_file())
        self.assertTrue(writer.href(sample, first).startswith("../"))
        self.assertIsNone(writer.artifact_href(first, "missing.ipynb"))

    def test_dynamic_parameters_metrics_and_run_anchor(self):
        writer = self.writer
        writer.register_module("detect", filename="detect.ipynb", optimized=True)
        writer.register_sample("detect", "s-1")
        writer.register_sample("detect", "s-2")
        writer.register_run("detect", "r-1", sample_id="s-1")
        samples = [
            {"id": "s-1", "params": {"temperature": 0}, "metrics": {"accuracy": .5},
             "score": .5, "status": "succeeded", "runs": [{"run_id": "r-1", "status": "succeeded"}]},
            {"id": "s-2", "params": {"temperature": 1}, "metrics": {"coverage": .8},
             "score": .8, "status": "failed", "runs": []},
        ]
        fragment = self.renderer.render_module_samples_summary(samples=samples,
            param_names=["temperature"], writer=writer, module_id="detect")
        self.assertIn("Parameters<br>temperature", fragment)
        self.assertIn("Metrics<br>accuracy / coverage", fragment)
        self.assertIn("0.5 / —", fragment)
        self.assertIn("— / 0.8", fragment)
        runs = self.renderer.render_module_runs_summary(samples=samples, writer=writer, module_id="detect")
        self.assertIn("### [Sample 1]", runs)
        self.assertIn("### [Sample 2]", runs)
        self.assertIn("#sample-1", fragment)
        self.assertIn("#sample-2", fragment)

    def test_runs_report_shows_selected_outputs_and_sample_parameters(self):
        writer = self.writer
        writer.register_module("detect", optimized=True)
        writer.register_sample("detect", "s-1")
        writer.register_run("detect", "r-1", sample_id="s-1")
        samples = [{"id": "s-1", "params": {"encoding_format": "cell_all"},
                    "runs": [{"run_id": "r-1", "status": "succeeded", "duration": "5.0 s",
                              "prediction": {"ranges": ["A1:B2"], "raw_response": "large",
                                             "lm_time": 3.2, "item_tokens": 20},
                              "report_output_keys": ["ranges", "lm_time", "item_tokens"]}]}]
        report = self.renderer.render_module_runs_summary(samples=samples, writer=writer, module_id="detect")
        self.assertIn("[Sample 1](samples/", report)
        self.assertIn("**Parameters:** encoding_format = cell_all", report)
        self.assertIn("Outputs<br>ranges / LM Time / Item Tokens", report)
        self.assertIn('["A1:B2"] / 3.2 / 20', report)
        self.assertNotIn("raw_response", report)
        self.assertNotIn("| Run | Sample |", report)

    def test_secret_redaction_and_custom_fragment(self):
        writer = self.writer
        writer.add_secrets(["token-123"])
        ref = writer.register_module("m")
        writer.write_module_results("m", "## Custom\n\nresult token-123")
        doc = (writer.root / ref.path).read_text(encoding="utf-8")
        self.assertIn("## Custom", doc)
        self.assertIn("***", doc)
        self.assertNotIn("token-123", doc)

    def test_running_run_is_replaced_by_error_and_linked_from_sample(self):
        writer = self.writer
        writer.register_module("m", optimized=True)
        sample = writer.register_sample("m", "s")
        run = writer.register_run("m", "r", sample_id="s")
        writer.write_run_report("m", "r", "**Status:** running", sample_id="s")
        writer.write_run_report("m", "r", "**Status:** failed\n\n## Errors\n\nmodel unavailable", sample_id="s")
        run_text = (writer.root / run.path).read_text(encoding="utf-8")
        sample_text = (writer.root / sample.path).read_text(encoding="utf-8")
        self.assertNotIn("**Status:** running", run_text)
        self.assertIn("model unavailable", run_text)
        self.assertIn("../samples/", run_text)
        self.assertIn("../runs/", sample_text)

    def test_set_evaluation_permutation_duplicates_and_extra(self):
        item = {"id": "book::sheet", "input": {}, "ground_truth": ["A1:B2", "D1:E5"]}
        runs = [{"item": item, "prediction": {"ranges": ["D1:E5", "A1:B2", "A1:B2"]}},
                {"item": item, "prediction": {"ranges": ["A1:B2", "D1:E5", "F1:G2", "F1:G2"]}}]
        metrics, details = table_evaluator(type("Trial", (), {"runs": runs})(), params={})
        self.assertEqual((runs[0]["metrics"]["tp"], runs[0]["metrics"]["fp"], runs[0]["metrics"]["fn"]), (2, 0, 0))
        self.assertTrue(runs[0]["metrics"]["exact_match"])
        self.assertEqual((runs[1]["metrics"]["tp"], runs[1]["metrics"]["fp"]), (2, 1))
        self.assertFalse(runs[1]["metrics"]["exact_match"])
        self.assertEqual(metrics["tp"], 4)

    def test_mixed_job_single_and_optimizer_modes(self):
        writer = self.writer
        writer.register_module("single", filename="single.ipynb")
        writer.register_module("start", filename="start.ipynb", optimized=True)
        writer.register_module("search", filename="search.ipynb", optimized=True)
        modules = [
            {"id": "single", "name": "single.ipynb", "optimized": False, "status": "succeeded", "outputs": {}},
            {"id": "start", "name": "start.ipynb", "optimized": True, "mode": "start_only", "status": "succeeded", "samples": [], "runs": []},
            {"id": "search", "name": "search.ipynb", "optimized": True, "mode": "optimize", "status": "running", "samples": [], "runs": []},
        ]
        text = self.renderer.render_module_summary(modules=modules, writer=writer)
        self.assertIn("### Single Execution", text)
        self.assertIn("### Execution With Optimizer", text)
        self.assertIn("`start_only`", text)
        self.assertIn("`optimize`", text)
        self.assertIn("Samples<br>(OK / Total)", text)
        self.assertIn("Runs<br>(OK / Total)", text)


if __name__ == "__main__":
    unittest.main()

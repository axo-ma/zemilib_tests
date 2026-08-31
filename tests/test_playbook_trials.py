from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from zemi.component import ZemiComponent, _summarize_trials
from zemi.playbook import PLAYBOOK_OUTPUT_MIME, output_params, validate_output_params

from tests.test_component import ComponentFixture


def _config(body: str, *, stop_on_error: bool = True) -> str:
    flag = "true" if stop_on_error else "false"
    return f"[pipeline_params]\n[component_params]\nstop_on_error = {flag}\n{body.strip()}\n"


class ExpansionTests(ComponentFixture):
    def setUp(self) -> None:
        super().setUp()
        self.write_notebook("same.ipynb")
        self.write_notebook("other.ipynb")

    def test_configuration_without_each_is_one_backward_compatible_trial(self) -> None:
        self.write_default(_config('''
[[playbooks_params]]
playbook_name = "same.ipynb"
[playbooks_params.playbook_params]
temperature = 0.2
stops = ["END", "STOP"]
[playbooks_params.playbook_params.options]
top_k = 20
'''))
        component = ZemiComponent()
        self.assertEqual(len(component.playbooks), 1)
        self.assertEqual(component.playbooks[0].params, {
            "temperature": 0.2,
            "stops": ["END", "STOP"],
            "options": {"top_k": 20},
        })
        component.close()

    def test_cartesian_product_literal_values_and_stable_ids(self) -> None:
        self.write_default(_config('''
[[playbooks_params]]
playbook_name = "same.ipynb"
[playbooks_params.playbook_params]
temperature = { each = [0.0, 0.5] }
seed = { each = [1, 2, 3] }
stops = ["END", "STOP"]
options = { top_k = 20 }

[[playbooks_params]]
playbook_name = "same.ipynb"
[playbooks_params.playbook_params]
temperature = 0.7

[[playbooks_params]]
playbook_name = "other.ipynb"
[playbooks_params.playbook_params]
mode = "literal"
'''))
        component = ZemiComponent()
        self.assertEqual(len(component.playbooks), 8)
        self.assertEqual(
            [(p.params["temperature"], p.params["seed"]) for p in component.playbooks[:6]],
            [(0.0, 1), (0.0, 2), (0.0, 3), (0.5, 1), (0.5, 2), (0.5, 3)],
        )
        self.assertTrue(all(p.params["stops"] == ["END", "STOP"] for p in component.playbooks[:6]))
        ids = [p.trial_id for p in component.playbooks]
        paths = [p.output_path for p in component.playbooks]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(p.parent == component.run_directory / "notebooks" for p in paths))
        self.assertEqual(ids[0], "p001-t0001-same")
        self.assertEqual(ids[6], "p002-t0001-same")
        component.close()

    def test_invalid_each_wrappers_are_clear_configuration_errors(self) -> None:
        cases = [
            ('value = { each = [] }', "must not be empty"),
            ('value = { each = 1 }', "must be an array"),
            ('value = { each = [1], extra = 2 }', "only the 'each' key"),
        ]
        for declaration, message in cases:
            with self.subTest(declaration=declaration):
                self.write_default(_config(f'''
[[playbooks_params]]
playbook_name = "same.ipynb"
[playbooks_params.playbook_params]
{declaration}
'''))
                with self.assertRaisesRegex(ValueError, message):
                    ZemiComponent()


class StructuredOutputTests(ComponentFixture):
    def setUp(self) -> None:
        super().setUp()
        self.write_notebook("one.ipynb")
        self.write_default(_config('''
[[playbooks_params]]
playbook_name = "one.ipynb"
'''))

    def tearDown(self) -> None:
        import zemi.playbook
        zemi.playbook._published = False
        super().tearDown()

    def test_output_params_publishes_custom_mime_and_rejects_second_call(self) -> None:
        import zemi.playbook
        zemi.playbook._published = False
        display = Mock()
        with patch("IPython.display.display", display):
            output_params({"score": 1, "nested": [1, {"ok": True}]})
            with self.assertRaisesRegex(RuntimeError, "only once"):
                output_params({"score": 2})
        display.assert_called_once_with(
            {PLAYBOOK_OUTPUT_MIME: {"score": 1, "nested": [1, {"ok": True}]}},
            raw=True,
        )

    def test_output_validation(self) -> None:
        with self.assertRaisesRegex(TypeError, "mapping"):
            validate_output_params([1, 2])
        with self.assertRaisesRegex(TypeError, "keys"):
            validate_output_params({1: "bad"})
        with self.assertRaisesRegex(ValueError, "finite JSON"):
            validate_output_params({"bad": float("nan")})
        with self.assertRaisesRegex(ValueError, "JSON-serializable"):
            validate_output_params({"bad": object()})

    def _write_output_notebook(self, output_values: list[object]) -> ZemiComponent:
        component = ZemiComponent()
        playbook = component.playbooks[0]
        playbook.output_path.parent.mkdir(parents=True, exist_ok=True)
        outputs = [
            {"output_type": "display_data", "metadata": {}, "data": {PLAYBOOK_OUTPUT_MIME: value}}
            for value in output_values
        ]
        playbook.output_path.write_text(json.dumps({
            "cells": [{"cell_type": "code", "execution_count": 1, "id": "out", "metadata": {}, "outputs": outputs, "source": []}],
            "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
        }), encoding="utf-8")
        return component

    def test_runner_extracts_absent_one_and_multiple_mime_outputs(self) -> None:
        component = self._write_output_notebook([])
        self.assertEqual(component.playbooks[0]._extract_output_params(), {})
        playbook = component.playbooks[0]
        outputs = [{"output_type": "display_data", "metadata": {}, "data": {PLAYBOOK_OUTPUT_MIME: {"answer": 42, "items": [1, 2]}}}]
        document = json.loads(playbook.output_path.read_text(encoding="utf-8"))
        document["cells"][0]["outputs"] = outputs
        playbook.output_path.write_text(json.dumps(document), encoding="utf-8")
        self.assertEqual(playbook._extract_output_params(), {"answer": 42, "items": [1, 2]})
        document["cells"][0]["outputs"].append(
            {"output_type": "display_data", "metadata": {}, "data": {PLAYBOOK_OUTPUT_MIME: {"answer": 43}}}
        )
        playbook.output_path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "more than once"):
            playbook._extract_output_params()
        component.close()


class ExecutionAndReportTests(ComponentFixture):
    def setUp(self) -> None:
        super().setUp()
        self.write_notebook("one.ipynb")
        self.write_notebook("two.ipynb")

    def _two_playbooks(self, stop: bool) -> ZemiComponent:
        self.write_default(_config('''
[[playbooks_params]]
playbook_name = "one.ipynb"
[[playbooks_params]]
playbook_name = "two.ipynb"
''', stop_on_error=stop))
        return ZemiComponent()

    def test_stop_on_error_true_and_false(self) -> None:
        for stop, expected_calls in ((True, 1), (False, 2)):
            with self.subTest(stop=stop):
                component = self._two_playbooks(stop)
                calls = []
                def run(playbook):
                    calls.append(playbook.trial_id)
                    if len(calls) == 1:
                        raise RuntimeError("failed")
                for playbook in component.playbooks:
                    playbook.run = lambda p=playbook: run(p)
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    component.run()
                component.close()
                self.assertEqual(len(calls), expected_calls)
                self.assertEqual(component.report.data["status"], "failed")

    def test_atomic_report_is_updated_between_trials_and_html_is_offline(self) -> None:
        component = self._two_playbooks(False)
        papermill = Mock(); call_count = 0
        dangerous = '</script><img src=x onerror="alert(1)">'

        def execute(_source, output, **_kwargs):
            nonlocal call_count
            if call_count == 1:
                snapshot = json.loads(component.report.path.read_text(encoding="utf-8"))
                self.assertEqual(snapshot["trials"][0]["status"], "succeeded")
            payload = {"score": call_count + 1, "ok": call_count == 0, "nested": {"value": dangerous}}
            Path(output).write_text(json.dumps({
                "cells": [{"cell_type": "code", "execution_count": 1, "id": f"c{call_count}", "metadata": {}, "outputs": [{"output_type": "display_data", "metadata": {}, "data": {PLAYBOOK_OUTPUT_MIME: payload}}], "source": []}],
                "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
            }), encoding="utf-8")
            call_count += 1

        papermill.execute_notebook.side_effect = execute
        with patch.dict(sys.modules, {"papermill": papermill}):
            component.run()
        component.close()
        report = json.loads(component.report.path.read_text(encoding="utf-8"))
        self.assertEqual([t["status"] for t in report["trials"]], ["succeeded", "succeeded"])
        self.assertEqual(report["trials"][0]["output_params"]["nested"]["value"], dangerous)
        html = component.report.html_path.read_text(encoding="utf-8")
        for section in ("Overview", "Summary", "Runs", "Outputs", "Errors"):
            self.assertIn(f">{section}<", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("http://", html)
        self.assertNotIn(dangerous, html)
        self.assertIn("notebooks/p001-t0001-one.ipynb", html)
        self.assertIn('id="search"', html)
        self.assertIn('id="playbook-filter"', html)
        self.assertIn('id="status-filter"', html)
        self.assertIn('id="parameter-filter"', html)
        self.assertIn('id="parameter-value"', html)
        self.assertFalse((component.run_directory / ".report.json.tmp").exists())

    def test_summary_ignores_seed_and_aggregates_boolean_numeric_outputs(self) -> None:
        trials = [
            {"playbook_name": "one.ipynb", "status": "succeeded", "input_params": {"model": "m", "seed": 1}, "output_params": {"ok": True, "score": 1, "label": "a", "nested": [1]}},
            {"playbook_name": "one.ipynb", "status": "succeeded", "input_params": {"model": "m", "seed": 2}, "output_params": {"ok": False, "score": 3, "label": "b", "nested": [2]}},
            {"playbook_name": "one.ipynb", "status": "failed", "input_params": {"model": "m", "seed": 3}, "output_params": {}},
        ]
        summary = _summarize_trials(trials)
        self.assertEqual(len(summary), 1)
        group = summary[0]
        self.assertEqual(group["input_params"], {"model": "m"})
        self.assertEqual((group["total"], group["succeeded"], group["failed"]), (3, 2, 1))
        self.assertEqual(group["output_aggregates"]["ok"]["true_count"], 1)
        self.assertEqual(group["output_aggregates"]["ok"]["true_rate"], 0.5)
        self.assertEqual(group["output_aggregates"]["score"]["mean"], 2.0)
        self.assertEqual(group["output_aggregates"]["score"]["stddev"], 1.0)
        self.assertNotIn("label", group["output_aggregates"])
        self.assertNotIn("nested", group["output_aggregates"])


if __name__ == "__main__":
    unittest.main()

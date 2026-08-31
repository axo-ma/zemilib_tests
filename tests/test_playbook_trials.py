from __future__ import annotations

import json
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from zemi.component import ZemiComponent, _summarize_trials
from zemi import env
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
            ('value = { each = [1], extra = 2 }', "exactly one mode key"),
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

    def test_select_and_each_resolve_in_toml_order_without_extra_trials(self) -> None:
        self.write_default(_config('''
[[playbooks_params]]
playbook_name = "same.ipynb"
[playbooks_params.playbook_params]
model = { select = ["small", "large"] }
temperature = { select = [0.1, 0.2] }
seed = { each = [1, 2] }
literal_array = ["A", "B"]
literal_table = { top_k = 20 }

[[playbooks_params]]
playbook_name = "other.ipynb"
[playbooks_params.playbook_params]
mode = { select = ["fast", "safe"] }
'''))
        with patch("builtins.input", side_effect=["2", "1", "2"]) as prompt:
            component = ZemiComponent()
        self.assertEqual(prompt.call_count, 3)
        self.assertEqual(len(component.playbooks), 3)
        self.assertEqual([p.params["seed"] for p in component.playbooks[:2]], [1, 2])
        self.assertTrue(all(p.params["model"] == "large" for p in component.playbooks[:2]))
        self.assertTrue(all(p.params["temperature"] == 0.1 for p in component.playbooks[:2]))
        self.assertEqual(component.playbooks[2].params["mode"], "safe")
        self.assertEqual(component.playbooks[0].params["literal_array"], ["A", "B"])
        self.assertEqual(component.playbooks[0].params["literal_table"], {"top_k": 20})
        self.assertEqual(list(component.playbooks[0].resolved_params), ["model", "temperature", "seed"])
        component.close()

    def test_invalid_select_and_interactive_failures_name_context(self) -> None:
        cases = [
            ('value = { select = [] }', None, "must not be empty"),
            ('value = { select = 1 }', None, "must be an array"),
            ('value = { select = [1], extra = 2 }', None, "exactly one mode key"),
            ('value = { each = [1], select = [2] }', None, "each/select"),
            ('value = { select = [1, 2] }', "bad", "Invalid selection.*same\\.ipynb.*value"),
            ('value = { select = [1, 2] }', EOFError(), "same\\.ipynb.*value.*interactive input is unavailable"),
        ]
        for declaration, input_result, message in cases:
            with self.subTest(declaration=declaration, input_result=input_result):
                self.write_default(_config(f'''\n[[playbooks_params]]\nplaybook_name = "same.ipynb"\n[playbooks_params.playbook_params]\n{declaration}\n'''))
                effect = input_result if isinstance(input_result, BaseException) else None
                kwargs = {"side_effect": effect} if effect else {"return_value": input_result}
                with patch("builtins.input", **kwargs), self.assertRaisesRegex((ValueError, RuntimeError), message):
                    ZemiComponent()


class ParameterReferenceTests(ComponentFixture):
    def setUp(self) -> None:
        super().setUp()
        self.write_notebook("same.ipynb")

    def component(self, declarations: str, params: str) -> ZemiComponent:
        self.write_default(_config(f"{declarations.strip()}\n[[playbooks_params]]\nplaybook_name = \"same.ipynb\"\n[playbooks_params.playbook_params]\n{params.strip()}"))
        return ZemiComponent()

    def test_scalar_composite_and_recursive_nested_refs(self) -> None:
        component = self.component('''
[param_buckets]
scalar = 7
array = [1, 2]
[param_buckets.options]
top_k = 20
[param_buckets.options.nested]
label = "ok"
''', '''
scalar = { ref = "param_buckets.scalar" }
array = { ref = "param_buckets.array" }
options = { ref = "param_buckets.options" }
[playbooks_params.playbook_params.deep]
copied = { ref = "param_buckets.options.nested" }
''')
        playbook = component.playbooks[0]
        self.assertEqual(playbook.params, {"scalar": 7, "array": [1, 2], "options": {"top_k": 20, "nested": {"label": "ok"}}, "deep": {"copied": {"label": "ok"}}})
        self.assertEqual(playbook.resolved_params["scalar"]["refs"], ["param_buckets.scalar"])
        self.assertEqual(playbook.resolved_params["deep"]["source"], "ref")
        component.close()

    def test_single_multiple_include_precedence_and_no_reserved_key(self) -> None:
        component = self.component('''
[param_buckets.model]
model = "small"
shared = "model"
[param_buckets.generation]
temperature = 0.5
shared = "generation"
''', '''
__include__ = [{ ref = "param_buckets.model" }, { ref = "param_buckets.generation" }]
shared = "local"
''')
        playbook = component.playbooks[0]
        self.assertEqual(playbook.params, {"model": "small", "shared": "local", "temperature": 0.5})
        self.assertNotIn("__include__", playbook.params)
        self.assertEqual(playbook.resolved_params["temperature"]["source"], "include")
        self.assertNotIn("shared", playbook.resolved_params)
        component.close()

    def test_component_arsenal_include_allows_local_lifecycle_override(self) -> None:
        component = self.component('''
[component_params.arsenal]
arsenal_config_path = "@comp/zemi/llm_curated_set_model_mode.toml"
arsenal_stop_before_playbook_begin = false
arsenal_stop_after_playbook_end = false
''', '''
__include__ = { ref = "component_params.arsenal" }
arsenal_stop_after_playbook_end = true
''')
        self.assertEqual(component.playbooks[0].params, {
            "arsenal_config_path": "@comp/zemi/llm_curated_set_model_mode.toml",
            "arsenal_stop_before_playbook_begin": False,
            "arsenal_stop_after_playbook_end": True,
        })
        self.assertNotIn("__include__", component.playbooks[0].params)
        self.assertNotIn(
            "arsenal_stop_after_playbook_end",
            component.playbooks[0].resolved_params,
        )
        self.assertEqual(
            component.arsenal_config_path,
            "@comp/zemi/llm_curated_set_model_mode.toml",
        )
        component.close()

    def test_playbook_cannot_override_shared_arsenal_config_path(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "arsenal_config_path must match the shared",
        ):
            self.component('''
[component_params.arsenal]
arsenal_config_path = "@comp/zemi/llm_curated_set_model_mode.toml"
arsenal_stop_before_playbook_begin = false
arsenal_stop_after_playbook_end = false
''', '''
__include__ = { ref = "component_params.arsenal" }
arsenal_config_path = "@comp/zemi/llm_curated_set_router_mode.toml"
''')

    def test_nested_include_and_bucket_in_bucket(self) -> None:
        component = self.component('''
[param_buckets.base]
a = 1
[param_buckets.extended]
__include__ = { ref = "param_buckets.base" }
b = 2
''', '''
[playbooks_params.playbook_params.nested]
__include__ = { ref = "param_buckets.extended" }
b = 3
''')
        self.assertEqual(component.playbooks[0].params, {"nested": {"a": 1, "b": 3}})
        self.assertEqual(component.playbooks[0].resolved_params["nested"]["refs"], ["param_buckets.extended", "param_buckets.base"])
        component.close()

    def test_refs_are_resolved_before_select_and_each(self) -> None:
        self.write_default(_config('''
[param_buckets]
models = ["small", "large"]
seeds = [1, 2]
[[playbooks_params]]
playbook_name = "same.ipynb"
[playbooks_params.playbook_params]
model = { select = [{ ref = "param_buckets.models" }, "fallback"] }
seed = { each = [{ ref = "param_buckets.seeds" }, 3] }
'''))
        with patch("builtins.input", return_value="1"):
            component = ZemiComponent()
        self.assertEqual([p.params["seed"] for p in component.playbooks], [[1, 2], 3])
        self.assertEqual(component.playbooks[0].params["model"], ["small", "large"])
        self.assertEqual(component.playbooks[0].resolved_params["model"]["source"], "select")
        self.assertEqual(component.playbooks[0].resolved_params["model"]["refs"], ["param_buckets.models"])
        component.close()

    def test_reference_validation_errors_are_contextual(self) -> None:
        cases = [
            ('[param_buckets]\nvalue = 1', 'x = { ref = "param_buckets.missing" }', "ref path.*was not found"),
            ('[param_buckets]\nvalue = 1', 'x = { ref = "param_buckets.value", extra = 2 }', "ref wrapper must contain exactly one"),
            ('[param_buckets]\nvalue = 1', '__include__ = { ref = "param_buckets.value" }', "does not resolve to a table"),
            ('[param_buckets]\nvalue = 1', '__include__ = "bad"', "must be a ref wrapper"),
            ('[param_buckets]\nvalue = 1', 'x = { ref = "param_buckets.value.child" }', "non-table"),
        ]
        for declarations, params, message in cases:
            with self.subTest(params=params):
                with self.assertRaisesRegex(ValueError, message):
                    self.component(declarations, params)

    def test_direct_and_include_cycles_are_rejected(self) -> None:
        cases = [
            ('[param_buckets]\na = { ref = "param_buckets.b" }\nb = { ref = "param_buckets.a" }', 'x = { ref = "param_buckets.a" }'),
            ('[param_buckets.a]\n__include__ = { ref = "param_buckets.b" }\n[param_buckets.b]\n__include__ = { ref = "param_buckets.a" }', '__include__ = { ref = "param_buckets.a" }'),
        ]
        for declarations, params in cases:
            with self.subTest(params=params), self.assertRaisesRegex(ValueError, "cyclic ref detected"):
                self.component(declarations, params)

    def test_papermill_receives_resolved_values_without_service_keys(self) -> None:
        component = self.component('[param_buckets.common]\nvalue = 42', '__include__ = { ref = "param_buckets.common" }')
        papermill = Mock()
        papermill.execute_notebook.side_effect = lambda _source, output, **_kwargs: Path(output).write_text(json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}), encoding="utf-8")
        with patch.dict(sys.modules, {"papermill": papermill}):
            component.playbooks[0].run()
        parameters = papermill.execute_notebook.call_args.kwargs["parameters"]
        self.assertEqual(parameters, {"value": 42})
        self.assertNotIn("__include__", json.dumps(parameters))
        self.assertNotIn("ref", parameters)
        component.close()


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
            {
                PLAYBOOK_OUTPUT_MIME: {"score": 1, "nested": [1, {"ok": True}]},
                "text/plain": '{\n  "score": 1,\n  "nested": [\n    1,\n    {\n      "ok": true\n    }\n  ]\n}',
            },
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
        outputs = [{"output_type": "display_data", "metadata": {}, "data": {PLAYBOOK_OUTPUT_MIME: {"answer": 42, "items": [1, 2]}, "text/plain": "not JSON and ignored"}}]
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
        for section in ("Overview", "Runs", "Summary", "Run details", "Errors"):
            self.assertIn(f">{section}<", html)
        positions = [html.index(f">{section}<") for section in ("Overview", "Runs", "Summary", "Run details", "Errors")]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn(">Outputs<", html)
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

    def test_summary_contains_only_counts_and_trial_ids(self) -> None:
        trials = [
            {"trial_id": "t1", "status": "succeeded", "input_params": {"seed": 1}, "output_params": {"score": 1}},
            {"trial_id": "t2", "status": "succeeded", "input_params": {"seed": 2}, "output_params": {"score": 3}},
            {"trial_id": "t3", "status": "failed", "input_params": {"seed": 3}, "output_params": {}},
        ]
        summary = _summarize_trials(trials)
        self.assertEqual(summary["counts"], {"total": 3, "succeeded": 2, "failed": 1, "running": 0})
        self.assertEqual(summary["trial_ids"]["total"], ["t1", "t2", "t3"])
        self.assertEqual(summary["trial_ids"]["succeeded"], ["t1", "t2"])
        self.assertEqual(summary["trial_ids"]["failed"], ["t3"])
        self.assertNotIn("input_params", json.dumps(summary))
        self.assertNotIn("output_aggregates", json.dumps(summary))

    def test_terminal_lists_only_resolved_parameter_origins(self) -> None:
        self.write_default(_config('''
[[playbooks_params]]
playbook_name = "one.ipynb"
[playbooks_params.playbook_params]
selected = { select = ["x", "y"] }
swept = { each = [3] }
literal = "quiet"
'''))
        with patch("builtins.input", return_value="2"):
            component = ZemiComponent()
        output = StringIO()
        with redirect_stdout(output):
            component.playbooks[0]._print_start()
        text_output = output.getvalue()
        self.assertIn(component.playbooks[0].trial_id, text_output)
        self.assertIn('selected [select] = "y"', text_output)
        self.assertIn("swept [each] = 3", text_output)
        self.assertNotIn("literal =", text_output)
        component.close()

    def test_complete_marker_appears_only_after_close_for_success_and_failure(self) -> None:
        for failed in (False, True):
            with self.subTest(failed=failed):
                env_path = None
                component = self._two_playbooks(False)
                env_path = component.run_directory
                self.assertFalse((env_path / "complete").exists())
                if failed:
                    component.report.record_failure(RuntimeError("recorded"))
                component.close()
                self.assertTrue((env_path / "complete").is_file())
                self.assertEqual((env_path / "complete").read_bytes(), b"")
                env.path.comp._runid = None


if __name__ == "__main__":
    unittest.main()

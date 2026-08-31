from __future__ import annotations

import json
import os
import runpy
import shutil
import subprocess
import sys
import tomllib
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from zemi import env
from zemi.component import ZemiComponent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOB_PATH = PROJECT_ROOT / "job.exp.py"


class ComponentFixture(unittest.TestCase):
    def setUp(self) -> None:
        env.path.tmp.mkdir(parents=True, exist_ok=True)
        self.root = env.path.tmp / f"zemi-component-test-{os.getpid()}-{id(self)}"
        self.root.mkdir()
        (self.root / ".zemicomp").write_text("", encoding="utf-8")
        (self.root / "params").mkdir()
        self.original_cwd = Path.cwd()
        os.chdir(self.root)
        env.path.comp._runid = None

    def tearDown(self) -> None:
        os.chdir(self.original_cwd)
        env.path.comp._runid = None
        shutil.rmtree(self.root, ignore_errors=True)

    def write_notebook(self, name: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "cells": [],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            ),
            encoding="utf-8",
        )
        return path

    def write_default(self, content: str) -> None:
        self.write_params("default_params.toml", content)

    def write_params(self, name: str, content: str) -> Path:
        path = self.root / "params" / name
        path.write_text(
            content,
            encoding="utf-8",
        )
        return path

    @staticmethod
    def write_executed_notebook(path: str | Path, duration: float = 1.23456) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "cells": [
                        {
                            "cell_type": "code",
                            "execution_count": 1,
                            "id": "executed-cell",
                            "metadata": {
                                "papermill": {
                                    "duration": duration,
                                    "status": "completed",
                                }
                            },
                            "outputs": [],
                            "source": ["value = 1"],
                        }
                    ],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            ),
            encoding="utf-8",
        )


class ComponentLoadingTests(ComponentFixture):
    def setUp(self) -> None:
        super().setUp()
        for name in ("one.ipynb", "two.ipynb", "three.ipynb"):
            self.write_notebook(name)
        self.default_content = """
[pipeline_params]
source = "default"
[pipeline_params.nested]
left = 1

[component_params]
stop_on_error = true

[[playbooks_params]]
playbook_name = "one.ipynb"
enabled = true
[playbooks_params.playbook_params]
model_name = "default"
[playbooks_params.playbook_params.nested]
left = 1

[[playbooks_params]]
playbook_name = "two.ipynb"
enabled = false
""".strip() + "\n"
        self.write_default(self.default_content)

    def write_alternative(self, name: str = "experiment.toml") -> Path:
        content = self.default_content.replace(
            "stop_on_error = true",
            'stop_on_error = true\ncomponent_name = "logical"',
        ).replace('model_name = "default"', 'model_name = "experiment"')
        return self.write_params(name, content)

    def test_single_file_is_selected_automatically(self) -> None:
        default = ZemiComponent()
        self.assertEqual(default.name, self.root.name)
        self.assertEqual(default.params_path.name, "default_params.toml")
        default.close()

    def test_explicit_file_selection_does_not_prompt(self) -> None:
        self.write_alternative()
        env.path.comp._runid = None
        with patch("builtins.input") as input_mock:
            component = ZemiComponent(
                params_file="@comp/params/experiment.toml"
            )

        input_mock.assert_not_called()
        self.assertEqual(component.name, "logical")
        self.assertEqual(component.params_path.name, "experiment.toml")
        self.assertEqual(component.playbooks[0].params["model_name"], "experiment")
        component.close()

    def test_multiple_files_prompt_for_selection(self) -> None:
        self.write_alternative()

        with patch("builtins.input", return_value="2") as input_mock:
            component = ZemiComponent()

        input_mock.assert_called_once()
        self.assertEqual(component.params_path.name, "experiment.toml")
        self.assertEqual(component.name, "logical")
        component.close()

    def test_multiple_files_require_explicit_selection_when_noninteractive(self) -> None:
        self.write_alternative()

        with (
            patch("builtins.input", side_effect=EOFError),
            self.assertRaisesRegex(RuntimeError, "Set params_file explicitly"),
        ):
            ZemiComponent()

    def test_papermill_receives_only_playbook_params_and_preserves_source(self) -> None:
        source = self.root / "one.ipynb"
        before = source.read_bytes()
        papermill = Mock()

        def execute(_source, output, **kwargs) -> None:
            kwargs["stdout_file"].write("cell stdout\n")
            kwargs["stderr_file"].write("cell stderr\n")
            self.write_executed_notebook(output)

        papermill.execute_notebook.side_effect = execute
        component = ZemiComponent()

        terminal_stdout = StringIO()
        terminal_stderr = StringIO()
        with (
            patch.dict(sys.modules, {"papermill": papermill}),
            redirect_stdout(terminal_stdout),
            redirect_stderr(terminal_stderr),
        ):
            component.playbooks[0].run()
        component.close()

        papermill.execute_notebook.assert_called_once_with(
            str(source),
            str(component.playbooks[0].output_path),
            parameters={"model_name": "default", "nested": {"left": 1}},
            cwd=str(self.root),
            progress_bar=True,
            log_output=False,
            stdout_file=terminal_stdout,
            stderr_file=terminal_stderr,
        )
        self.assertEqual(source.read_bytes(), before)
        report = json.loads(component.report.path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "succeeded")
        self.assertEqual(report["component_name"], self.root.name)
        self.assertEqual(report["params_file"], "params/default_params.toml")
        self.assertEqual(report["playbooks"][0]["status"], "succeeded")
        self.assertEqual(report["playbooks"][0]["timed_cells"], 1)
        self.assertGreaterEqual(report["playbooks"][0]["duration_seconds"], 0)
        terminal_text = terminal_stdout.getvalue()
        self.assertIn("ZEMI COMPONENT · PLAYBOOK START", terminal_text)
        self.assertIn("✓ PLAYBOOK COMPLETED · one.ipynb", terminal_text)
        self.assertIn("Parameters: params/default_params.toml", terminal_text)
        self.assertIn("Output    : .tmp/", terminal_text)
        self.assertEqual(terminal_text.count("cell stdout"), 1)
        self.assertEqual(terminal_stderr.getvalue().count("cell stderr"), 1)
        executed = json.loads(
            component.playbooks[0].output_path.read_text(encoding="utf-8")
        )
        timing = executed["cells"][1]
        self.assertEqual(timing["cell_type"], "markdown")
        self.assertEqual(timing["metadata"]["tags"], ["zemi-cell-timing"])
        self.assertEqual(timing["metadata"]["zemi"]["source_cell_id"], "executed-cell")
        self.assertIn("1.235 с", "".join(timing["source"]))

    def test_failed_playbook_is_saved_in_component_report(self) -> None:
        papermill = Mock()
        component = ZemiComponent()

        def fail_after_writing(_source, output, **kwargs) -> None:
            kwargs["stdout_file"].write("stdout before failure\n")
            kwargs["stderr_file"].write("stderr before failure\n")
            self.write_executed_notebook(output, duration=0.5)
            raise RuntimeError("execution failed")

        papermill.execute_notebook.side_effect = fail_after_writing

        terminal_stdout = StringIO()
        terminal_stderr = StringIO()
        with (
            patch.dict(sys.modules, {"papermill": papermill}),
            redirect_stdout(terminal_stdout),
            redirect_stderr(terminal_stderr),
        ):
            try:
                component.playbooks[0].run()
            except RuntimeError as error:
                component.report.record_failure(error)
            else:
                self.fail("Papermill failure was not propagated")
        component.close()

        report = json.loads(component.report.path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["message"], "execution failed")
        self.assertEqual(report["playbooks"][0]["status"], "failed")
        self.assertEqual(report["playbooks"][0]["timed_cells"], 1)
        terminal_text = terminal_stdout.getvalue()
        self.assertIn("✗ PLAYBOOK FAILED · one.ipynb", terminal_text)
        self.assertIn("Error   : RuntimeError: execution failed", terminal_text)
        self.assertEqual(terminal_text.count("stdout before failure"), 1)
        self.assertEqual(terminal_stderr.getvalue().count("stderr before failure"), 1)
        executed = json.loads(
            component.playbooks[0].output_path.read_text(encoding="utf-8")
        )
        self.assertIn("0.500 с", "".join(executed["cells"][1]["source"]))

    def test_timing_error_does_not_mask_papermill_error(self) -> None:
        papermill = Mock()
        papermill.execute_notebook.side_effect = RuntimeError("execution failed")
        component = ZemiComponent()
        playbook = component.playbooks[0]

        with (
            patch.dict(sys.modules, {"papermill": papermill}),
            patch.object(
                playbook,
                "_add_cell_timings",
                side_effect=ValueError("timing failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "execution failed"),
        ):
            playbook.run()

        entry = component.report.data["playbooks"][0]
        self.assertEqual(entry["error"]["message"], "execution failed")
        self.assertEqual(entry["timing_error"]["message"], "timing failed")


class ComponentPathTests(ComponentFixture):
    def test_runid_is_local_cached_and_collision_safe(self) -> None:
        fixed = datetime(2026, 8, 30, 12, 34, 56)
        base = self.root / ".tmp" / "run260830-123456"
        base.mkdir(parents=True)

        with patch("zemi.env.datetime") as datetime_mock:
            datetime_mock.now.return_value = fixed
            first = env.path.comp.runid
            second = env.path.comp.runid

        self.assertEqual(first, self.root / ".tmp" / "run260830-123456-01")
        self.assertIs(first, second)
        self.assertTrue(first.is_dir())


class JobLifecycleTests(unittest.TestCase):
    def test_successful_job_closes_component(self) -> None:
        component = Mock()

        with patch("zemi.component.ZemiComponent", return_value=component):
            runpy.run_path(str(JOB_PATH), run_name="__main__")

        component.run.assert_called_once_with()
        component.close.assert_called_once_with()

    def test_failure_is_reported_closed_and_propagated(self) -> None:
        failure = RuntimeError("notebook failed")
        component = Mock()
        component.run.side_effect = failure

        with (
            patch("zemi.component.ZemiComponent", return_value=component),
            self.assertRaisesRegex(RuntimeError, "notebook failed"),
        ):
            runpy.run_path(str(JOB_PATH), run_name="__main__")

        component.close.assert_called_once_with()

    def test_job_failure_produces_nonzero_process_exit(self) -> None:
        env.path.tmp.mkdir(parents=True, exist_ok=True)
        root = env.path.tmp / f"zemi-job-test-{os.getpid()}-{id(self)}"
        package = root / "zemi"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "component.py").write_text(
            """
class Report:
    def record_failure(self, error):
        open("failure.txt", "w", encoding="utf-8").write(str(error))

class Playbook:
    enabled = True
    def run(self):
        raise RuntimeError("subprocess failure")

class ZemiComponent:
    def __init__(self, **kwargs):
        self.playbooks = [Playbook()]
        self.report = Report()
    def run(self):
        try:
            self.playbooks[0].run()
        except Exception as error:
            self.report.record_failure(error)
            raise
    def close(self):
        open("closed.txt", "w", encoding="utf-8").write("closed")
""".lstrip(),
            encoding="utf-8",
        )
        shutil.copyfile(JOB_PATH, root / "job.exp.py")
        try:
            result = subprocess.run(
                [sys.executable, str(root / "job.exp.py")],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((root / "failure.txt").read_text(), "subprocess failure")
            self.assertEqual((root / "closed.txt").read_text(), "closed")
        finally:
            shutil.rmtree(root, ignore_errors=True)


class ComponentConventionTests(unittest.TestCase):
    def test_ignore_rules_track_parameter_files(self) -> None:
        rules = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn(".tmp/", rules)

    def test_example_notebook_has_exact_parameters_tag(self) -> None:
        notebook = json.loads((PROJECT_ROOT / "playbook.ipynb").read_text(encoding="utf-8"))
        cells = [cell for cell in notebook["cells"] if cell.get("id") == "parameters"]
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]["metadata"].get("tags"), ["parameters"])
        self.assertIn('model_name = "lfm2_350m"', "".join(cells[0]["source"]))
        tagged = [
            cell for cell in notebook["cells"]
            if "parameters" in cell.get("metadata", {}).get("tags", [])
        ]
        self.assertEqual(tagged, cells)

    def test_template_style_output_parameters_example_has_no_required_tag(self) -> None:
        notebook = json.loads((PROJECT_ROOT / "playbook.ipynb").read_text(encoding="utf-8"))
        headings = [cell for cell in notebook["cells"] if cell.get("id") == "output-parameters-heading"]
        outputs = [cell for cell in notebook["cells"] if cell.get("id") == "output-parameters"]
        self.assertEqual(len(headings), 1)
        self.assertIn("Output parameters", "".join(headings[0]["source"]))
        self.assertEqual(len(outputs), 1)
        source = "".join(outputs[0]["source"])
        self.assertIn("from zemi.playbook import output_params", source)
        self.assertIn("output_params({", source)
        self.assertNotIn("tags", outputs[0]["metadata"])

    def test_default_params_toml_parses_and_commented_sweep_is_inert(self) -> None:
        path = PROJECT_ROOT / "params" / "default_params.toml"
        text = path.read_text(encoding="utf-8")
        with path.open("rb") as file:
            params = tomllib.load(file)
        self.assertEqual(len(params["playbooks_params"]), 1)
        self.assertFalse(any(
            isinstance(value, dict) and "each" in value
            for value in params["playbooks_params"][0]["playbook_params"].values()
        ))
        self.assertIn("#     temperature = { each =", text)
        self.assertIn("#     seed = { each =", text)
        self.assertIn("#     backend = { select =", text)
        self.assertIn('#     __include__ = { ref = "param_buckets.combined" }', text)
        self.assertIn('#     copied_options = { ref = "param_buckets.model.options" }', text)
        self.assertIn("#     stop_sequences =", text)
        self.assertIn("# [[playbooks_params]]", text)


if __name__ == "__main__":
    unittest.main()

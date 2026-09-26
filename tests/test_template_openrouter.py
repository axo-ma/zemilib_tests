from __future__ import annotations

import json
from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2] / "zemi_component_template"


class OpenRouterFreePlaybookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(
            (ROOT / "example/optional_openrouter/playbook_openrouter_free.ipynb").read_text(encoding="utf-8")
        )
        cls.cells = cls.notebook["cells"]

    def test_canonical_heading_and_exact_parameters_tag(self) -> None:
        tagged = [
            cell for cell in self.cells
            if cell.get("metadata", {}).get("tags") == ["parameters"]
        ]
        self.assertEqual(len(tagged), 1)
        parameters = tagged[0]
        index = self.cells.index(parameters)
        self.assertEqual("".join(self.cells[index - 1]["source"]), "## Input parameters")
        self.assertEqual("".join(self.cells[index + 1]["source"]), "## Playbook preparation")
        source = "".join(parameters["source"])
        self.assertIn("arsenal_start_and_stop_at_job_level = False", source)
        self.assertNotIn("arsenal_stop_before_playbook_begin", source)
        self.assertNotIn("arsenal_stop_after_playbook_end", source)

    def test_begin_work_and_end_are_separate_ordered_cells(self) -> None:
        ids = [cell.get("id") for cell in self.cells]
        begin, select, client = map(ids.index, ("arsenal-begin", "select-model", "get-client"))
        request, result, end = map(ids.index, ("chat-completion", "show-result", "arsenal-end"))
        self.assertLess(begin, select)
        self.assertLess(select, client)
        self.assertLess(client, request)
        self.assertLess(request, result)
        self.assertLess(result, end)
        self.assertEqual(ids[-1], "arsenal-end")

        begin_source = "".join(self.cells[begin]["source"])
        end_source = "".join(self.cells[end]["source"])
        self.assertIn("ArsenalSession(arsenal_config_path)", begin_source)
        self.assertIn("if not arsenal_start_and_stop_at_job_level:", begin_source)
        self.assertIn("zemi.arsenal.begin(arsenal, stop_before_begin=True)", begin_source)
        self.assertIn("if not arsenal_start_and_stop_at_job_level:", end_source)
        self.assertIn("zemi.arsenal.end(arsenal, stop_after_end=True)", end_source)
        self.assertNotIn("chat.completions", begin_source + end_source)

    def test_interactive_work_has_no_wrapper_and_uses_remote_model_id(self) -> None:
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in self.cells if cell.get("cell_type") == "code"
        )
        self.assertNotIn("try:", code)
        self.assertNotIn("finally:", code)
        self.assertNotIn("def run(", code)
        self.assertIn("model = arsenal.endpoints.openrouter.models.openrouter_free", code)
        self.assertIn("client = assistant.clients.openai.client", code)
        self.assertIn("model=model.model", code)
        self.assertNotIn("assistant.clients.model", code)

    def test_batch_config_uses_job_level_lifecycle(self) -> None:
        with (ROOT / "example/optional_openrouter/params.toml").open("rb") as file:
            params = tomllib.load(file)
        lifecycle = params["arsenals"][0]
        self.assertEqual(lifecycle["lifecycle"], "job")
        self.assertNotIn("arsenal_stop_before_playbook_begin", lifecycle)
        self.assertNotIn("arsenal_stop_after_playbook_end", lifecycle)
        self.assertFalse(params["modules"][0]["enabled"])

    def test_all_template_arsenal_notebooks_use_the_single_flag(self) -> None:
        for name in (
            "example/playbook.ipynb",
            "optimizer_example/playbook.ipynb",
            "example/optional_openrouter/playbook_openrouter_free.ipynb",
        ):
            with self.subTest(notebook=name):
                notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
                parameters = [
                    cell for cell in notebook["cells"]
                    if cell.get("metadata", {}).get("tags") == ["parameters"]
                ]
                self.assertEqual(len(parameters), 1)
                defaults = "".join(parameters[0].get("source", []))
                self.assertIn("arsenal_start_and_stop_at_job_level = False", defaults)
                code = "\n".join(
                    "".join(cell.get("source", []))
                    for cell in notebook["cells"] if cell.get("cell_type") == "code"
                )
                self.assertEqual(code.count("if not arsenal_start_and_stop_at_job_level:"), 2)
                self.assertRegex(code, r"zemi\.arsenal\.begin\(\w+, stop_before_begin=True\)")
                self.assertRegex(code, r"zemi\.arsenal\.end\(\w+, stop_after_end=True\)")
                self.assertNotIn("arsenal_stop_before_playbook_begin", defaults + code)
                self.assertNotIn("arsenal_stop_after_playbook_end", defaults + code)


if __name__ == "__main__":
    unittest.main()

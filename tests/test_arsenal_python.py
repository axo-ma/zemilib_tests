from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from zemi import env
from zemi.arsenal.python import PythonVenv


class PythonVenvTests(unittest.TestCase):
    def setUp(self) -> None:
        env.path.tmp.mkdir(parents=True, exist_ok=True)
        self.instance = env.path.tmp / f"arsenal-python-{uuid4().hex}"
        self.component = self.instance / "demo_component"
        self.component.mkdir(parents=True)
        (self.instance / ".zemiinst_exp").touch()
        (self.component / ".zemicomp").touch()
        self.addCleanup(shutil.rmtree, self.instance, True)

    def test_standard_venv_name_contains_zemi_and_winpython_versions(self) -> None:
        venv = PythonVenv.standard(self.component)

        self.assertEqual(
            venv.root,
            self.instance / "_venvs" / "z260814-WPy64-312101",
        )

    def test_component_venv_name_contains_component_and_all_versions(self) -> None:
        venv = PythonVenv.for_component(
            component_name="demo-component",
            version="260814",
            start=self.component,
        )

        self.assertEqual(
            venv.root,
            self.instance
            / "_venvs"
            / "demo-component-260814-z260814-WPy64-312101",
        )

    def test_component_name_and_version_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "component_name"):
            PythonVenv.for_component(
                component_name="Demo Component",
                version="260814",
                start=self.component,
            )

        with self.assertRaisesRegex(ValueError, "YYMMDD"):
            PythonVenv.for_component(
                component_name="demo-component",
                version="1",
                start=self.component,
            )

    def test_create_if_missing_uses_system_site_packages(self) -> None:
        venv = PythonVenv.standard(self.component)
        paths = venv._paths
        paths.base_python.parent.mkdir(parents=True)
        paths.base_python.touch()

        with (
            patch("zemi.arsenal.python.subprocess.run") as run_mock,
            patch("zemi.arsenal.python._verify") as verify_mock,
            redirect_stdout(StringIO()),
        ):
            venv.create_if_missing()

        command = run_mock.call_args.args[0]
        self.assertEqual(command[0], str(paths.base_python))
        self.assertIn("--system-site-packages", command)
        self.assertEqual(command[-1], str(venv.root))
        verify_mock.assert_called_once_with(paths)

    def test_create_if_missing_does_not_recreate_existing_venv(self) -> None:
        venv = PythonVenv.standard(self.component)
        paths = venv._paths
        paths.base_python.parent.mkdir(parents=True)
        paths.base_python.touch()
        venv.python.parent.mkdir(parents=True)
        venv.python.touch()

        with (
            patch("zemi.arsenal.python.subprocess.run") as run_mock,
            patch("zemi.arsenal.python._verify"),
            redirect_stdout(StringIO()),
        ):
            venv.create_if_missing()

        run_mock.assert_not_called()

    def test_install_packages_skips_empty_list_for_standard_venv(self) -> None:
        venv = PythonVenv.standard(self.component)

        with redirect_stdout(output := StringIO()):
            venv.install_packages()

        self.assertIn("этап пропущен", output.getvalue())

    def test_install_packages_rejects_changes_to_standard_venv(self) -> None:
        venv = PythonVenv.standard(self.component)

        with (
            self.assertRaisesRegex(RuntimeError, "стандартный общий venv"),
            redirect_stdout(StringIO()),
        ):
            venv.install_packages("requests>=2.32,<3")

    def test_install_packages_uses_component_venv_python(self) -> None:
        venv = PythonVenv.for_component(
            component_name="demo-component",
            version="260814",
            start=self.component,
        )

        with (
            patch("zemi.arsenal.python.subprocess.run") as run_mock,
            redirect_stdout(StringIO()),
        ):
            venv.install_packages("requests>=2.32,<3")

        run_mock.assert_called_once_with(
            [str(venv.python), "-m", "pip", "install", "requests>=2.32,<3"],
            cwd=self.component,
            check=True,
        )

    def test_run_script_resolves_component_marker_path(self) -> None:
        script = self.component / "install.py"
        script.write_text("print('install')\n", encoding="utf-8")
        venv = PythonVenv.for_component(
            component_name="demo-component",
            version="260814",
            start=self.component,
        )

        with (
            patch("zemi.arsenal.python.subprocess.run") as run_mock,
            redirect_stdout(StringIO()),
        ):
            venv.run_script("@comp/install.py", "--quiet")

        run_mock.assert_called_once_with(
            [str(venv.python), str(script), "--quiet"],
            cwd=self.component,
            check=True,
        )

    def test_set_as_vscode_interpreter_preserves_existing_settings(self) -> None:
        venv = PythonVenv.standard(self.component)
        settings_path = self.component / ".vscode" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(
            json.dumps({"editor.formatOnSave": True}),
            encoding="utf-8",
        )

        with redirect_stdout(StringIO()):
            venv.set_as_vscode_interpreter()

        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        self.assertTrue(settings["editor.formatOnSave"])
        self.assertEqual(
            settings["python.defaultInterpreterPath"],
            "${workspaceFolder}/../_venvs/"
            "z260814-WPy64-312101/Scripts/python.exe",
        )

    def test_each_public_operation_prints_its_action(self) -> None:
        venv = PythonVenv.standard(self.component)

        with (
            patch("zemi.arsenal.python._create_if_missing", return_value=False),
            patch("zemi.arsenal.python._install_zemi_packages"),
            patch("zemi.arsenal.python._verify"),
            patch("zemi.arsenal.python._set_as_vscode_interpreter"),
            redirect_stdout(output := StringIO()),
        ):
            venv.create_if_missing()
            venv.install_zemi_packages()
            venv.install_packages()
            venv.verify()
            venv.set_as_vscode_interpreter()

        text = output.getvalue()
        self.assertIn("[1] СОЗДАНИЕ VENV", text)
        self.assertIn("[2] ПАКЕТЫ ZEMI", text)
        self.assertIn("[3] ПАКЕТЫ КОМПОНЕНТА", text)
        self.assertIn("[4] ПРОВЕРКА VENV", text)
        self.assertIn("[5] ИНТЕРПРЕТАТОР VS CODE", text)


if __name__ == "__main__":
    unittest.main()

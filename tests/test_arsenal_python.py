from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from zemi import env
from zemi.arsenal import python as arsenal_python


class ArsenalPythonTests(unittest.TestCase):
    def setUp(self) -> None:
        env.path.tmp.mkdir(parents=True, exist_ok=True)
        self.instance = env.path.tmp / f"arsenal-python-{uuid4().hex}"
        self.component = self.instance / "demo_component"
        self.component.mkdir(parents=True)
        (self.instance / ".zemiinst_exp").touch()
        (self.component / ".zemicomp").touch()
        self.addCleanup(shutil.rmtree, self.instance, True)

    def test_environment_uses_instance_venvs_and_winpython(self) -> None:
        versions = arsenal_python.config()
        paths = arsenal_python.environment(self.component)

        self.assertEqual(paths.component_root, self.component)
        self.assertEqual(paths.instance_root, self.instance)
        self.assertEqual(
            paths.base_python,
            self.instance
            / "_pythons"
            / "WPy64-312101"
            / "python"
            / "python.exe",
        )
        self.assertEqual(
            paths.environment_root,
            self.instance
            / "_venvs"
            / f"{versions.zemi_venv_version}-{versions.winpython_version}",
        )

    def test_create_uses_system_site_packages(self) -> None:
        paths = arsenal_python.environment(self.component)
        paths.base_python.parent.mkdir(parents=True)
        paths.base_python.touch()

        with (
            patch("zemi.arsenal.python.subprocess.run") as run_mock,
            patch("zemi.arsenal.python.check") as check_mock,
        ):
            result = arsenal_python.create(paths)

        self.assertIs(result, paths)
        command = run_mock.call_args.args[0]
        self.assertEqual(command[0], str(paths.base_python))
        self.assertIn("--system-site-packages", command)
        self.assertEqual(command[-1], str(paths.environment_root))
        check_mock.assert_called_once_with(paths)

    def test_configure_vscode_preserves_existing_settings(self) -> None:
        versions = arsenal_python.config()
        paths = arsenal_python.environment(self.component)
        paths.settings_path.parent.mkdir(parents=True)
        paths.settings_path.write_text(
            json.dumps({"editor.formatOnSave": True}),
            encoding="utf-8",
        )

        arsenal_python.configure_vscode(paths)

        settings = json.loads(paths.settings_path.read_text(encoding="utf-8"))
        self.assertTrue(settings["editor.formatOnSave"])
        self.assertEqual(
            settings["python.defaultInterpreterPath"],
            "${workspaceFolder}/../_venvs/"
            f"{versions.zemi_venv_version}-{versions.winpython_version}/"
            "Scripts/python.exe",
        )

    def test_config_loads_both_versions(self) -> None:
        versions = arsenal_python.config()

        self.assertEqual(versions.winpython_version, "WPy64-312101")
        self.assertTrue(versions.zemi_venv_version)

    def test_check_rejects_environment_without_inheritance(self) -> None:
        paths = arsenal_python.environment(self.component)
        paths.python.parent.mkdir(parents=True)
        paths.python.touch()
        (paths.environment_root / "pyvenv.cfg").write_text(
            "include-system-site-packages = false\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(RuntimeError, "не наследует"):
            arsenal_python.check(paths)

    def test_create_does_not_recreate_existing_environment(self) -> None:
        paths = arsenal_python.environment(self.component)
        paths.base_python.parent.mkdir(parents=True)
        paths.base_python.touch()
        paths.python.parent.mkdir(parents=True)
        paths.python.touch()

        with (
            patch("zemi.arsenal.python.subprocess.run") as run_mock,
            patch("zemi.arsenal.python.check") as check_mock,
        ):
            arsenal_python.create(paths)

        run_mock.assert_not_called()
        check_mock.assert_called_once_with(paths)

    def test_setup_prints_three_visible_steps(self) -> None:
        paths = arsenal_python.environment(self.component)
        paths.python.parent.mkdir(parents=True)
        paths.python.touch()

        with (
            patch("zemi.arsenal.python.environment", return_value=paths),
            patch("zemi.arsenal.python.create"),
            patch("zemi.arsenal.python.install"),
            patch("zemi.arsenal.python.configure_vscode"),
            redirect_stdout(output := StringIO()),
        ):
            result = arsenal_python.setup(self.component)

        text = output.getvalue()
        self.assertIs(result, paths)
        self.assertIn("[1/3] PYTHON-СРЕДА", text)
        self.assertIn("Среда уже существует", text)
        self.assertIn(paths.environment_root.name, text)
        self.assertIn("прозрачное наследование пакетов WinPython", text)
        self.assertIn("[2/3] БИБЛИОТЕКИ", text)
        self.assertIn("подробный вывод pip", text)
        self.assertIn("'Attempting uninstall' можно игнорировать", text)
        self.assertIn("[3/3] ИНТЕРПРЕТАТОР ПРОЕКТА", text)
        self.assertIn("python.defaultInterpreterPath", text)
        self.assertIn("ZEMI ARSENAL PYTHON ГОТОВ", text)


if __name__ == "__main__":
    unittest.main()

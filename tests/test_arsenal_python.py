from __future__ import annotations

import os, shutil, subprocess, unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from zemi import env
from zemi.arsenal.python import PythonVenv
from zemi.arsenal.python import IMPORTS, PACKAGES


class PythonVenvTests(unittest.TestCase):
    def test_removed_integrations_are_not_installed_or_verified(self):
        joined = " ".join((*PACKAGES, *IMPORTS)).lower()
        self.assertNotIn("baml", joined)
        self.assertNotIn("llama-cpp-agent", joined)
        self.assertNotIn("llama_cpp_agent", joined)

    def setUp(self):
        env.path.tmp.mkdir(parents=True, exist_ok=True)
        self.instance = env.path.tmp / f"arsenal-python-{uuid4().hex}"
        self.component = self.instance / "demo"; self.component.mkdir(parents=True)
        (self.instance / ".zemiinst_exp").touch(); (self.component / ".zemicomp").touch()
        self.config = self.component / "00_init.toml"; self.config.write_text("# disabled\n", encoding="utf-8")
        self.addCleanup(shutil.rmtree, self.instance, True)

    def from_config(self, text=None):
        if text is not None: self.config.write_text(text, encoding="utf-8")
        old = Path.cwd(); os.chdir(self.component)
        try: return PythonVenv.from_config("@comp/00_init.toml")
        finally: os.chdir(old)

    def prepare_base(self, venv, prompt=None):
        venv._paths.base_python.parent.mkdir(parents=True); venv._paths.base_python.touch()
        venv.python.parent.mkdir(parents=True); venv.python.touch()
        (venv.root / "pyvenv.cfg").write_text(
            f"include-system-site-packages = true\nprompt = {prompt or venv.prompt}\n", encoding="utf-8"
        )

    def test_commented_config_selects_standard_venv_and_prompt(self):
        venv = self.from_config()
        self.assertEqual(venv.root.name, "z260814-WPy64-312101")
        self.assertEqual(venv.prompt, "z260814")
        self.assertFalse(venv._c.active)

    def test_active_component_selects_long_name_and_prompt(self):
        venv = self.from_config('REQUIRED_C_BUNDLE_VERSION="mycomp260816"\nC_BUNDLE_PACKAGES=["requests==2.32.4"]\n')
        self.assertEqual(venv.root.name, "mycomp260816-z260814-WPy64-312101")
        self.assertEqual(venv.prompt, "mycomp260816")

    def test_component_version_and_packages_are_validated(self):
        bad = ('REQUIRED_C_BUNDLE_VERSION="mycomp260231"\nC_BUNDLE_PACKAGES=[]\n', "RunID")
        with self.assertRaisesRegex(ValueError, bad[1]): self.from_config(bad[0])
        with self.assertRaisesRegex(ValueError, "C_BUNDLE_PACKAGES"):
            self.from_config('REQUIRED_C_BUNDLE_VERSION="mycomp260816"\nC_BUNDLE_PACKAGES="requests"\n')
        with self.assertRaisesRegex(ValueError, "C_BUNDLE_PACKAGES"):
            self.from_config('REQUIRED_C_BUNDLE_VERSION="mycomp260816"\n')

    def test_create_uses_prompt_and_only_base_verification(self):
        venv = self.from_config(); venv._paths.base_python.parent.mkdir(parents=True); venv._paths.base_python.touch()
        with patch.object(venv, "_verify_base"), patch("zemi.arsenal.python.subprocess.run") as run:
            venv.create_if_missing()
        command = run.call_args.args[0]
        self.assertIn("--system-site-packages", command); self.assertEqual(command[command.index("--prompt") + 1], "z260814")

    def test_component_install_and_disabled_skip(self):
        standard = self.from_config()
        with patch("zemi.arsenal.python.subprocess.run") as run: standard.install_component_packages()
        run.assert_not_called()
        component = self.from_config('REQUIRED_C_BUNDLE_VERSION="mycomp260816"\nC_BUNDLE_PACKAGES=["requests==2.32.4"]\n')
        with patch("zemi.arsenal.python.subprocess.run") as run: component.install_component_packages()
        self.assertEqual(run.call_args.args[0][-1], "requests==2.32.4")

    def test_failed_component_install_does_not_allow_stamp(self):
        venv = self.from_config('REQUIRED_C_BUNDLE_VERSION="mycomp260816"\nC_BUNDLE_PACKAGES=["bad"]\n')
        venv._z_done = True
        with patch("zemi.arsenal.python.subprocess.run", side_effect=subprocess.CalledProcessError(1, "pip")):
            with self.assertRaises(subprocess.CalledProcessError): venv.install_component_packages()
        with self.assertRaisesRegex(RuntimeError, "incomplete installation"): venv.finalize_install()
        self.assertFalse((venv.root / "00_init.toml").exists())

    def test_finalize_and_verify_stamps_compare_toml_values(self):
        venv = self.from_config('REQUIRED_C_BUNDLE_VERSION="mycomp260816"\nC_BUNDLE_PACKAGES=["a==1"]\n')
        venv.root.mkdir(parents=True); venv._z_done = venv._c_done = True; venv.finalize_install()
        (venv.root / "zemi_python_venv.toml").write_text('REQUIRED_Z_BUNDLE_VERSION="z260814"\nREQUIRED_WINPYTHON_VERSION="WPy64-312101"\n', encoding="utf-8")
        self.prepare_base(venv)
        result = type("Result", (), {"stdout": str(venv._paths.base_python.parent) + "\n"})()
        with patch("zemi.arsenal.python.subprocess.run", return_value=result), patch("zemi.arsenal.python.sys.executable", str(venv.python)):
            venv.verify()
        before = (venv.root / "00_init.toml").read_bytes()
        with patch("zemi.arsenal.python.subprocess.run", return_value=result), patch("zemi.arsenal.python.sys.executable", str(venv.python)):
            venv.verify()
        self.assertEqual(before, (venv.root / "00_init.toml").read_bytes())

    def test_missing_environment_message_includes_path_and_setup_steps(self):
        venv = self.from_config()
        expected = (
            "The required Python environment does not exist:\n"
            f"  {venv.python}\n\n"
            "Run 00_init.py from the component root to create and configure this environment.\n"
            "After initialization completes, select the environment above as the notebook kernel or use it to run the Python script, then try again."
        )
        with self.assertRaises(FileNotFoundError) as raised:
            venv.verify()
        self.assertEqual(str(raised.exception), expected)

    def test_wrong_environment_message_includes_current_and_required_python(self):
        venv = self.from_config()
        self.prepare_base(venv)
        current_python = self.component / "other" / "python.exe"
        result = type("Result", (), {"stdout": str(venv._paths.base_python.parent) + "\n"})()
        expected = (
            "The current Python process is using the wrong environment.\n\n"
            "Current Python:\n"
            f"  {current_python.resolve()}\n\n"
            "Required Python:\n"
            f"  {venv.python.resolve()}\n\n"
            "For a notebook, select the required Python environment as the kernel and run this cell again.\n"
            "For a Python script, run the script with the required Python interpreter.\n"
            "If the required environment is unavailable, run 00_init.py from the component root first."
        )
        with patch("zemi.arsenal.python.subprocess.run", return_value=result), patch("zemi.arsenal.python.sys.executable", str(current_python)):
            with self.assertRaises(RuntimeError) as raised:
                venv.verify()
        self.assertEqual(str(raised.exception), expected)

    def test_changed_packages_make_c_stamp_stale(self):
        venv = self.from_config('REQUIRED_C_BUNDLE_VERSION="mycomp260816"\nC_BUNDLE_PACKAGES=["a==1"]\n')
        venv.root.mkdir(parents=True); shutil.copy2(self.config, venv.root / "00_init.toml")
        self.config.write_text('REQUIRED_C_BUNDLE_VERSION="mycomp260816"\nC_BUNDLE_PACKAGES=["a==2"]\n', encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "C-bundle is outdated"): venv._verify_c_stamp()

    def test_run_script_failure_propagates(self):
        venv = self.from_config('REQUIRED_C_BUNDLE_VERSION="mycomp260816"\nC_BUNDLE_PACKAGES=[]\n')
        (self.component / "install.py").touch()
        with patch("zemi.arsenal.python.subprocess.run", side_effect=subprocess.CalledProcessError(1, "script")):
            with self.assertRaises(subprocess.CalledProcessError): venv.run_script("@comp/install.py")


if __name__ == "__main__": unittest.main()

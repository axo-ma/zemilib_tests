"""Windows ZEMI commands can print Unicode without launcher flags."""

import os
import subprocess
import sys
import unittest


class UnicodeConsoleTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "Windows console encoding")
    def test_import_enables_utf8_on_redirected_console(self):
        environment = os.environ.copy()
        environment.pop("PYTHONUTF8", None)
        environment.pop("PYTHONIOENCODING", None)
        result = subprocess.run(
            [sys.executable, "-X", "utf8=0", "-c", "import zemi; print('Кириллица')"],
            capture_output=True, env=environment, check=True,
        )
        self.assertEqual(result.stdout.decode("utf-8").strip(), "Кириллица")

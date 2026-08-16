import sys

from zemi.arsenal.python import PythonVenv

PythonVenv.from_config("@comp/00_init.toml").verify()

print("Hello from zc_hello!")
print(f"Python: {sys.executable}")
print(f"Base Python: {sys.base_prefix}")

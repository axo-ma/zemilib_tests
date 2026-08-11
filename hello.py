import sys
from pathlib import Path


assert Path(sys.prefix).name == ".venv", "Use @comp/.venv/Scripts/python.exe"

print("Hello from zc_hello!")
print(f"Python: {sys.executable}")
print(f"Base Python: {sys.base_prefix}")

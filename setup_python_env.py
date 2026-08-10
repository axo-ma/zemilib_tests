from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys
import sysconfig


COMPONENT_ROOT = Path(__file__).resolve().parent
VENV_ROOT = COMPONENT_ROOT / ".venv"
EXPECTED_WINPYTHON = (
    COMPONENT_ROOT.parent / "_pythons" / "WPy64-312101" / "python"
).resolve()

PACKAGES = [
    "openai>=1.30.0",
    "httpx>=0.27.0",
    "pydantic>=2.7.0",
    # Keep compatibility with FastAPI and MSAL inherited from WinPython.
    "starlette==0.46.2",
    "cryptography==44.0.0",
    "joserfc==1.1.0",
    "python-calamine>=0.2.0",
    "openpyxl>=3.1.0",
    "markitdown>=0.0.1a0",
    "pandas>=2.2.0",
    "duckdb>=1.0.0",
    "fastembed>=0.3.0",
    "streamlit>=1.35.0",
    "dspy-ai>=2.4.0",
    "instructor>=1.3.0",
    "pydantic-ai>=0.0.14",
    "baml-py>=0.70.0",
    "smolagents>=1.0.0",
    "litellm>=1.35.0",
    "outlines>=0.0.40",
    "guidance>=0.1.15",
    "llama-index-core>=0.10.0",
    "llama-index-llms-openai>=0.1.0",
    "llama-index-llms-openai-like",
    "unstructured-client>=0.25.0",
    "ipykernel",
]

IMPORTS = [
    "python_calamine",
    "openpyxl",
    "markitdown",
    "pandas",
    "duckdb",
    "fastembed",
    "streamlit",
    "dspy",
    "instructor",
    "pydantic_ai",
    "baml_py",
    "smolagents",
    "litellm",
    "outlines",
    "guidance",
    "llama_index.core",
    "llama_index.llms.openai_like",
    "unstructured_client",
    "llama_cpp_agent",
    "ipykernel",
]

LLAMA_CPP_STUB = '''\
"""ZEMI REST-mode compatibility stub for external llama-server.exe."""
from unittest.mock import MagicMock

def __getattr__(name: str):
    return MagicMock(name=name)

Llama = MagicMock
LlamaGrammar = MagicMock
'''


def run_pip(*arguments: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", *arguments],
        cwd=COMPONENT_ROOT,
        check=True,
    )


def validate_environment() -> None:
    if not (COMPONENT_ROOT / ".zemicomp").is_file():
        raise RuntimeError("The script is not in a ZEMI Component root.")

    instance_markers = [
        marker
        for marker in (".zemiinst_dev", ".zemiinst_exp", ".zemiinst_prod")
        if (COMPONENT_ROOT.parent / marker).is_file()
    ]
    if len(instance_markers) != 1:
        raise RuntimeError("Expected exactly one ZEMI Instance marker.")

    if Path(sys.prefix).resolve() != VENV_ROOT.resolve():
        raise RuntimeError(
            "Run this script with @comp/.venv/Scripts/python.exe."
        )
    if Path(sys.base_prefix).resolve() != EXPECTED_WINPYTHON:
        raise RuntimeError(
            "@comp/.venv is not based on @inst/_pythons/WPy64-312101/python."
        )

    pyvenv_config = (VENV_ROOT / "pyvenv.cfg").read_text(encoding="utf-8")
    normalized_config = pyvenv_config.lower().replace(" ", "")
    if "include-system-site-packages=true" not in normalized_config:
        raise RuntimeError("@comp/.venv does not inherit WinPython packages.")


def install_dependencies() -> None:
    run_pip("install", "--only-binary", ":all:", *PACKAGES)
    run_pip("install", "--no-deps", "llama-cpp-agent>=0.2.0")


def install_rest_stub() -> None:
    site_packages = Path(sysconfig.get_paths()["purelib"])
    (site_packages / "llama_cpp.py").write_text(
        LLAMA_CPP_STUB,
        encoding="utf-8",
    )


def check_imports() -> None:
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    for module_name in IMPORTS:
        importlib.import_module(module_name)
        print(f"[OK] {module_name}")


def main() -> None:
    validate_environment()
    install_dependencies()
    install_rest_stub()
    check_imports()
    print("@comp/.venv is ready. Base WinPython was not modified.")


if __name__ == "__main__":
    main()

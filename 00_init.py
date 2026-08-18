"""Initialize the Python venv for the current ZEMI Component."""

from zemi.arsenal.python import PythonVenv


def initialize_component_python_venv() -> None:
    # Enable and configure the C-bundle in @comp/00_init.toml.
    # No Python code changes are required to add packages.
    venv = PythonVenv.from_config("@comp/00_init.toml")

    venv.create_if_missing()
    venv.install_zemi_packages()

    venv.install_component_packages()

    # Additional installation code:
    # venv.run_script("@comp/install.py")

    venv.finalize_install()
    venv.verify()
    venv.set_as_vscode_interpreter()


if __name__ == "__main__":
    initialize_component_python_venv()


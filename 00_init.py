"""Инициализация Python venv для текущего ZEMI Component."""

from zemi.arsenal.python import PythonVenv


def initialize_component_python_venv() -> None:
    # C-bundle активируется и настраивается в @comp/00_init.toml.
    # Python-код для добавления пакетов изменять не требуется.
    venv = PythonVenv.from_config("@comp/00_init.toml")

    venv.create_if_missing()
    venv.install_zemi_packages()

    venv.install_component_packages()

    # Дополнительный установочный код:
    # venv.run_script("@comp/install.py")

    venv.finalize_install()
    venv.verify()
    venv.set_as_vscode_interpreter()


if __name__ == "__main__":
    initialize_component_python_venv()

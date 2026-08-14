"""Инициализация Python venv для текущего ZEMI Component."""

from zemi.arsenal.python import PythonVenv


def initialize_component_python_venv() -> None:
    # Стандартный общий venv ZEMI.
    # Используйте его, если компоненту не нужны собственные пакеты.
    venv = PythonVenv.standard()

    # Для собственных пакетов или установочного кода замените строку выше:
    #
    # venv = PythonVenv.for_component(
    #     component_name="my-component",
    #     version="260814",
    # )
    #
    # Меняйте version после изменения состава или настройки venv.

    venv.create_if_missing()
    venv.install_zemi_packages()

    # Пакеты компонента:
    venv.install_packages(
        # "requests>=2.32,<3",
    )

    # Дополнительный установочный код:
    # venv.run_script("@comp/install.py")

    venv.verify()
    venv.set_as_vscode_interpreter()


if __name__ == "__main__":
    initialize_component_python_venv()

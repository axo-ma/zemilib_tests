# General Instructions

Do not assume paths, dependencies, branch policies, or deployment behavior unless they are explicitly specified. If a required value is unknown, first find it in the available context or ask for clarification.

## Paths in TOML

Do not use absolute paths. Specify paths in only one of these two forms:

- `@inst/…` — relative to the ZEMI Instance root.
- `@comp/…` — relative to the current ZEMI Component root.

Examples: `@inst/pythons/WPy64-312101`, `@comp/context`.

Use `env.path` when working with paths.

## Temporary files and directories

For all temporary files and directories inside ZEMI, use only `env.path.tmp`
(that is, `@inst/_tmp`). Create this directory before use if it does not exist.
Do not use the default system temporary directory.

## Hugging Face models

A model identifier consists of the owner, Hugging Face repository, and exact GGUF file.

Example: `hf:bartowski/Qwen_Qwen3.5-4B-GGUF/Qwen_Qwen3.5-4B-Q4_K_M.gguf`.

The model directory name follows this rule:

`hf:{owner}/{repo}/{filename} → hf--{owner}--{repo}--{filename without .gguf}`

Example: `hf--bartowski--Qwen_Qwen3.5-4B-GGUF--Qwen_Qwen3.5-4B-Q4_K_M`.

Store the model in `@inst/_models/<model_directory_name>`, where the directory
name is derived from the identifier using the rule above.

For future models, use the exact lowercase identifier prefix `zemi:`, analogous
to the existing `hf:` prefix.

## Llama CPP

Engine identifier: `llama:b1234`; corresponding directory: `llama--b1234`.

`b1234` is the llama.cpp build version.

Store the engine in `@inst/_llamas/<engine_directory_name>`, where the directory
name is derived from the identifier using the rule above.

## Pythons

Store WinPython environments in `@inst/_pythons/<Python_directory_name>`.

All WinPython directory names inside ZEMI must follow a single naming standard.
The current WinPython version is 3.12; its standard directory name is `WPy64-312101`.

### Project Python interpreter

For Python commands, use the interpreter configured in the current project's
VS Code `python.defaultInterpreterPath` setting. Run tests, scripts, import
checks, and dependency management through that interpreter.

## ZEMI library development and tests

- Develop the `zemi` library in `zemilib_tests/zemi` and add its tests in
  `zemilib_tests/tests`.
- Do not add `zemi` library tests to `zemi_component_template`. That repository
  is a consumer-facing component template, not the library development or test
  location.

## Standard markers

- **ZEMI Instance** — the root directory of an installed ZEMI platform instance.
  It contains resources shared by components and service directories such as
  `_models`, `_llamas`, `_tmp`, and `_pythons`.
- **ZEMI Component** — a standalone project or functional component that runs
  inside a ZEMI Instance. One ZEMI Instance can contain multiple ZEMI Components.
- Markers are regular files placed directly in the roots of the corresponding
  directories. They are not directories or environment names.
- A ZEMI Instance root contains exactly one marker file depending on the
  environment: `.zemiinst_dev`, `.zemiinst_exp`, or `.zemiinst_prod`.
- A ZEMI Component root contains the `.zemicomp` marker file.
- A directory that must be a separate root in a VS Code multi-root workspace
  contains the `.zemiworkroot` marker file. A ZEMI Component does not need a
  separate marker because `.zemicomp` is sufficient.

Before starting work, verify that the directories exist and that the markers
are located directly in the expected roots.

## Git rules

- Use standard Git only, without GitHub CLI.
- Use the `main` branch.

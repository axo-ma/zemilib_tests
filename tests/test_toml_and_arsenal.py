from __future__ import annotations

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from zemi import arsenal, env, toml
from zemi.arsenal import ArsenalSession, Assistant, Llama, Model
from zemi.arsenal.libs import Libs


CONFIG_PATH = (
    env.path.comp.root
    / "tests"
    / "playbook_arsenal"
    / "test_playbook_arsenal_router_mode.toml"
)


class TomlTests(unittest.TestCase):
    def test_load_preserves_plain_dicts_lists_and_references(self) -> None:
        config = toml.load(CONFIG_PATH)

        self.assertIs(type(config), dict)
        self.assertIs(type(config["arsenal"]), dict)
        self.assertIs(type(config["arsenal"]["llamas"]), list)
        self.assertEqual(
            config["text_reference"],
            "@comp/tests/zemi_toml/prefixes/plain-text.txt",
        )
        self.assertEqual(
            config["arsenal"]["llamas"][0]["models"][0]["assistants"][0][
                "prefix"
            ],
            "@comp/tests/zemi_toml/prefixes/qwen-system.md",
        )

    def test_load_rejects_duplicate_names(self) -> None:
        path = self._write_temp_toml(
            "[[items]]\nname = 'same'\n[[items]]\nname = 'same'\n"
        )

        with self.assertRaisesRegex(ValueError, "duplicate name 'same'"):
            toml.load(path)

    def test_load_rejects_missing_zemi_reference(self) -> None:
        path = self._write_temp_toml(
            "reference = '@comp/does-not-exist-for-toml-test.txt'\n"
        )

        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            toml.load(path)

    def _write_temp_toml(self, content: str) -> Path:
        env.path.tmp.mkdir(parents=True, exist_ok=True)
        path = env.path.tmp / f"zemi-toml-test-{id(self)}.toml"
        path.write_text(content, encoding="utf-8")
        self.addCleanup(path.unlink, missing_ok=True)
        return path


class ArsenalObjectTreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = toml.load(CONFIG_PATH)
        self.arsenal = ArsenalSession(self.config)

    def test_builds_named_runtime_tree(self) -> None:
        primary = self.arsenal.llamas["primary"]
        qwen = primary.models["qwen"]
        assistant = qwen.assistants["assistant"]

        self.assertIsInstance(primary, Llama)
        self.assertIsInstance(qwen, Model)
        self.assertIsInstance(assistant, Assistant)
        self.assertIs(primary, self.arsenal.llamas[0])
        self.assertIs(qwen, primary.models[0])
        self.assertIs(assistant, qwen.assistants[0])
        self.assertEqual(primary.host, "127.0.0.1")
        self.assertEqual(qwen.alias, "qwen3.5-4b")
        self.assertEqual(
            assistant.prefix,
            "@comp/tests/zemi_toml/prefixes/qwen-system.md",
        )
        self.assertIsInstance(assistant.clients, Libs)
        self.assertFalse(hasattr(assistant, "libs"))
        self.assertEqual(assistant.clients.server_url, "http://127.0.0.1:8080")
        self.assertEqual(assistant.clients.openai_url, "http://127.0.0.1:8080/v1")
        self.assertEqual(assistant.clients.model, "qwen3.5-4b")
        self.assertEqual(assistant.clients.context_window, 8192)

    def test_default_config_uses_curated_router_set(self) -> None:
        default = ArsenalSession()

        self.assertEqual(
            default.config_path,
            "zemi/llm_curated_set_router_mode.toml",
        )
        ling = default.llamas.curated_router.models.ling30_tiny
        self.assertEqual(ling.alias, "ling-3.0-tiny")

    def test_each_assistant_has_own_libs_object(self) -> None:
        qwen = self.arsenal.llamas.primary.models.qwen

        assistant = qwen.assistants.assistant
        json_converter = qwen.assistants.json_converter

        self.assertIsNot(assistant.clients, json_converter.clients)
        self.assertEqual(assistant.clients.server_url, json_converter.clients.server_url)
        self.assertEqual(assistant.clients.model, json_converter.clients.model)
        self.assertEqual(
            assistant.clients.context_window,
            json_converter.clients.context_window,
        )

    def test_runtime_tree_wraps_but_does_not_replace_raw_config(self) -> None:
        primary = self.arsenal.llamas.primary

        self.assertIs(type(self.arsenal.config), dict)
        self.assertIs(type(self.arsenal.config["arsenal"]["llamas"]), list)
        self.assertIs(primary.config, self.config["arsenal"]["llamas"][0])

    def test_runtime_types_have_separate_modules(self) -> None:
        self.assertEqual(Assistant.__module__, "zemi.arsenal.objects")
        self.assertEqual(Model.__module__, "zemi.arsenal.objects")
        self.assertEqual(Llama.__module__, "zemi.arsenal.objects")
        self.assertEqual(ArsenalSession.__module__, "zemi.arsenal.runtime")

    def test_download_exists_only_on_arsenal_session(self) -> None:
        self.assertFalse(hasattr(arsenal, "download"))
        self.assertTrue(callable(ArsenalSession.download))

    def test_model_lookup_activates_only_the_selected_model(self) -> None:
        selected = self.arsenal.model("qwen")

        self.assertEqual(selected.name, "qwen")

    def test_model_lookup_reports_zero_and_ambiguous_matches(self) -> None:
        with self.assertRaisesRegex(LookupError, "No Arsenal model"):
            self.arsenal.model("missing")

        duplicate_config = json.loads(json.dumps(self.config))
        duplicate_config["arsenal"]["llamas"].append(
            json.loads(json.dumps(duplicate_config["arsenal"]["llamas"][0]))
        )
        duplicate_config["arsenal"]["llamas"][1]["name"] = "secondary"
        duplicate = ArsenalSession(duplicate_config)
        with self.assertRaisesRegex(LookupError, "ambiguous across servers"):
            duplicate.model("qwen")

    def test_arsenal_exposes_lifecycle(self) -> None:
        self.assertTrue(callable(arsenal.begin))
        self.assertTrue(callable(arsenal.end))

    def test_arsenal_end_delegates_to_session(self) -> None:
        with patch.object(self.arsenal, "_end") as end_mock:
            arsenal.end(self.arsenal, stop_after_end=True)

        end_mock.assert_called_once_with(stop_arsenal_after_end=True)


class ArsenalLazyActivationTests(unittest.TestCase):
    MODEL_MODE_PATH = (
        "@comp/tests/playbook_arsenal/"
        "test_playbook_arsenal_model_mode.toml"
    )
    ROUTER_MODE_PATH = (
        "@comp/tests/playbook_arsenal/"
        "test_playbook_arsenal_router_mode.toml"
    )

    @patch("zemi.arsenal.runtime.download_model")
    @patch("zemi.arsenal.runtime.download_llama")
    def test_constructor_and_begin_do_not_download_or_start(
        self,
        download_llama_mock,
        download_model_mock,
    ) -> None:
        with redirect_stdout(StringIO()):
            with patch.object(ArsenalSession, "_stop_arsenal") as stop_mock:
                result = arsenal.begin(
                    self.MODEL_MODE_PATH,
                    stop_before_begin=True,
                    llama_router_mode=False,
                )

        stop_mock.assert_called_once_with()
        self.assertEqual(result.config_path, self.MODEL_MODE_PATH)
        self.assertEqual(len(result.llamas), 2)
        primary = result.llamas._by_name["primary"]
        self.assertEqual(primary.models._by_name["qwen"].name, "qwen")
        download_llama_mock.assert_not_called()
        download_model_mock.assert_not_called()
        self.assertEqual(result._processes, {})

    @patch("zemi.arsenal.runtime.download_model")
    @patch("zemi.arsenal.runtime.download_llama")
    def test_download_eagerly_prepares_every_resource_without_starting(
        self,
        download_llama_mock,
        download_model_mock,
    ) -> None:
        download_llama_mock.side_effect = [
            Path("llama-primary"),
            Path("llama-secondary"),
        ]
        download_model_mock.side_effect = [
            Path("qwen.gguf"),
            Path("phi.gguf"),
        ]
        result = ArsenalSession(self.MODEL_MODE_PATH)

        with (
            patch.object(result, "_start_server") as start_mock,
            redirect_stdout(StringIO()),
        ):
            result.download()
            result.download()

        self.assertEqual(download_llama_mock.call_count, 2)
        self.assertEqual(download_model_mock.call_count, 2)
        self.assertEqual(len(result._llama_paths), 2)
        self.assertEqual(len(result._model_paths), 2)
        self.assertFalse(result._active)
        start_mock.assert_not_called()

    @patch("zemi.arsenal.runtime.download_model")
    @patch("zemi.arsenal.runtime.download_llama")
    def test_model_access_downloads_and_starts_only_once(
        self,
        download_llama_mock,
        download_model_mock,
    ) -> None:
        download_llama_mock.return_value = Path("llama")
        download_model_mock.return_value = Path("model.gguf")
        with (
            patch.object(ArsenalSession, "_server_path", return_value=Path("server.exe")),
            patch.object(ArsenalSession, "_model_path", return_value=Path("model.gguf")),
            patch.object(ArsenalSession, "_start_server") as start_mock,
            patch.object(ArsenalSession, "_is_server_ready", return_value=True),
            redirect_stdout(StringIO()),
        ):
            result = arsenal.begin(
                self.MODEL_MODE_PATH,
                stop_before_begin=False,
                llama_router_mode=False,
            )
            first = result.llamas["primary"].models["qwen"]
            second = result.llamas.primary.models.qwen

        self.assertIs(first, second)
        download_llama_mock.assert_called_once_with("llama:b9222")
        download_model_mock.assert_called_once_with(
            "bartowski",
            "Qwen_Qwen3.5-4B-GGUF",
            "Qwen_Qwen3.5-4B-Q4_K_M.gguf",
            source="hf",
        )
        start_mock.assert_called_once()

    @patch("zemi.arsenal.runtime.download_model")
    @patch("zemi.arsenal.runtime.download_llama")
    def test_router_adds_models_without_restarting_parent_server(
        self,
        download_llama_mock,
        download_model_mock,
    ) -> None:
        download_llama_mock.return_value = Path("llama")
        download_model_mock.side_effect = [Path("qwen.gguf"), Path("smollm.gguf")]
        result = ArsenalSession(self.ROUTER_MODE_PATH)

        process = Mock()
        process.poll.return_value = None

        def remember_process(llama, _command) -> None:
            result._processes[llama.name] = process

        with (
            patch.object(result, "_server_path", return_value=Path("server.exe")),
            patch.object(
                result,
                "_write_router_preset",
                return_value=Path("models.ini"),
            ) as preset_mock,
            patch.object(
                result,
                "_start_server",
                side_effect=remember_process,
            ) as start_mock,
            patch.object(result, "_load_router_model") as load_mock,
            patch.object(result, "_stop_llama") as stop_mock,
            redirect_stdout(StringIO()),
        ):
            result._begin(
                stop_arsenal_before_begin=False,
                llama_router_mode=True,
            )
            qwen = result.llamas.primary.models.qwen
            smollm = result.llamas.primary.models.smollm

        self.assertEqual((qwen.name, smollm.name), ("qwen", "smollm"))
        download_llama_mock.assert_called_once_with("llama:b9222")
        self.assertEqual(download_model_mock.call_count, 2)
        preset_mock.assert_called_once()
        start_mock.assert_called_once()
        self.assertEqual(load_mock.call_count, 2)
        stop_mock.assert_not_called()

    def test_router_load_endpoint_uses_model_alias(self) -> None:
        result = ArsenalSession(self.ROUTER_MODE_PATH)
        llama = result.llamas.primary
        model = llama.models.qwen
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b'{"success": true}'

        with (
            patch(
                "zemi.arsenal.runtime.urlopen",
                return_value=response,
            ) as urlopen_mock,
            redirect_stdout(StringIO()),
        ):
            result._load_router_model(llama, model)

        request = urlopen_mock.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "http://127.0.0.1:8080/models/load",
        )
        self.assertEqual(json.loads(request.data), {"model": "qwen3.5-4b"})


class LlmCuratedSetTests(unittest.TestCase):
    ROUTER_PATH = "@comp/zemi/llm_curated_set_router_mode.toml"
    MODEL_PATH = "@comp/zemi/llm_curated_set_model_mode.toml"
    EXISTING_MODELS = {
        "qwen35_4b": (
            "Qwen_Qwen3.5-4B-GGUF",
            "Qwen_Qwen3.5-4B-Q4_K_M.gguf",
        ),
        "phi35_mini": (
            "Phi-3.5-mini-instruct-GGUF",
            "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        ),
        "llama32_3b": (
            "Llama-3.2-3B-Instruct-GGUF",
            "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        ),
        "smollm2_1_7b": (
            "SmolLM2-1.7B-Instruct-GGUF",
            "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        ),
        "ling30_tiny": (
            "Ling-3.0-tiny-GGUF",
            "Ling-3.0-tiny-Q4_K_M.gguf",
        ),
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.router = ArsenalSession(cls.ROUTER_PATH)
        cls.model_mode = ArsenalSession(cls.MODEL_PATH)

    @staticmethod
    def _models(session: ArsenalSession) -> list[Model]:
        return [model for llama in session.llamas for model in llama.models]

    def test_files_parse_and_have_required_server_layout(self) -> None:
        self.assertEqual(len(self.router.llamas), 1)
        self.assertEqual(len(self.router.llamas[0].models), 13)
        self.assertEqual(len(self.model_mode.llamas), 13)
        self.assertTrue(
            all(len(llama.models) == 1 for llama in self.model_mode.llamas)
        )

    def test_model_sets_and_identifiers_are_unique(self) -> None:
        router_models = self._models(self.router)
        model_mode_models = self._models(self.model_mode)
        router_aliases = {model.alias for model in router_models}
        model_mode_aliases = {model.alias for model in model_mode_models}

        self.assertEqual(router_aliases, model_mode_aliases)
        for models in (router_models, model_mode_models):
            self.assertEqual(len({model.name for model in models}), 13)
            self.assertEqual(len({model.alias for model in models}), 13)

        ports = [llama.port for llama in self.model_mode.llamas]
        self.assertEqual(len(set(ports)), 13)
        self.assertEqual(set(ports), set(range(8080, 8093)))

    def test_every_server_uses_curated_llama_build(self) -> None:
        for session in (self.router, self.model_mode):
            for llama in session.llamas:
                self.assertEqual(llama.llama_build, "llama:b10507")

    def test_every_model_uses_non_thinking_q4_hugging_face_artifact(self) -> None:
        for session in (self.router, self.model_mode):
            for model in self._models(session):
                artifact = f"{model.repository}/{model.filename}"
                self.assertEqual(model.reasoning, "off")
                self.assertEqual(model.source, "hf")
                self.assertTrue(model.filename.endswith("Q4_K_M.gguf"))
                self.assertNotIn("Base", artifact)
                self.assertNotIn("Thinking", artifact)
                self.assertEqual(len(model.assistants), 1)
                self.assertEqual(model.assistants[0].name, "assistant")

    def test_qwen25_is_coder_instruct(self) -> None:
        model = self.router.llamas[0].models["qwen25_coder_3b"]
        self.assertIn("Qwen2.5-Coder-3B-Instruct", model.repository)
        self.assertIn("Qwen2.5-Coder-3B-Instruct", model.filename)

    def test_ling30_tiny_parameters(self) -> None:
        for session in (self.router, self.model_mode):
            model = {model.name: model for model in self._models(session)}[
                "ling30_tiny"
            ]
            self.assertEqual(model.owner, "bloomer010")
            self.assertEqual(model.alias, "ling-3.0-tiny")
            self.assertEqual(model.ctx_size, 4096)
            self.assertEqual(model.threads, 4)
            self.assertEqual(model.threads_batch, 4)
            self.assertEqual(model.reasoning, "off")

    def test_existing_models_keep_verified_artifacts(self) -> None:
        for session in (self.router, self.model_mode):
            models = {model.name: model for model in self._models(session)}
            for name, expected in self.EXISTING_MODELS.items():
                self.assertEqual(
                    (models[name].repository, models[name].filename),
                    expected,
                )

if __name__ == "__main__":
    unittest.main()

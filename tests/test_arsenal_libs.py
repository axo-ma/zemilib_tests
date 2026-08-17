from __future__ import annotations

import socket
import unittest
from unittest.mock import patch

from zemi.arsenal.libs import LibDependencyError, Libs


class ArsenalLibsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.libs = Libs(
            "http://127.0.0.1:8080/v1/",
            model="local-alias",
            context_window=8192,
            api_key="test-key",
            timeout=17.0,
        )

    def test_exactly_ten_named_integrations_and_public_roles(self) -> None:
        self.assertEqual(
            Libs.names,
            ("openai", "litellm", "dspy", "instructor", "pydantic_ai", "smolagents", "llama_index", "httpx", "outlines", "guidance"),
        )
        roles = {
            "openai": "client", "litellm": "router", "dspy": "model",
            "instructor": "client", "pydantic_ai": "model",
            "smolagents": "model", "llama_index": "model",
            "httpx": "client", "outlines": "model", "guidance": "model",
        }
        self.assertEqual(len(roles), 10)
        for library, role in roles.items():
            self.assertTrue(hasattr(getattr(self.libs, library).__class__, role))
        self.assertFalse(hasattr(self.libs, "baml"))
        self.assertFalse(hasattr(self.libs, "llama_cpp_agent"))

    def test_url_normalization_and_shared_configuration(self) -> None:
        self.assertEqual(self.libs.server_url, "http://127.0.0.1:8080")
        self.assertEqual(self.libs.openai_url, "http://127.0.0.1:8080/v1")
        root = Libs("http://127.0.0.1:8080")
        self.assertEqual(root.server_url, self.libs.server_url)
        self.assertEqual(root.openai_url, self.libs.openai_url)
        for name in Libs.names:
            config = getattr(self.libs, name)._config
            self.assertEqual(config.model, "local-alias")
            self.assertEqual(config.context_window, 8192)
            self.assertEqual(config.api_key, "test-key")
            self.assertEqual(config.timeout, 17.0)

    def test_constructor_is_lazy_and_one_lib_does_not_import_others(self) -> None:
        with patch("zemi.arsenal.libs.import_module") as importer:
            libs = Libs("http://localhost:8080", model="alias")
            importer.assert_not_called()
            fake_httpx = type("HTTPX", (), {"Client": lambda **kwargs: kwargs})
            importer.return_value = fake_httpx
            self.assertEqual(libs.httpx.client["base_url"], libs.server_url)
            importer.assert_called_once_with("httpx")

    def test_missing_dependency_names_required_package(self) -> None:
        def missing(name: str):
            raise ModuleNotFoundError(name)

        with patch("zemi.arsenal.libs.import_module", side_effect=missing):
            with self.assertRaisesRegex(LibDependencyError, "assistant.libs.outlines.model требуется пакет 'outlines'"):
                _ = self.libs.outlines.model

    def test_real_types_cache_and_factory_makes_no_network_requests(self) -> None:
        import guidance.models
        import httpx
        import instructor
        import litellm
        import openai
        import outlines.models.openai
        import smolagents
        import dspy
        from llama_index.llms.openai_like import OpenAILike
        import pydantic_ai.models.openai as pydantic_openai

        expected = {
            "openai": ("client", openai.OpenAI),
            "litellm": ("router", litellm.Router),
            "dspy": ("model", dspy.LM),
            "instructor": ("client", instructor.Instructor),
            "pydantic_ai": ("model", tuple(
                model_type for model_type in (
                    getattr(pydantic_openai, "OpenAIModel", None),
                    getattr(pydantic_openai, "OpenAIChatModel", None),
                ) if model_type is not None
            )),
            "smolagents": ("model", smolagents.OpenAIServerModel),
            "llama_index": ("model", OpenAILike),
            "httpx": ("client", httpx.Client),
            "outlines": ("model", outlines.models.openai.OpenAI),
            "guidance": ("model", guidance.models.OpenAI),
        }
        old_connect = socket.socket.connect
        try:
            socket.socket.connect = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network request in factory"))
            for library, (role, expected_type) in expected.items():
                adapter = getattr(self.libs, library)
                first = getattr(adapter, role)
                self.assertIsInstance(first, expected_type, library)
                self.assertIs(first, getattr(adapter, role), library)
        finally:
            socket.socket.connect = old_connect


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import os
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

from zemi import arsenal
from zemi.arsenal import ArsenalSession, UnsupportedProtocolError
from zemi import toml


def external_config(**endpoint_overrides):
    endpoint = {
        "name": "host_llama", "kind": "external", "protocol": "openai",
        "base_url": "${TEST_LLM_URL}", "healthcheck": "models",
        "connect_timeout": 1.0, "request_timeout": 17.0,
        "validate_model": True,
        "models": [{"name": "host_qwen", "model": "remote-qwen",
                    "context_window": 32768,
                    "assistants": [{"name": "assistant"}]}],
    }
    endpoint.update(endpoint_overrides)
    return {"arsenal": {"mode": "model", "endpoints": [endpoint]}}


class _ModelsHandler(BaseHTTPRequestHandler):
    status = 200
    models = ["remote-qwen"]
    authorization = None

    def do_GET(self):
        type(self).authorization = self.headers.get("Authorization")
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"data": [{"id": item} for item in type(self).models]}).encode())

    def log_message(self, *_args):
        pass


class ArsenalEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _ModelsHandler.status = 200
        _ModelsHandler.models = ["remote-qwen"]
        _ModelsHandler.authorization = None

    def test_external_tree_env_and_clients_are_lazy(self):
        with patch.dict(os.environ, {"TEST_LLM_URL": self.url, "HOST_KEY": "secret"}):
            session = ArsenalSession(external_config(api_key_env="HOST_KEY"))
        self.assertEqual(session.endpoints.host_llama.models.host_qwen.name, "host_qwen")
        model = session.models["host_qwen"]
        libs = model.assistants.assistant.clients
        self.assertEqual((libs.server_url, libs.model, libs.context_window, libs.timeout),
                         (self.url, "remote-qwen", 32768, 17.0))
        self.assertNotIn("secret", repr(libs.openai._config))
        self.assertNotIn("secret", repr(session.resolved_config()))

    def test_external_begin_end_never_start_or_stop(self):
        with patch.dict(os.environ, {"TEST_LLM_URL": self.url}):
            session = ArsenalSession(external_config())
        with patch("zemi.arsenal.runtime.subprocess.Popen") as popen:
            with redirect_stdout(StringIO()):
                arsenal.begin(session, stop_before_begin=True)
                arsenal.end(session, stop_after_end=True)
        popen.assert_not_called()
        self.assertEqual(session._processes, {})

    def test_models_healthcheck_and_bearer_key(self):
        with patch.dict(os.environ, {"TEST_LLM_URL": self.url, "HOST_KEY": "secret"}):
            session = ArsenalSession(external_config(api_key_env="HOST_KEY"))
            session.check("host_llama", "host_qwen")
        self.assertEqual(_ModelsHandler.authorization, "Bearer secret")
        _ModelsHandler.models = ["different"]
        with self.assertRaisesRegex(LookupError, "host_llama.*host_qwen.*not found"):
            session.check("host_llama", "host_qwen")

    def test_none_and_tcp_healthchecks(self):
        with patch.dict(os.environ, {"TEST_LLM_URL": self.url}):
            ArsenalSession(external_config(healthcheck="none")).check()
            ArsenalSession(external_config(healthcheck="tcp")).check()

    def test_authentication_diagnostic_is_secret_safe(self):
        _ModelsHandler.status = 401
        with patch.dict(os.environ, {"TEST_LLM_URL": self.url, "HOST_KEY": "secret"}):
            session = ArsenalSession(external_config(api_key_env="HOST_KEY"))
            with self.assertRaisesRegex(ConnectionError, "host_llama.*authentication") as raised:
                session.check()
        self.assertNotIn("secret", str(raised.exception))

    def test_strict_schema_errors(self):
        cases = [
            ({"kind": "bad"}, "kind"), ({"protocol": "bad"}, "protocol"),
            ({"healthcheck": "bad"}, "healthcheck"),
            ({"connect_timeout": 0}, "connect_timeout"),
            ({"runtime": {}}, "external.*runtime"),
        ]
        for changes, message in cases:
            with self.subTest(changes=changes), patch.dict(os.environ, {"TEST_LLM_URL": self.url}):
                with self.assertRaisesRegex(ValueError, message):
                    ArsenalSession(external_config(**changes))
        with self.assertRaisesRegex(ValueError, "environment variable 'MISSING'"):
            ArsenalSession(external_config(base_url="${MISSING}"))

    def test_global_model_and_endpoint_names_are_unique(self):
        with patch.dict(os.environ, {"TEST_LLM_URL": self.url}):
            config = external_config()
            config["arsenal"]["endpoints"].append(config["arsenal"]["endpoints"][0].copy())
            with self.assertRaisesRegex(ValueError, "duplicate arsenal.endpoints"):
                ArsenalSession(config)

    def test_anthropic_is_explicit_unsupported_client_boundary(self):
        with patch.dict(os.environ, {"TEST_LLM_URL": self.url}):
            session = ArsenalSession(external_config(protocol="anthropic", healthcheck="none"))
        with self.assertRaisesRegex(UnsupportedProtocolError, "native Anthropic"):
            _ = session.models["host_qwen"].assistants.assistant.clients.openai.client

    def test_managed_session_does_not_kill_unowned_port_process(self):
        config = {"arsenal": {"mode": "model", "endpoints": [{
            "name": "managed", "kind": "managed", "protocol": "openai",
            "runtime": {"engine": "llama:b1", "host": "127.0.0.1", "port": 8088,
                        "startup_timeout": 1},
            "models": [{"name": "local", "model": "local-id",
                        "artifact": {"source": "hf", "owner": "o", "repository": "r", "filename": "f.gguf"},
                        "runtime": {"ctx_size": 1024, "threads": 1, "threads_batch": 1, "reasoning": "off"}}]
        }]}}
        session = ArsenalSession(config)
        with patch("zemi.arsenal.runtime.subprocess.run") as unsafe:
            with redirect_stdout(StringIO()):
                session._stop_arsenal()
        unsafe.assert_not_called()

    def test_tracked_endpoint_examples_parse(self):
        from pathlib import Path
        root = Path(__file__).parents[1] / "zemi"
        env = {"ZEMI_HOST_LLM_BASE_URL": self.url,
               "OPENROUTER_API_KEY": "placeholder",
               "DIRECT_PROVIDER_BASE_URL": self.url,
               "DIRECT_PROVIDER_API_KEY": "placeholder"}
        with patch.dict(os.environ, env):
            for name in ("llm_external_local_example.toml",
                         "llm_external_providers_example.toml",
                         "llm_managed_endpoint_example.toml"):
                with self.subTest(name=name):
                    ArsenalSession(toml.load(root / name))


if __name__ == "__main__":
    unittest.main()

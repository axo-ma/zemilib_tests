from __future__ import annotations

import json
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from zemi import arsenal, env, toml
from zemi.arsenal import ArsenalSession, UnsupportedProtocolError
from zemi.arsenal.secrets import ArsenalEnvError, SecretStore


def ref(name, secret=False, validate="non_empty", suggested=None):
    value = {"env": name, "prompt": f"Enter {name}", "validate": validate}
    if secret:
        value["secret"] = True
    if suggested is not None:
        value["suggested"] = suggested
    return value


def external(url, **changes):
    endpoint = {
        "name": "host_llm",
        "kind": "external",
        "protocol": "openai",
        "provider": "custom",
        "base_url": url,
        "authentication": "none",
        "healthcheck": "models",
        "connect_timeout": 1.0,
        "request_timeout": 17.0,
        "validate_model": True,
        "models": [{
            "name": "host_model",
            "model": "remote-qwen",
            "context_window": 32768,
            "assistants": [{"name": "assistant"}],
        }],
    }
    endpoint.update(changes)
    return {"arsenal": {"mode": "model", "endpoints": [endpoint]}}


class Handler(BaseHTTPRequestHandler):
    status, models, authorization = 200, ["remote-qwen"], None

    def do_GET(self):
        type(self).authorization = self.headers.get("Authorization")
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps({"data": [{"id": item} for item in type(self).models]}).encode()
        )

    def log_message(self, *_):
        pass


class SecretStoreTests(unittest.TestCase):
    def setUp(self):
        self.root = env.path.tmp / f"arsenal-test-{os.getpid()}-{id(self)}"
        self.root.mkdir(parents=True)
        self.path = self.root / "arsenal.env"
        self.store = SecretStore(self.path)

    def tearDown(self):
        for item in self.root.iterdir():
            item.unlink()
        self.root.rmdir()

    def test_existing_value_no_prompt_or_process_environment(self):
        self.path.write_text("KEY=file-value\n", encoding="utf-8")
        with patch.dict(os.environ, {"KEY": "process-value"}), patch("builtins.input") as ask:
            self.assertEqual(self.store.resolve(ref("KEY")), "file-value")
        ask.assert_not_called()

    def test_prompt_save_quote_and_second_read(self):
        with patch("getpass.getpass", return_value="s e c # ret"):
            self.assertEqual(self.store.resolve(ref("KEY", secret=True)), "s e c # ret")
        with patch("getpass.getpass") as ask:
            self.assertEqual(SecretStore(self.path).resolve(ref("KEY", secret=True)), "s e c # ret")
        ask.assert_not_called()
        self.assertIn('KEY="s e c # ret"', self.path.read_text(encoding="utf-8"))

    def test_suggested_retry_preserves_comments_and_other_keys(self):
        self.path.write_text("# keep\nOTHER=x\nURL=bad\n", encoding="utf-8")
        with patch("builtins.input", side_effect=["bad", ""]):
            self.assertEqual(
                self.store.resolve(ref("URL", validate="url", suggested="http://host:8080/v1")),
                "http://host:8080/v1",
            )
        text = self.path.read_text(encoding="utf-8")
        self.assertIn("# keep", text)
        self.assertIn("OTHER=x", text)

    def test_malformed_duplicate_eof_and_concurrency(self):
        for text in ("broken\n", "A=1\nA=2\n", "BAD-NAME=x\n", 'A="bad\n'):
            self.path.write_text(text, encoding="utf-8")
            with self.subTest(text=text), self.assertRaises(ArsenalEnvError):
                self.store.get("A")
        self.path.unlink()
        with patch("builtins.input", side_effect=EOFError), self.assertRaisesRegex(ArsenalEnvError, "unavailable"):
            self.store.resolve(ref("A"))
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda number: SecretStore(self.path).set(f"K{number}", str(number)), range(8)))
        self.assertEqual(len(self.store._read()[1]), 8)


class EndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        Handler.status, Handler.models, Handler.authorization = 200, ["remote-qwen"], None
        self.path = env.path.tmp / f"arsenal-session-{os.getpid()}-{id(self)}.env"

    def tearDown(self):
        if self.path.exists():
            self.path.unlink()

    def test_lazy_references_and_client_fields(self):
        config = external(ref("BASE", validate="url"), authentication="bearer", api_key=ref("KEY", True))
        config["arsenal"]["endpoints"][0]["models"][0]["model"] = ref("MODEL")
        self.path.write_text(f"BASE={self.url}\nKEY=secret\nMODEL=remote-qwen\n", encoding="utf-8")
        with patch("builtins.input") as plain, patch("getpass.getpass") as hidden:
            session = ArsenalSession(config, _secret_store_path=self.path)
            plain.assert_not_called()
            hidden.assert_not_called()
            libs = session.endpoints.host_llm.models.host_model.assistants.assistant.clients
        self.assertEqual(
            (libs.server_url, libs.model, libs.context_window, libs.timeout),
            (self.url, "remote-qwen", 32768, 17.0),
        )
        self.assertNotIn("secret", repr(libs.openai._config))
        self.assertNotIn("secret", repr(session.resolved_config()))

    def test_external_lifecycle_and_health_diagnostics(self):
        session = ArsenalSession(external(self.url), _secret_store_path=self.path)
        with patch("zemi.arsenal.runtime.subprocess.Popen") as popen:
            arsenal.begin(session, stop_before_begin=True)
            arsenal.end(session, stop_after_end=True)
        popen.assert_not_called()
        session.check()
        Handler.models = ["other"]
        with self.assertRaisesRegex(LookupError, "not found"):
            session.check()

    def test_first_external_model_access_after_begin_does_not_recurse(self):
        session = ArsenalSession(external(self.url), _secret_store_path=self.path)
        arsenal.begin(session, stop_before_begin=False)
        model = session.endpoints.host_llm.models.host_model
        self.assertEqual(model.model, "remote-qwen")
        self.assertIn(("host_llm", "host_model"), session._validated_external)

    def test_bearer_auth_redaction(self):
        self.path.write_text("KEY=secret\n", encoding="utf-8")
        session = ArsenalSession(
            external(self.url, authentication="bearer", api_key=ref("KEY", True)),
            _secret_store_path=self.path,
        )
        session.check()
        self.assertEqual(Handler.authorization, "Bearer secret")
        Handler.status = 401
        with self.assertRaisesRegex(ConnectionError, "authentication") as raised:
            session.check()
        self.assertNotIn("secret", str(raised.exception))

    def test_schema_and_protocol_boundaries(self):
        cases = [
            ({"kind": "bad"}, "kind"),
            ({"protocol": "bad"}, "protocol"),
            ({"healthcheck": "bad"}, "healthcheck"),
            ({"connect_timeout": 0}, "connect_timeout"),
            ({"api_key": "literal", "authentication": "bearer"}, "env reference"),
            ({"api_key": ref("KEY"), "authentication": "bearer"}, "secret = true"),
            ({"runtime": {}}, "external.*runtime"),
        ]
        for changes, message in cases:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                ArsenalSession(external(self.url, **changes), _secret_store_path=self.path)
        native = ArsenalSession(
            external(self.url, protocol="anthropic", healthcheck="none"),
            _secret_store_path=self.path,
        )
        with self.assertRaisesRegex(UnsupportedProtocolError, "native Anthropic"):
            _ = native.endpoints.host_llm.models.host_model.assistants.assistant.clients.openai.client

    def test_endpoint_and_model_names_are_unique(self):
        config = external(self.url)
        config["arsenal"]["endpoints"].append(config["arsenal"]["endpoints"][0].copy())
        with self.assertRaisesRegex(ValueError, "duplicate arsenal.endpoints"):
            ArsenalSession(config, _secret_store_path=self.path)

    def test_managed_session_does_not_kill_unowned_port_process(self):
        config = {"arsenal": {"mode": "model", "endpoints": [{
            "name": "managed",
            "kind": "managed",
            "protocol": "openai",
            "runtime": {
                "engine": "llama:b1",
                "host": "127.0.0.1",
                "port": 8088,
                "startup_timeout": 1,
            },
            "models": [{
                "name": "local",
                "model": "local-id",
                "artifact": {
                    "source": "hf",
                    "owner": "o",
                    "repository": "r",
                    "filename": "f.gguf",
                },
                "runtime": {
                    "ctx_size": 1024,
                    "threads": 1,
                    "threads_batch": 1,
                    "reasoning": "off",
                },
            }],
        }]}}
        session = ArsenalSession(config)
        with patch("zemi.arsenal.runtime.subprocess.run") as unsafe, redirect_stdout(StringIO()):
            session._stop_arsenal()
        unsafe.assert_not_called()

    def test_tracked_configs_are_lazy_and_legacy_normalizes(self):
        root = Path(arsenal.__file__).parents[1]
        with patch("builtins.input") as plain, patch("getpass.getpass") as hidden:
            sessions = [
                ArsenalSession(toml.load(root / name), _secret_store_path=self.path)
                for name in (
                    "llm_curated_set_model_mode.toml",
                    "llm_curated_set_router_mode.toml",
                    "llm_external_local.toml",
                    "llm_external_providers.toml",
                )
            ]
        plain.assert_not_called()
        hidden.assert_not_called()
        self.assertEqual(sessions[1].endpoints.curated_router.kind, "managed")


if __name__ == "__main__":
    unittest.main()

"""Conversation fidelity, lifecycle configuration and compact report links."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from zemi import env
from zemi.conversation import instrument_client, notebook_contexts
from zemi.report_chat import continue_chat, create_input_session, load_context, prepare_session
from zemi.report_viewer import render_markdown, write_launcher


class ReportChatTests(unittest.TestCase):
    def setUp(self):
        env.path.tmp.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=env.path.tmp)
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_pasted_multiline_request_is_one_message(self):
        from prompt_toolkit.input.defaults import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        with create_pipe_input() as terminal_input:
            session = create_input_session(input=terminal_input, output=DummyOutput())
            terminal_input.send_text('\x1b[200~First line\nSecond line\x1b[201~\r')
            self.assertEqual(session.prompt('Вы > '), 'First line\nSecond line')

    def test_capture_uses_exact_messages_response_and_generation_settings(self):
        from zemi import conversation
        settings = SimpleNamespace(openai_url='http://localhost:9191/v1')
        message = {'role': 'assistant', 'content': 'Original answer'}
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(model_dump=lambda **kw: message))])
        calls, snapshots = [], []
        def create(**kwargs):
            calls.append(deepcopy(kwargs))
            return response
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        source = {'model_name': 'original', 'config_path': 'config'}
        request = {'model': 'alias', 'messages': [{'role': 'system', 'content': 'Rules'}, {'role': 'user', 'content': 'Table'}],
                   'temperature': .2, 'max_tokens': 500, 'extra_body': {'top_k': 30}}
        with patch.dict(os.environ, {'ZEMI_PLAYBOOK_OUTPUT_DIR': str(self.root)}), \
             patch.dict(conversation._sources, {('http://localhost:9191', 'alias'): source}, clear=True), \
             patch('zemi.conversation.publish', side_effect=lambda c: snapshots.append(deepcopy(c))):
            instrument_client(client, settings)
            self.assertIs(client.chat.completions.create(**request), response)
        self.assertEqual(calls, [request])
        self.assertEqual(snapshots[0]['request'], request)
        self.assertEqual(snapshots[0]['assistant'], message)
        self.assertEqual(snapshots[0]['source'], source)
        request['messages'][0]['content'] = 'Changed'
        self.assertEqual(snapshots[0]['request']['messages'][0]['content'], 'Rules')

    def test_failed_generation_does_not_append_user_message(self):
        request = {'model': 'alias', 'temperature': .3, 'messages': [{'role': 'assistant', 'content': 'Before'}]}
        before = deepcopy(request)
        def fail(**kw):
            raise RuntimeError('context full')
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
        with self.assertRaisesRegex(RuntimeError, 'context full'):
            continue_chat(client, request, 'Why?')
        self.assertEqual(request, before)

    def test_continuation_retains_roles_and_parameters(self):
        request = {'model': 'alias', 'temperature': .3, 'max_tokens': 900,
                   'messages': [{'role': 'system', 'content': 'Rules'}, {'role': 'user', 'content': 'Original'}, {'role': 'assistant', 'content': 'Answer'}]}
        sent = []
        answer = {'role': 'assistant', 'content': 'Explanation'}
        def create(**kw):
            sent.append(deepcopy(kw))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(model_dump=lambda **kw: answer))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        self.assertEqual(continue_chat(client, request, 'Why?'), 'Explanation')
        self.assertEqual([m['role'] for m in request['messages']], ['system', 'user', 'assistant', 'user', 'assistant'])
        self.assertEqual(sent[0]['temperature'], .3)
        self.assertEqual(sent[0]['max_tokens'], 900)

    def test_viewer_keeps_disclosures_and_checkmark_without_extra_labels(self):
        path = self.root / 'm.dataset.md'
        path.write_text('| Item ID | Matches | Target | Sample 1 | Sample 2 |\n|---|---|---|---|---|\n'
                        '| item | 1 / 2 | [] | <details><summary>A1:B2...</summary><pre>full</pre></details> | ✅ |\n', encoding='utf-8')
        context = {'source': {'model_name': 'm'}}
        path.with_suffix('.chat.json').write_text(json.dumps({'rows': [{'item_id': 'item', 'cells': [
            {'contexts': [context]}, {'contexts': [context]}]}]}), encoding='utf-8')
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(render_markdown(path), 'html.parser')
        self.assertEqual(soup.select_one('summary a')['href'], 'zemi-chat:0:0')
        self.assertEqual(soup.select_one('summary a').get_text(), 'A1:B2...')
        self.assertEqual(soup.select('tbody td')[-1].get_text(), '✅')
        self.assertEqual(soup.select('tbody td')[-1].a['href'], 'zemi-chat:0:1')
        self.assertEqual(soup.pre.get_text(), 'full')

    def test_report_scripts_and_event_handlers_are_not_executable(self):
        path = self.root / 'item.md'
        path.write_text('<script>bad()</script><img src="x" onerror="bad()"><a href="javascript:bad()">x</a>', encoding='utf-8')
        markup = render_markdown(path)
        self.assertNotIn('bad()', markup)
        self.assertNotIn('onerror', markup)

    def test_old_cell_without_capture_fails_clearly(self):
        manifest = self.root / 'm.dataset.chat.json'
        manifest.write_text(json.dumps({'rows': [{'item_id': 'i', 'cells': [{'contexts': []}]}]}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'no unique captured'):
            load_context(manifest, 0, 0)

    def test_notebook_capture_is_separate_from_prediction(self):
        from zemi.conversation import CHAT_MIME
        context = {'request': {'messages': []}}
        self.assertEqual(notebook_contexts({'cells': [{'outputs': [{'data': {CHAT_MIME: context}}]}]}), [context])

    def test_managed_chat_uses_same_model_new_port_and_large_context(self):
        config = self.root / 'llm.toml'
        config.write_text('''[arsenal]
mode="model"
[[arsenal.llamas]]
name="local"
llama_build="llama:b1234"
host="127.0.0.1"
port=8888
startup_timeout=120
[[arsenal.llamas.models]]
name="test"
alias="alias"
source="hf"
owner="owner"
repository="repo"
filename="model.gguf"
ctx_size=8192
threads=4
threads_batch=4
reasoning="off"
''', encoding='utf-8')
        source = {'config_path': str(config), 'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
                  'component_root': str(self.root), 'endpoint_name': 'local', 'model_name': 'test'}
        previous = Path.cwd()
        try:
            session, endpoint, model, kind = prepare_session(source)
            runtime = session._managed_llamas[endpoint]
            self.assertNotEqual(runtime.port, 8888)
            restored = runtime.models._get_raw(model)
            self.assertEqual(restored.ctx_size, 32768)
            self.assertEqual(restored.filename, 'model.gguf')
            self.assertEqual(restored.threads, 4)
            config.write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed since'):
                prepare_session(source)
        finally:
            os.chdir(previous)

    def test_launcher_points_to_report_next_to_it(self):
        path = self.root / 'my.dataset.md'
        write_launcher(path)
        content = path.with_suffix('.cmd').read_text(encoding='utf-8')
        self.assertIn('%~dp0my.dataset.md', content)
        self.assertIn('report_viewer.py', content)

    def test_external_unauthenticated_session_keeps_provider_context(self):
        config = self.root / 'external.toml'
        config.write_text('''[arsenal]
mode="model"
[[arsenal.endpoints]]
name="remote"
kind="external"
protocol="openai"
base_url="http://localhost:9191/v1"
authentication="none"
healthcheck="none"
[[arsenal.endpoints.models]]
name="test"
model="alias"
context_window=8192
''', encoding='utf-8')
        source = {'config_path': str(config), 'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
                  'component_root': str(self.root), 'endpoint_name': 'remote', 'model_name': 'test'}
        previous = Path.cwd()
        try:
            session, endpoint, model, kind = prepare_session(source)
            self.assertEqual(kind, 'external')
            self.assertEqual(session.endpoints[endpoint].models[model].context_window, 8192)
            self.assertEqual(session._managed_llamas, {})
        finally:
            os.chdir(previous)

    def test_real_sdk_capture_and_followup_use_the_same_request(self):
        import httpx
        from openai import OpenAI
        requests, captures = [], []
        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={'id': 'c1', 'object': 'chat.completion', 'created': 1,
                'model': 'alias', 'choices': [{'index': 0, 'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': 'Answer'}}]})
        client = OpenAI(base_url='http://localhost:9191/v1', api_key='secret-not-saved',
                        http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        self.addCleanup(client.close)
        with patch.dict(os.environ, {'ZEMI_PLAYBOOK_OUTPUT_DIR': str(self.root)}), \
             patch('zemi.conversation.publish', side_effect=lambda c: captures.append(deepcopy(c))):
            instrument_client(client, SimpleNamespace(openai_url='http://localhost:9191/v1'))
            client.chat.completions.create(model='alias', messages=[{'role': 'user', 'content': 'Original'}], temperature=.2, max_tokens=200)
        context = captures[0]
        request = context['request']
        request['messages'].append(context['assistant'])
        self.assertEqual(continue_chat(client, request, 'Why?'), 'Answer')
        self.assertEqual([m['role'] for m in requests[-1]['messages']], ['user', 'assistant', 'user'])
        self.assertNotIn('secret-not-saved', json.dumps(context))


class StreamingChatTests(unittest.TestCase):
    def test_stream_preserves_text_settings_usage_and_closes(self):
        from unittest.mock import Mock
        request = {'model': 'alias', 'temperature': .3, 'max_tokens': 900,
                   'messages': [{'role': 'assistant', 'content': 'Before'}]}
        chunk = lambda text, finish=None: SimpleNamespace(choices=[SimpleNamespace(
            index=0, delta=SimpleNamespace(content=text), finish_reason=finish)], usage=None)
        stream = Mock()
        stream.__iter__ = Mock(return_value=iter([chunk('Привет'),chunk(' мир','stop'),
            SimpleNamespace(choices=[],usage=SimpleNamespace(completion_tokens=7))]))
        create = Mock(return_value=stream)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        parts, stats = [], {}
        self.assertEqual(continue_chat(client,request,'Why?',on_token=parts.append,stats=stats),'Привет мир')
        self.assertEqual(parts,['Привет',' мир'])
        self.assertEqual(stats['completion_tokens'],7)
        sent=create.call_args.kwargs
        self.assertTrue(sent['stream'])
        self.assertEqual(sent['stream_options'],{'include_usage':True})
        self.assertEqual(sent['temperature'],.3)
        self.assertEqual(sent['max_tokens'],900)
        self.assertEqual(request['messages'][-1],{'role':'assistant','content':'Привет мир'})
        self.assertFalse(request['stream'])
        self.assertNotIn('stream_options',request)
        stream.close.assert_called_once()

    def test_broken_stream_keeps_history_and_closes(self):
        from unittest.mock import Mock
        request={'model':'alias','messages':[{'role':'assistant','content':'Before'}]}
        before=deepcopy(request)
        stream=Mock()
        stream.__iter__=Mock(return_value=iter([SimpleNamespace(choices=[SimpleNamespace(
            index=0,delta=SimpleNamespace(content='partial'),finish_reason=None)],usage=None)]))
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=stream))))
        with self.assertRaisesRegex(RuntimeError,'ended before completion'):
            continue_chat(client,request,'Why?',on_token=lambda text:None)
        self.assertEqual(request,before)
        stream.close.assert_called_once()

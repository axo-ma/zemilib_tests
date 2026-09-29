"""Shared managed-server logging keeps console output readable."""
import io
import sys
import tempfile
from pathlib import Path
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch, Mock
from zemi import env
from zemi.arsenal.runtime import ArsenalSession


class ServerLogTests(TestCase):
    def test_real_failed_process_retains_diagnostics_without_ansi(self):
        env.path.tmp.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=env.path.tmp) as tmp:
            session=object.__new__(ArsenalSession)
            session._processes={};session._server_logs={}
            llama=SimpleNamespace(name='test',host='127.0.0.1',port=1,startup_timeout=3)
            output=io.StringIO()
            with patch('zemi.arsenal.runtime.env',SimpleNamespace(path=SimpleNamespace(tmp=Path(tmp)))), \
                 patch.object(session,'_is_server_ready',return_value=False), redirect_stdout(output):
                with self.assertRaisesRegex(RuntimeError,'load failed') as failure:
                    session._start_server(llama,[sys.executable,'-c',
                        "import sys;print('server stdout');print('\\x1b[31mload failed\\x1b[0m',file=sys.stderr);sys.exit(1)"])
            self.assertEqual(output.getvalue(),'')
            self.assertNotIn('\x1b',str(failure.exception))
            log=session._server_logs['test'].read_text(encoding='utf-8')
            self.assertIn('server stdout',log)
            self.assertIn('load failed',log)
            self.assertEqual(session._processes,{})

    def test_ready_server_redirects_stdout_and_stderr_for_all_callers(self):
        env.path.tmp.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=env.path.tmp) as tmp:
            session=object.__new__(ArsenalSession)
            session._processes={};session._server_logs={}
            llama=SimpleNamespace(name='test',host='127.0.0.1',port=1,startup_timeout=3)
            process=Mock(pid=123);process.poll.return_value=None
            with patch('zemi.arsenal.runtime.env',SimpleNamespace(path=SimpleNamespace(tmp=Path(tmp)))), \
                 patch.object(session,'_is_server_ready',side_effect=[False,True]), \
                 patch('zemi.arsenal.runtime.subprocess.Popen',return_value=process) as launch, \
                 redirect_stdout(io.StringIO()) as output:
                session._start_server(llama,['server'])
            self.assertIn('server ready',output.getvalue())
            self.assertIn('Server log:',output.getvalue())
            self.assertTrue(launch.call_args.kwargs['stdout'].closed)
            self.assertEqual(launch.call_args.kwargs['stderr'],-2)
            self.assertTrue(session._server_logs['test'].is_file())

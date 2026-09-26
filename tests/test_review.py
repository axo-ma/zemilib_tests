import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zemi import env
from zemi.review import capture_review, render_review
from zemi.reporting import ReportWriter


class ReviewTests(unittest.TestCase):
    def test_automatic_snapshot_captures_configuration_prompts_sources_and_data(self):
        env.path.tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=env.path.tmp) as directory:
            root = Path(directory)
            (root / 'zemi').mkdir()
            for name in ('job.py', 'params.toml', 'playbook.ipynb', 'trial.py'):
                (root / name).write_text('original', encoding='utf-8')
            (root / 'prompts.md').write_text('# custom\n### Input\n```text\nexample\n```\n{{item}}\n', encoding='utf-8')
            (root / 'encoder.py').write_text('def encode(*args, **kwargs): return "text"', encoding='utf-8')
            (root / 'data.json').write_text('{"items":[]}', encoding='utf-8')
            (root / 'book.xlsx').write_bytes(b'fixture bytes')
            (root / 'arsenal.toml').write_text('[[arsenal.llamas]]\nllama_build = "b1234"\n[[arsenal.llamas.models]]\nname = "chosen"\nowner = "owner"\nrepository = "repo"\nfilename = "model.gguf"\nctx_size = 4096\n', encoding='utf-8')
            binding = {'prompt_name': 'custom', 'prompt_file': '@comp/prompts.md',
                       'encoder': '@comp/encoder.py:encode', 'encoding_format': 'cells'}
            module = SimpleNamespace(source_path=root / 'playbook.ipynb', playbook_name='playbook.ipynb',
                params={'arsenal_config_path': '@comp/arsenal.toml', 'model_name': 'chosen', 'temperature': 0},
                config={'_v05_space': {'encoding_prompt': binding}},
                optimizer_config={'strategy': 'grid', 'max_trials': 8,
                    'trial_dataset': {'path': '@comp/data.json'}, 'sample_trial': {'type': '@comp/trial.py:Trial'}})
            component = SimpleNamespace(root=root, params_path=root / 'params.toml')
            dataset = SimpleNamespace(items=[{'input': {'workbook_path': '@comp/book.xlsx'}}])
            path = lambda ref: root / ref.removeprefix('@comp/')
            with patch('sys.argv', [str(root / 'job.py')]), patch('zemi.review.zemi_path', side_effect=path), patch('zemi.prompting.zemi_path', side_effect=path):
                snapshot = capture_review(component, module, dataset)
            self.assertEqual(snapshot['configuration']['Model'], 'hf:owner/repo/model.gguf')
            self.assertEqual(snapshot['configuration']['Runtime'], 'b1234')
            self.assertEqual(snapshot['entrypoint'], 'job.py')
            self.assertEqual(set(snapshot['sources']), {'job.py', 'params.toml', 'playbook.ipynb',
                'prompts.md', 'encoder.py', 'data.json', 'trial.py', 'arsenal.toml'})
            self.assertEqual(len(snapshot['data_files']['@comp/book.xlsx']), 64)
            (root / 'prompts.md').write_text('changed', encoding='utf-8')
            self.assertIn('{{item}}', snapshot['prompts']['custom'])
            self.assertIn('{{item}}', snapshot['sources']['prompts.md'])
            writer = ReportWriter(root / 'run-test')
            writer.register_module('m', optimized=True)
            writer.register_review('m')
            samples = [{'id': f'custom-{i}', 'params': {'temperature': i},
                'score': 0.666666, 'runs': [
                    {'status': 'succeeded', 'prediction': {'item_tokens': 10, 'prompt_tokens': 100}},
                    {'status': 'failed', 'prediction': None, 'evaluation_error': {'message': 'failure'}}]}
                for i in range(3)]
            body = render_review(json.loads(json.dumps(snapshot)), samples=samples, report={'status': 'failed'},
                module_id='m', writer=writer, item_count=17)
            self.assertIn('| Samples | 3 |', body)
            self.assertIn('| Dataset items | 17 |', body)
            self.assertIn('0.667', body)
            self.assertIn('10.000', body)
            self.assertIn('3 / 6', body)
            self.assertIn('````text', body)
            for i in range(3):
                self.assertIn(f'custom-{i}', body)

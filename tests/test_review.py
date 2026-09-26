import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
from zemi import env
from zemi.review import capture_review, render_review, configure_review
from zemi.reporting import ReportWriter


class ReviewTests(unittest.TestCase):
    def test_standard_setup_collects_runtime_and_custom_metadata(self):
        env.path.tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=env.path.tmp) as directory:
            config = Path(directory) / 'arsenal.toml'
            config.write_text('''[[arsenal.llamas]]
llama_build = "b1234"
[[arsenal.llamas.models]]
name = "chosen"
owner = "owner"
repository = "repo"
filename = "model.gguf"
ctx_size = 4096
threads = 4
reasoning = false
''', encoding='utf-8')
            module = SimpleNamespace(module_id='m', optimizer_config={'mode': 'optimize'},
                params={'arsenal_config_path': '@comp/arsenal.toml', 'model_name': 'chosen',
                        'temperature': 0, 'max_tokens': 100})
            component = SimpleNamespace(modules=[module], reporting=Mock())
            with patch('zemi.review.zemi_path', return_value=config):
                configure_review(component, '@comp/job.py', settings={'Score': 'F1'},
                    prompts={'plain': 'prompt'}, sources=['@comp/spec.md'])
            call = component.reporting.configure_review.call_args
            self.assertEqual(call.args, ('m',))
            self.assertEqual(call.kwargs['settings']['Model'], 'hf:owner/repo/model.gguf')
            self.assertEqual(call.kwargs['settings']['Runtime'], 'b1234')
            self.assertEqual(call.kwargs['settings']['Score'], 'F1')
            self.assertEqual(call.kwargs['sources'], ['@comp/spec.md', '@comp/arsenal.toml'])

    def test_standard_setup_selects_optimized_modules_without_experiment_dependencies(self):
        component = SimpleNamespace(reporting=Mock(), modules=[
            SimpleNamespace(module_id='plain', optimizer_config=None, params={}),
            SimpleNamespace(module_id='a', optimizer_config={'mode': 'start_only'}, params={}),
            SimpleNamespace(module_id='b', optimizer_config={'mode': 'optimize'}, params={})])
        configure_review(component, '@comp/job.py')
        self.assertEqual([c.args[0] for c in component.reporting.configure_review.call_args_list], ['a', 'b'])
        component.reporting.reset_mock()
        configure_review(component, '@comp/job.py', module_id='b')
        self.assertEqual(component.reporting.configure_review.call_args.args, ('b',))
        with self.assertRaises(ValueError):
            configure_review(component, '@comp/job.py', module_id='missing')

    def test_saved_prompt_and_source_survive_changes_and_partial_results_render(self):
        env.path.tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=env.path.tmp) as directory:
            root = Path(directory)
            (root / 'zemi').mkdir()
            for filename in ('job.py', 'params.toml', 'playbook.ipynb', 'prompts.md'):
                (root / filename).write_text('original', encoding='utf-8')
            component = SimpleNamespace(root=root, params_path=root / 'params.toml')
            module = SimpleNamespace(source_path=root / 'playbook.ipynb',
                                     playbook_name='playbook.ipynb', optimizer_config={'strategy': 'grid', 'max_trials': 8})
            with patch('zemi.review.zemi_path', side_effect=lambda p: root / p):
                snapshot = capture_review(component, module, entrypoint='job.py', settings={'Temperature': 0.0},
                    prompts={'cell_all': '### Input\n```data```\n### Output\n{"ranges":[]}'},
                    sources=['prompts.md'], repositories=[])
            (root / 'prompts.md').write_text('changed', encoding='utf-8')
            self.assertEqual(snapshot['sources']['prompts.md'], 'original')
            snapshot = json.loads(json.dumps(snapshot))
            writer = ReportWriter(root / 'run-test')
            writer.register_module('m', optimized=True)
            writer.register_review('m')
            body = render_review(snapshot, samples=[{'params': {'encoding_format': 'cell_all'},
                'score': 0.666666, 'runs': [
                    {'status': 'succeeded', 'prediction': {'item_tokens': 10, 'prompt_tokens': 100}},
                    {'status': 'failed', 'prediction': None, 'evaluation_error': {'message': 'failure'}}]}],
                report={'status': 'failed'}, module_id='m', writer=writer, item_count=2)
            writer.write_review_report('m', body)
            self.assertIn('0.667', body)
            self.assertIn('10.000', body)
            self.assertIn('1 / 2', body)
            self.assertIn('````text', body)
            self.assertIn('### Output', body)
            self.assertNotIn('Array score', body)
            self.assertNotIn('| Change |', body)
            self.assertIn('Module Report', (root / 'run-test/m.review.md').read_text(encoding='utf-8'))
            samples = [{'sample_trial_id': f'custom-{i}',
                        'params': {'encoding_format': 'plain', 'temperature': i},
                        'runs': []} for i in range(3)]
            body = render_review(snapshot, samples=samples, report={'status': 'running'},
                module_id='m', writer=writer, item_count=17)
            self.assertIn('| Samples | 3 |', body)
            self.assertIn('| Dataset items | 17 |', body)
            for i in range(3):
                self.assertIn(f'custom-{i}', body)
                self.assertIn(f'"temperature": {i}', body)

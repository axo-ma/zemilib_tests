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

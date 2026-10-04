import unittest
from types import SimpleNamespace

from zemi.dataset import table_evaluator
from zemi.reporting import _dataset_prediction_cell


class ModelResponseErrorTests(unittest.TestCase):
    def test_response_error_is_penalized_without_execution_failure(self):
        run = {'status': 'succeeded', 'item': {'id': 'i', 'input': {},
            'ground_truth': ['A1:B4'], 'tags': []}, 'error': None,
            'prediction': {'raw_response': 'Explanation <script>\n{"ranges":[]}',
                           'response_error': 'Expected JSON with ranges'}}
        metrics, _ = table_evaluator(SimpleNamespace(runs=[run]), params={})
        self.assertEqual(metrics['errors'], 1)
        self.assertEqual(metrics['tp'], 0)
        self.assertEqual(metrics['fp'], 1)
        self.assertEqual(metrics['fn'], 1)
        self.assertEqual(run['status'], 'succeeded')
        cell = _dataset_prediction_cell(run)
        self.assertIn('Model response error', cell)
        self.assertIn('Explanation &lt;script&gt;', cell)
        self.assertNotIn('Execution failure', cell)

    def test_execution_failure_remains_distinct(self):
        cell = _dataset_prediction_cell({'status': 'failed', 'error': 'Connection refused'})
        self.assertIn('Execution failure', cell)
        self.assertIn('Connection refused', cell)

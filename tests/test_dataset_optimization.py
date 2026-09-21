import copy
import hashlib
import json
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from openpyxl import Workbook, load_workbook
from zemi import env
from zemi.component import ZemiComponent
from zemi.dataset import RunContext, resolve_adapter, table_dataset, table_evaluator
from zemi.params import ParamSampler, ParamSpace, SampleTrialResult, run_playbook_trial


class DatasetTests(unittest.TestCase):
    def setUp(self):
        env.path.tmp.mkdir(parents=True, exist_ok=True)
        self.root = env.path.tmp / f"dataset-test-{os.getpid()}-{id(self)}"
        self.root.mkdir()
        (self.root / '.zemicomp').touch()
        self.cwd = Path.cwd()
        os.chdir(self.root)
        env.path.comp._runid = None
        book = Workbook()
        book.active.title = 'Data'
        book.active.append(['City', 'Amount'])
        book.active.append(['A', 10])
        book.create_sheet('Empty')
        book.save('book.xlsx')
        book.close()
        Path('policy.md').write_text('Reviewed policy', encoding='utf-8')
        self.data = {'info': {'split': 'validation'}, 'annotation_policy': 'policy.md',
            'workbooks': [{'id': 'b', 'path': 'book.xlsx', 'sha256': hashlib.sha256(Path('book.xlsx').read_bytes()).hexdigest()}],
            'worksheets': [{'id': 's', 'workbook_id': 'b', 'name': 'Data', 'status': 'reviewed', 'tags': ['header']},
                           {'id': 'e', 'workbook_id': 'b', 'name': 'Empty', 'status': 'reviewed', 'tags': ['empty']}],
            'annotations': [{'id': 'a', 'workbook_id': 'b', 'worksheet_id': 's', 'range': 'A1:B2', 'status': 'reviewed'}]}
        self.write()

    def tearDown(self):
        os.chdir(self.cwd)
        env.path.comp._runid = None
        shutil.rmtree(self.root)

    def write(self):
        Path('data.json').write_text(json.dumps(self.data), encoding='utf-8')

    def load(self):
        return table_dataset(path='@comp/data.json', params={})

    def test_light_items_and_single_book_preflight(self):
        with patch('openpyxl.load_workbook', wraps=load_workbook) as load:
            items = self.load()
        self.assertEqual(load.call_count, 1)
        self.assertEqual(items[0]['ground_truth'], ['A1:B2'])
        self.assertEqual(items[1]['ground_truth'], [])
        json.dumps(items)
        self.assertEqual(set(items[0]['input']), {'workbook_path', 'worksheet_name'})

    def test_rejects_all_invalid_contracts(self):
        original = copy.deepcopy(self.data)
        changes = [
            ('worksheets', 0, 'status', 'draft'), ('worksheets', 1, 'status', 'blocked'),
            ('worksheets', 0, 'name', 'Missing'), ('worksheets', 0, 'workbook_id', 'absent'),
            ('worksheets', 1, 'id', 's'), ('annotations', 0, 'worksheet_id', 'absent'),
            ('annotations', 0, 'workbook_id', 'absent'), ('annotations', 0, 'range', 'B2:A1'),
            ('annotations', 0, 'status', 'draft'), ('workbooks', 0, 'path', 'missing.xlsx'),
            ('workbooks', 0, 'path', 'C:/unsafe.xlsx'), ('workbooks', 0, 'path', '../../../../outside.xlsx'),
            ('workbooks', 0, 'sha256', '0'*64),
        ]
        for section, index, key, value in changes:
            with self.subTest(section=section, key=key, value=value):
                self.data = copy.deepcopy(original)
                self.data[section][index][key] = value
                self.write()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.load()

    def test_exact_one_to_one_micro_metrics_errors_and_negatives(self):
        items = self.load()
        trial = SampleTrialResult(ParamSpace.from_params({}).start, [
            {'item': items[0], 'prediction': {'ranges': ['A1:B2', 'A1:B2', 'A1:B3']}},
            {'item': items[1], 'prediction': {'ranges': []}},
            {'item': items[0], 'prediction': None, 'error': 'model failed'},
            {'item': items[1], 'prediction': {'ranges': ['nonsense']}},
        ], {})
        metrics, feedback = table_evaluator(trial, params={})
        self.assertEqual((metrics['tp'], metrics['fp'], metrics['fn']), (1, 4, 1))
        self.assertAlmostEqual(metrics['f1'], 2/7)
        self.assertEqual((metrics['errors'], metrics['reviewed_empty'], metrics['correct_empty']), (2, 2, 1))
        self.assertEqual(len(feedback['items']), 4)
        self.assertEqual(feedback['tags']['header']['fn'], 1)

    def test_adapter_resolver_restricts_local_paths(self):
        Path('adapter.py').write_text('def evaluate(trial, *, params):\n    return {"score": 3}\n')
        self.assertEqual(resolve_adapter('evaluator', '@comp/adapter.py:evaluate')(None, params={}), {'score': 3})
        for name in ('os:system', '@inst/adapter.py:evaluate', '@comp/../adapter.py:evaluate', '@comp/adapter.py:missing'):
            with self.subTest(name=name), self.assertRaises((ValueError, FileNotFoundError)):
                resolve_adapter('evaluator', name)

    def test_cache_closes_before_opening_next_book(self):
        context = RunContext()
        with patch('openpyxl.load_workbook', wraps=load_workbook) as load:
            first = context.workbook(str(self.root / 'book.xlsx'))
            self.assertIs(first, context.workbook(str(self.root / 'book.xlsx')))
            with patch.object(first, 'close', wraps=first.close) as close:
                context.workbook('book.xlsx')
                close.assert_called_once()
            context.close()
        self.assertEqual(load.call_count, 2)

    def component(self):
        Path('one.ipynb').write_text(json.dumps({'cells': [], 'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5}))
        Path('params').mkdir()
        Path('params/test.toml').write_text('''
[system]
version = "0.3"
[component]
[[arsenals]]
id = "local"
config_path = "@comp/arsenal.toml"
lifecycle = "job"
[[playbooks]]
id = "detect"
path = "one.ipynb"
arsenal = "local"
param_space_mode = "sampler"
[playbooks.params]
x = { values = [0, 1], start = 0 }
[playbooks.sampler]
strategy = "block_coordinate"
max_samples = 2
blocks = [["x"]]
[playbooks.sampler.sample_trial.dataset]
adapter = "table_detection"
path = "@comp/data.json"
[playbooks.sampler.sample_trial.evaluator]
adapter = "table_detection"
[playbooks.sampler.sample_trial.objective]
metric = "f1"
direction = "maximize"
''', encoding='utf-8')
        return ZemiComponent('@comp/params/test.toml')

    def test_component_failfast_before_arsenal_and_notebook(self):
        component = self.component()
        self.data['worksheets'][1]['status'] = 'draft'
        self.write()
        with patch('zemi.arsenal.begin') as begin, patch('zemi.component.Playbook.run') as run:
            with self.assertRaisesRegex(ValueError, 'worksheets\\[e\\].status'):
                component.run()
            begin.assert_not_called()
            run.assert_not_called()
        component.close()

    def test_component_lifecycle_report_and_no_ground_truth_in_notebook(self):
        component = self.component()
        captured = []
        def notebook(playbook):
            captured.append(copy.deepcopy(playbook.params))
            entry = component.report.start_trial(playbook)
            name = playbook.params['dataset_input']['worksheet_name']
            if playbook.params['x'] == 0 and name == 'Data':
                error = RuntimeError('model unavailable')
                component.report.fail_playbook(entry, error)
                raise error
            entry['output_params'] = {'ranges': ['A1:B2'] if name == 'Data' else []}
            component.report.finish_playbook(entry)
        with patch('zemi.arsenal.ArsenalSession'), patch('zemi.arsenal.begin') as begin, patch('zemi.arsenal.end') as end, patch('zemi.component.Playbook.run', notebook):
            component.run()
            begin.assert_called_once()
            end.assert_called_once()
        component.close()
        report = json.loads(component.report.path.read_text(encoding='utf-8'))
        parent = report['job_trial']['playbook_trials'][0]
        self.assertEqual(parent['param_space_mode'], 'sampler')
        self.assertEqual(parent['sampler']['blocks'], [['x']])
        self.assertIn('ParamSpace mode: `sampler`', component.report.main_path.read_text(encoding='utf-8'))
        self.assertIn('"blocks": [', component.report.main_path.read_text(encoding='utf-8'))
        self.assertEqual(len(captured), 4)
        self.assertTrue(all(set(p['dataset_input']) == {'workbook_path', 'worksheet_name'} for p in captured))
        self.assertNotIn('ground_truth', json.dumps(captured))
        self.assertEqual(len(parent['samples']), 2)
        self.assertTrue(all(len(s['runs']) == 2 for s in parent['samples']))
        self.assertEqual(parent['best_sample'], 'detect-sample-0002')
        self.assertEqual(parent['samples'][0]['metrics']['fn'], 1)
        self.assertEqual(parent['samples'][1]['metrics']['f1'], 1)
        for filename in ('main.md', 'report.md'):
            content = (component.run_directory / filename).read_text(encoding='utf-8')
            self.assertIn('Ground truth', content)
            self.assertIn('model unavailable', content)

    def test_real_papermill_dataset_smoke_without_model(self):
        import nbformat
        component = self.component()
        library_parent = str(Path(__file__).resolve().parents[1])
        notebook = nbformat.v4.new_notebook()
        notebook.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
        notebook.cells = [
            nbformat.v4.new_code_cell('x = 0\ndataset_input = {}\narsenal_config_path = ""\narsenal_start_and_stop_at_job_level = True', metadata={'tags': ['parameters']}),
            nbformat.v4.new_code_cell(f'import sys\nsys.path.insert(0, {library_parent!r})\nfrom zemi.playbook import output_params\nassert set(dataset_input) == {{"workbook_path", "worksheet_name"}}\noutput_params({{"ranges": []}})'),
        ]
        nbformat.write(notebook, 'one.ipynb')
        # Freeze a valid sample using the public holdout API.
        component.close()
        env.path.comp._runid = None
        component = ZemiComponent('@comp/params/test.toml', sample_overrides={'detect': {'x': 0}})
        with patch('zemi.arsenal.ArsenalSession'), patch('zemi.arsenal.begin'), patch('zemi.arsenal.end'):
            try:
                component.run()
            finally:
                component.close()
        parent = component.report.data['job_trial']['playbook_trials'][0]
        self.assertEqual(len(parent['samples']), 1)
        self.assertEqual(len(parent['samples'][0]['runs']), 2)
        self.assertEqual(parent['samples'][0]['metrics']['fn'], 1)
        self.assertEqual(parent['samples'][0]['metrics']['correct_empty'], 1)
        for run in parent['samples'][0]['runs']:
            self.assertTrue((component.run_directory / run['artifacts']['output_notebook']).is_file())
            self.assertEqual(run['prediction'], {'ranges': []})

    def test_component_can_replay_best_sample_from_report(self):
        component = self.component()
        report_path = self.root / 'validation-report.json'
        report_path.write_text(json.dumps({'job_trial': {'playbook_trials': [{
            'playbook_id': 'detect', 'status': 'succeeded', 'best_sample': 'detect-sample-0002',
            'samples': [
                {'sample_trial_id': 'detect-sample-0001', 'params': {'x': 0}},
                {'sample_trial_id': 'detect-sample-0002', 'params': {'x': 1}},
            ],
        }]}}), encoding='utf-8')
        component.close()
        env.path.comp._runid = None
        replay = ZemiComponent.from_best_report('@comp/params/test.toml', report_path)
        try:
            self.assertEqual(replay.playbooks[0].params['x'], 1)
            self.assertEqual(replay.playbooks[0].sampler_config['max_samples'], 1)
        finally:
            replay.close()

    def test_best_report_must_resolve_a_successful_sample(self):
        self.component().close()
        for payload in (
            {},
            {'job_trial': {'playbook_trials': [{'playbook_id': 'detect', 'status': 'failed'}]}},
            {'job_trial': {'playbook_trials': [{'playbook_id': 'detect', 'status': 'succeeded',
                                               'best_sample': 'missing', 'samples': []}]}},
        ):
            with self.subTest(payload=payload):
                report_path = self.root / 'bad-report.json'
                report_path.write_text(json.dumps(payload), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'best sample report'):
                    ZemiComponent.from_best_report('@comp/params/test.toml', report_path)


class AdaptiveTests(unittest.TestCase):
    def test_coordinate_follows_improving_anchor_and_direction(self):
        for strategy, extra in [('coordinate', {}), ('block_coordinate', {'blocks': [['x'], ['y']]})]:
            for direction, sign in [('maximize', 1), ('minimize', -1)]:
                with self.subTest(strategy=strategy, direction=direction):
                    space = ParamSpace.from_params({key: {'values': [0, 1], 'start': 0} for key in ('x', 'y')})
                    sampler = ParamSampler(space, strategy, max_samples=4, objective_metric='score', direction=direction, **extra)
                    result = run_playbook_trial(sampler=sampler, dataset=[None], run=lambda s, i: None,
                        evaluator=lambda s, r: {'score': sign*(2*s.values['x']+s.values['y'])}, metric='score', direction=direction)
                    self.assertEqual(result.best('score', direction).sample.values, {'x': 1, 'y': 1})

    def test_block_coordinate_can_improve_jointly_with_unlisted_singletons(self):
        space = ParamSpace.from_params({
            'x': {'values': [0, 1], 'start': 0},
            'y': {'values': [0, 1], 'start': 0},
            'z': {'values': [0, 1], 'start': 0},
        })
        sampler = ParamSampler(
            space, 'block_coordinate', max_samples=6,
            blocks=[['x', 'y']], objective_metric='score', direction='maximize',
        )
        result = run_playbook_trial(
            sampler=sampler, dataset=[None], run=lambda sample, item: None,
            evaluator=lambda sample, runs: {
                'score': 10 * (sample.values['x'] == sample.values['y'] == 1) + sample.values['z']
            }, metric='score', direction='maximize',
        )
        self.assertEqual(result.best('score', 'maximize').sample.values, {'x': 1, 'y': 1, 'z': 1})

    def test_block_coordinate_order_restart_uniqueness_and_exhaustion(self):
        space = ParamSpace.from_params({
            'x': {'values': [0, 1], 'start': 0},
            'y': {'values': [0, 1], 'start': 0},
        })
        sampler = ParamSampler(
            space, 'block_coordinate', max_samples=10,
            blocks=[['x'], ['y']], objective_metric='score', direction='maximize',
        )
        scores = {(0, 0): 0, (1, 0): 0, (0, 1): 10, (1, 1): 11}
        result = run_playbook_trial(
            sampler=sampler, dataset=[None], run=lambda sample, item: None,
            evaluator=lambda sample, runs: {'score': scores[(sample.values['x'], sample.values['y'])]},
            metric='score', direction='maximize',
        )
        values = [item.sample.values for item in result.history]
        self.assertEqual(values, [
            {'x': 0, 'y': 0}, {'x': 1, 'y': 0},
            {'x': 0, 'y': 1}, {'x': 1, 'y': 1},
        ])
        self.assertEqual(len({item.sample.key() for item in result.history}), len(result.history))
        self.assertEqual(len(result.history), 4)  # finite space exhausted before max_samples

        limited = ParamSampler(space, 'block_coordinate', max_samples=3, blocks=[['x'], ['y']])
        limited_result = run_playbook_trial(
            sampler=limited, dataset=[None], run=lambda sample, item: None,
            evaluator=lambda sample, runs: {'score': 0}, metric='score', direction='maximize',
        )
        self.assertEqual(len(limited_result.history), 3)

    def test_block_coordinate_respects_declared_block_order(self):
        space = ParamSpace.from_params({
            'x': {'values': [0, 1], 'start': 0},
            'y': {'values': [0, 1], 'start': 0},
        })
        sampler = ParamSampler(space, 'block_coordinate', max_samples=2, blocks=[['y'], ['x']])
        result = run_playbook_trial(
            sampler=sampler, dataset=[None], run=lambda sample, item: None,
            evaluator=lambda sample, runs: {'score': 0}, metric='score', direction='maximize',
        )
        self.assertEqual(result.history[1].sample.values, {'x': 0, 'y': 1})

    def test_invalid_objective_is_failed_observation_and_run_failure_retained(self):
        sampler = ParamSampler(ParamSpace.from_params({'x': {'values': [0, 1], 'start': 0}}))
        def run(sample, item):
            raise RuntimeError('failure')
        result = run_playbook_trial(sampler=sampler, dataset=[1, 2], run=run,
            evaluator=lambda s, r: {'score': float('nan')}, metric='score', direction='maximize')
        self.assertEqual(len(result.history), 2)
        self.assertTrue(all(s.error and len(s.runs) == 2 for s in result.history))
        self.assertIsNone(result.best('score', 'maximize'))

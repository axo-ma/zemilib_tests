"""Deterministic report routing, layout and evaluation checks."""
import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from zemi.reporting import DefaultReportRenderer, ReportWriter, _cell, _replace_report
from zemi.dataset import table_evaluator
from zemi import env


class ReportingTests(unittest.TestCase):
    def test_combined_module_exports_and_best_score_ties(self):
        import json
        from types import SimpleNamespace
        from bs4 import BeautifulSoup
        w = self.writer
        w.register_module('m', optimized=True)
        self.assertEqual(w.register_dataset('m'), w.ref('module', 'm'))
        w.register_item('m', 'i')
        history = []
        for n, score in enumerate([0.4, 0.9, 0.9], 1):
            sid = f's{n}'
            w.register_sample('m', sid)
            history.append(SimpleNamespace(report_sample_id=sid, score=score, runs=[]))
        w.write_module_samples_summary('m', '## Samples\n\nTest samples')
        w.write_trial_dataset('m', self.renderer.render_trial_dataset(
            dataset=SimpleNamespace(items=[{'id': 'i', 'ground_truth': []}]),
            history=history, writer=w, module_id='m'))
        w.write_module_data('m', {'samples': [{'number': 1, 'score': 0.4}]})
        source = (w.root / 'm.md').read_text(encoding='utf-8')
        self.assertLess(source.index('## Samples'), source.index('## Items'))
        self.assertFalse((w.root / 'm.dataset.md').exists())
        self.assertFalse((w.root / 'm.dataset.cmd').exists())
        self.assertIn('%~dp0m.html', (w.root / 'm.cmd').read_text(encoding='utf-8'))
        markup = (w.root / 'm.html').read_text(encoding='utf-8')
        soup = BeautifulSoup(markup, 'html.parser')
        self.assertEqual(len(soup.select('table.dataset-items .best-sample')), 2)
        self.assertIn('0.900', soup.select('table.dataset-items th')[-1].get_text())
        self.assertIsNone(soup.find('script'))
        model = json.loads((w.root / 'm.json').read_text(encoding='utf-8'))
        self.assertEqual(model['samples'][0]['score'], 0.4)
        self.assertEqual(model['sections'][0]['name'], 'module_samples_summary')

    def test_notebook_retention_preserves_failures_and_rejects_outside_paths(self):
        from zemi.execution import retain_sample_notebooks
        runs, entries = [], []
        for n, status in enumerate(['failed', 'succeeded', 'succeeded', 'succeeded']):
            name = f'{n}.ipynb'
            (self.writer.root / name).write_text('{}', encoding='utf-8')
            runs.append({'status': status, 'artifacts': {'output_notebook': name}})
            entries.append({'output_notebook': name, 'output_path': name})
        runs[-1]['evaluation_error'] = 'bad response'
        retain_sample_notebooks(runs, entries, self.writer.root)
        self.assertTrue((self.writer.root / '0.ipynb').exists())
        self.assertTrue((self.writer.root / '1.ipynb').exists())
        self.assertFalse((self.writer.root / '2.ipynb').exists())
        self.assertTrue((self.writer.root / '3.ipynb').exists())
        self.assertNotIn('output_notebook', runs[2]['artifacts'])
        self.assertIsNone(entries[2]['output_notebook'])
        with self.assertRaises(ValueError):
            retain_sample_notebooks([{'status': 'succeeded', 'artifacts': {'output_notebook': '../outside.ipynb'}}], [], self.writer.root)

    def test_dataset_first_column_opens_workbook_second_opens_item_report(self):
        from types import SimpleNamespace
        from urllib.parse import quote
        w = self.writer
        w.register_module('m')
        w.register_dataset('m')
        w.register_item('m', 'sheet')
        file = w.root / 'Отчет с пробелом.xlsx'
        file.touch()
        path = '@inst/' + file.relative_to(env.path.inst).as_posix()
        item = {'id': 'sheet', 'input': {'workbook_path': path}, 'ground_truth': []}
        text = self.renderer.render_trial_dataset(dataset=SimpleNamespace(items=[item]),
            history=[], writer=w, module_id='m')
        self.assertIn('| # | Item ID | Target | Matches |', text)
        self.assertIn('| [1](dataset-items/m-sheet.md) |', text)
        w.register_item('m', 'second')
        w.register_item('m', 'sheet')
        self.assertEqual(w.item_number('m', 'sheet'), 1)
        self.assertEqual(w.item_number('m', 'second'), 2)
        self.assertIn('**Item #:** 1', self.renderer.render_run_report(
            run={'dataset_item_id': 'sheet'}, writer=w, module_id='m'))
        self.assertIn(f'| [sheet]({quote(file.name)}) | [] | [—](dataset-items/m-sheet.md) |', text)
        file.unlink()
        text = self.renderer.render_trial_dataset(dataset=SimpleNamespace(items=[item]),
            history=[], writer=w, module_id='m')
        self.assertIn('| sheet | [] | [—](dataset-items/m-sheet.md) |', text)

    def test_dataset_errors_display_raw_response_and_full_item_text(self):
        from types import SimpleNamespace
        w=self.writer
        w.register_module('m')
        w.register_dataset('m')
        w.register_sample('m','s')
        w.register_item('m','i')
        w.register_run('m','r')
        item={'id':'i','input':{},'ground_truth':[]}
        run={'dataset_item_id':'i','item':item,'run_id':'r','status':'succeeded',
             'prediction':{'raw_response':'{"unexpected":["A6"]}'},'evaluation_error':'invalid JSON contract'}
        trial=SimpleNamespace(sample=None,report_sample_id='s',runs=[run])
        dataset=SimpleNamespace(items=[item])
        def summary():
            return self.renderer.render_trial_dataset(dataset=dataset,history=[trial],writer=w,module_id='m')
        text=summary()
        self.assertIn('<summary>Error...</summary>',text)
        self.assertIn('invalid JSON contract',text)
        self.assertIn('Raw response: {&quot;unexpected&quot;:[&quot;A6&quot;]}',text)
        raw='```\n<answer>|x & y\n'+('long response '*20)
        run['prediction']['raw_response']=raw
        text=summary()
        self.assertIn('<summary>Error...</summary>',text)
        self.assertIn('long response '*20,text)
        self.assertIn('&lt;answer&gt;&#124;x &amp; y',text)
        self.assertNotIn('<answer>',text)
        full=self.renderer.render_worksheet_detection_report(dataset=dataset,item=item,history=[trial],writer=w,module_id='m')
        self.assertIn(raw,full)
        self.assertIn('## Raw responses for errors',full)
        run['prediction']=None
        self.assertIn('<summary>Error...</summary>',summary())
        self.assertNotIn('Raw response:',summary())
        run['prediction']={'answer':'wrong'}
        self.assertIn('&quot;answer&quot;: &quot;wrong&quot;',summary())

    def test_dataset_compact_predictions_expand_only_hidden_details(self):
        from zemi.reporting import _dataset_prediction_cell
        def render(value, **extra):
            return _cell(_dataset_prediction_cell({'comparison_prediction': value, **extra}))
        self.assertEqual(render(['A6:C14']), 'A6:C14')
        self.assertEqual(render([]), '[]')
        self.assertEqual(render(None), '—')
        self.assertEqual(render(['A6:C14'], status='succeeded', metrics={'exact_match': True}), '✅')
        self.assertEqual(render(None, status='failed'), 'Error')
        multi = render(['A6:C14', 'D1:E4'])
        self.assertIn('<summary>A6:C14...</summary>', multi)
        self.assertIn('D1:E4', multi)
        self.assertNotIn('href', multi)
        error = render(None, error='bad <script>|\nerror', prediction={'raw_response': '`x`'})
        self.assertIn('<summary>Error...</summary>', error)
        self.assertIn('&lt;script&gt;&#124;&#10;error', error)
        self.assertNotIn('\n', error)
        self.assertNotIn('|', error)
        self.assertIn('Raw response: `x`', error)

    def test_detail_navigation_has_one_parent_link(self):
        w = self.writer
        w.register_module('m', optimized=True)
        w.register_dataset('m')
        w.register_item('m','i')
        w.register_sample('m','s')
        w.register_run('m','r',sample_id='s')
        w.write_sample_trial('m','s','## Runs\n\n[Run r](../runs/m-run-r.md)')
        expected = [('sample','s','[Back to Module Report](../m.md)'),
                    ('module_runs',None,'[Back to Module Report](m.md)'),
                    ('run','r','[Back to Module Report](m.md)'),
                    ('item','i','[Back to Module Report](../m.md)')]
        for kind,identity,link in expected:
            text=(w.root / w.ref(kind,'m',identity).path).read_text(encoding='utf-8')
            self.assertEqual(text.splitlines()[2],link)
        self.assertIn('[Run r](../runs/m-run-r.md)',(w.root/w.ref('sample','m','s').path).read_text(encoding='utf-8'))
        w.register_module('single',optimized=False)
        ref=w.register_run('single','only')
        self.assertTrue((w.root / w.ref('module_runs','single').path).is_file())
        self.assertEqual((w.root/ref.path).read_text(encoding='utf-8').splitlines()[2],
                         '[Back to Module Report](single.md)')

    def test_sample_duration_sums_lm_times_and_keeps_missing_values(self):
        w=self.writer
        w.register_module('m',optimized=True)
        w.register_sample('m','s')
        sample={'id':'s','duration':'2m 15s','runs':[
            {'prediction':{'lm_time':65,'item_tokens':10,'prompt_tokens':100}},
            {'prediction':{'lm_time':45,'item_tokens':20,'prompt_tokens':200}},
            {'prediction':None}]}
        text=self.renderer.render_module_samples_summary(samples=[sample],param_names=[],writer=w,module_id='m')
        self.assertIn('Duration<br>(module / LM)',text)
        self.assertIn('15.000 / 150.000 | 2m 15s / 1m 50s',text)
        sample['runs']=[{'prediction':{'lm_time':0}},{'prediction':{'lm_time':True}}]
        text=self.renderer.render_module_samples_summary(samples=[sample],param_names=[],writer=w,module_id='m')
        self.assertIn('— / — | 2m 15s / 0m 00s',text)
        sample['runs']=[]
        text=self.renderer.render_module_samples_summary(samples=[sample],param_names=[],writer=w,module_id='m')
        self.assertIn('— / — | 2m 15s / —',text)

    def test_generic_predictions_metrics_and_standalone_dataset_reports(self):
        from types import SimpleNamespace
        from zemi.dataset import TrialDataset
        item = {'id': 'classification', 'input': {'text': 'hello'}, 'ground_truth': 'positive'}
        run = {'dataset_item_id': item['id'], 'item': item, 'run_id': 'r', 'status': 'succeeded',
               'prediction': 'negative', 'metrics': {'accuracy': 0.0, 'confidence': 0.6}}
        w = self.writer
        w.register_module('m')
        w.register_dataset('m')
        w.register_item('m',item['id'])
        w.register_sample('m','s')
        w.register_run('m','r')
        trial = SimpleNamespace(sample=SimpleNamespace(values={}), report_sample_id='s', runs=[run])
        dataset = TrialDataset([item])
        text = self.renderer.render_trial_dataset(dataset=dataset,history=[trial],writer=w,module_id='m')
        self.assertIn('| classification | positive | [—](dataset-items/m-classification.md) | negative |',text)
        detail = self.renderer.render_worksheet_detection_report(dataset=dataset,item=item,history=[trial],writer=w,module_id='m')
        self.assertIn('accuracy / confidence',detail)
        self.assertIn('0.000 / 0.600',detail)
        self.assertNotIn('ranges',detail)
        self.assertNotIn('Worksheet',detail)
        standalone, details = dataset.render_report(history=[trial])
        self.assertIn('Matches',standalone)
        self.assertIn('negative',standalone)
        self.assertIn('accuracy / confidence',next(iter(details.values())))
        self.assertNotIn('Avg F1',standalone)

    def test_sample_prompt_order_feedback_hidden_and_token_means(self):
        from types import SimpleNamespace
        from zemi.reporting import JobReporting
        params = {'encoding_prompt': {'prompt_name':'chosen'}}
        run = {'item':{'id':'i','ground_truth':42},'dataset_item_id':'i','run_id':'r',
               'status':'succeeded','comparison_prediction':42,
               'prediction':{'answer':42,'item_tokens':10,'prompt_tokens':100},
               'metrics':{'exact_match':True}}
        trial = SimpleNamespace(param_sample=SimpleNamespace(values=params),_report_prompt='Captured {{item}}')
        text = self.renderer.render_sample_trial(sample_trial=trial,runs=[run],metrics={'score':1},score=1,feedback={'secret_feedback':True})
        self.assertLess(text.index('## Parameters'),text.index('## Evaluation'))
        self.assertLess(text.index('## Evaluation'),text.index('## Prompt'))
        self.assertLess(text.index('## Prompt'),text.index('## Runs'))
        self.assertNotIn('Feedback',text)
        self.assertNotIn('secret_feedback',text)
        self.assertIn('Captured {{item}}',text)
        self.assertIn('✅',text)
        w=self.writer
        w.register_module('m')
        w.register_sample('m','s')
        sample={'id':'s','params':params,'runs':[run,dict(run,prediction={'item_tokens':20})]}
        summary=self.renderer.render_module_samples_summary(samples=[sample],param_names=[],writer=w,module_id='m')
        self.assertIn('Mean Tokens<br>(item / prompt)',summary)
        self.assertIn('15.000 / 100.000',summary)
        self.assertEqual(JobReporting._duration({'duration_seconds':135}), '2m 15s')
        self.assertEqual(JobReporting._duration({'duration_seconds':4080}), '1h 08m')
        self.assertEqual(run['prediction']['item_tokens'],10)

    def test_variable_configuration_has_no_concrete_value(self):
        from zemi.params import ParamSpace
        space=ParamSpace(config={'choice':{'values':['first','last'],'start':'first'},'fixed':'retained'})
        text=self.renderer.render_module_optimization_config(config={},space=space)
        self.assertIn('| choice | Variable | — |',text)
        self.assertIn('| fixed | Fixed | retained |',text)
        self.assertNotIn('first',text)

    def test_result_tables_shorten_prompt_binding_without_changing_configuration(self):
        from types import SimpleNamespace
        binding = {'prompt_name': 'cell_all_md', 'prompt_file': '@comp/prompts.md',
                   'encoder': '@comp/encoder.py:encode', 'encoding_format': 'cell_all'}
        params = {'encoding_prompt': binding, 'temperature': 0.0}
        sample = {'id': 's', 'params': params, 'runs': []}
        w = self.writer
        w.register_module('m')
        w.register_sample('m', 's')
        w.register_dataset('m')
        texts = [self.renderer.render_module_samples_summary(samples=[sample],
                     param_names=list(params), writer=w, module_id='m'),
                 self.renderer.render_module_selected_sample(selected='s', samples=[sample],
                     mode='optimize', writer=w, module_id='m'),
                 self.renderer.render_trial_dataset(dataset=SimpleNamespace(items=[]),
                     history=[SimpleNamespace(sample=SimpleNamespace(values=params),
                         report_sample_id='s', runs=[])], writer=w, module_id='m')]
        for text in texts[:2]:
            self.assertIn('cell_all_md', text)
        self.assertIn('[Sample 1](samples/m-sample-s.md)', texts[2])
        self.assertNotIn('cell_all_md', texts[2])
        for text in texts:
            self.assertNotIn('@comp/prompts.md', text)
            self.assertNotIn('@comp/encoder.py', text)
        self.assertIn('@comp/prompts.md', self.renderer.render_module_parameters(params=params))
        self.assertEqual(params['encoding_prompt'], binding)

    def test_dataset_exact_matches_are_checkmarks_including_empty_targets(self):
        from types import SimpleNamespace
        w = self.writer
        w.register_module('m')
        w.register_dataset('m')
        w.register_sample('m', 'm-sample-0001')
        items = [{'id': 'table', 'ground_truth': ['A1:B3']}, {'id': 'empty', 'ground_truth': []}]
        for item in items:
            w.register_item('m', item['id'])
        runs = [{'dataset_item_id': item['id'], 'status': 'succeeded',
                 'prediction': {'ranges': item['ground_truth']}, 'metrics': {'exact_match': True}}
                for item in items]
        trial = SimpleNamespace(sample=None, report_sample_id='m-sample-0001', runs=runs)
        text = self.renderer.render_trial_dataset(dataset=SimpleNamespace(items=items),
            history=[trial], writer=w, module_id='m')
        self.assertIn('| ["A1:B3"] | [1 / 1](dataset-items/m-table.md) | ✅ |', text)
        self.assertIn('| [] | [1 / 1](dataset-items/m-empty.md) | ✅ |', text)
        runs[0]['evaluation_error'] = 'bad range'
        text = self.renderer.render_trial_dataset(dataset=SimpleNamespace(items=items),
            history=[trial], writer=w, module_id='m')
        self.assertIn('<summary>Error...</summary>', text)

    def test_execution_filenames_do_not_repeat_module_and_kind(self):
        w = self.writer
        w.register_module('m')
        self.assertEqual(w.register_sample('m', 'm-sample-0001').path, 'samples/m-sample-0001.md')
        self.assertEqual(w.register_run('m', 'm-run-000001').path, 'm.runs.md')
        self.assertEqual(w.register_sample('m', 'custom').path, 'samples/m-sample-custom.md')

    def test_dataset_sample_columns_show_expected_and_actual_ranges(self):
        from types import SimpleNamespace
        writer = self.writer
        writer.register_module("m")
        writer.register_dataset("m")
        writer.register_item("m", "sheet")
        trials = []
        for sid, prediction in (("s1", {"ranges": []}), ("s2", None),
                                ("s3", {"ranges": [f"A{i}:B{i}" for i in range(1, 20)]})):
            writer.register_sample("m", sid)
            trials.append(SimpleNamespace(report_sample_id=sid,
                sample=SimpleNamespace(values={"encoding_format": sid}),
                runs=[{"dataset_item_id": "sheet", "prediction": prediction,
                       "comparison_prediction": prediction.get("ranges") if prediction else None}]))
        dataset = SimpleNamespace(items=[{"id": "sheet", "ground_truth": ["A1:B3"]}])
        text = self.renderer.render_trial_dataset(dataset=dataset, history=trials, writer=writer, module_id="m")
        self.assertIn("[Sample 1](samples/m-sample-s1.md)", text)
        self.assertIn("[Sample 2](samples/m-sample-s2.md)", text)
        self.assertNotIn("Sample 1 (", text)
        self.assertNotIn("Sample 2 (", text)
        self.assertIn("| Item ID | Target | Matches |", text)
        self.assertIn('| ["A1:B3"] |', text)
        self.assertIn('| [] | — |', text)
        self.assertNotIn('["A1:B3"] /', text)
        self.assertIn('<summary>A1:B1...</summary>', text)
        self.assertIn('A19:B19', text)
        full = self.renderer.render_worksheet_detection_report(dataset=dataset,
            item=dataset.items[0], history=trials, writer=writer, module_id="m")
        self.assertIn("| Prediction |", full)
        self.assertNotIn("Detected ranges", full)
        self.assertIn("A19:B19", full)

    def test_report_floats_show_three_decimal_places_without_changing_values(self):
        metrics = {"f1": 0.6666666667, "count": 24}
        self.assertEqual(_cell(metrics), '{"count": 24, "f1": 0.667}')
        self.assertEqual(_cell(metrics["f1"]), "0.667")
        self.assertEqual(_cell(1.0), "1.000")
        self.assertEqual(metrics["f1"], 0.6666666667)

    def setUp(self):
        env.path.tmp.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=env.path.tmp)
        self.addCleanup(self.tmp.cleanup)
        self.writer = ReportWriter(Path(self.tmp.name) / "run260924-120000")
        self.renderer = DefaultReportRenderer()

    def test_fragment_order_replacement_and_atomic_file(self):
        writer = self.writer
        ref = writer.register_module("first", filename="one.ipynb")
        writer.write_module_errors("first", "## Errors\n\nFirst error")
        writer.write_module_results("first", "## Results\n\nOld")
        writer.write_module_results("first", "## Results\n\nNew")
        doc = (writer.root / ref.path).read_text(encoding="utf-8")
        self.assertEqual(doc.count("## Results"), 1)
        self.assertNotIn("Old", doc)
        self.assertLess(doc.index("## Results"), doc.index("## Errors"))
        self.assertFalse((writer.root / f".{ref.path}.tmp").exists())

    def test_unchanged_fragment_does_not_replace_file_again(self):
        writer = self.writer
        writer.register_module("m")
        writer.write_module_header("m", "Header")
        with patch("zemi.reporting.os.replace", wraps=os.replace) as replace:
            writer.write_module_header("m", "Header")
        replace.assert_not_called()

    def test_transient_windows_lock_retries_atomic_replace(self):
        target = Path(self.tmp.name) / "report.md"
        tmp = Path(self.tmp.name) / ".report.md.tmp"
        target.write_text("old", encoding="utf-8")
        tmp.write_text("new", encoding="utf-8")
        calls = 0
        def transient(source, destination):
            nonlocal calls
            calls += 1
            if calls <= 2:
                error = PermissionError(13, "File is temporarily locked")
                error.winerror = 5
                raise error
            os_replace(source, destination)
        os_replace = os.replace
        with patch("zemi.reporting.os.replace", side_effect=transient), patch("zemi.reporting.time.sleep"):
            _replace_report(tmp, target)
        self.assertEqual(calls, 3)
        self.assertEqual(target.read_text(encoding="utf-8"), "new")
        self.assertFalse(tmp.exists())

    def test_persistent_windows_lock_keeps_previous_report_intact(self):
        target = Path(self.tmp.name) / "report.md"
        tmp = Path(self.tmp.name) / ".report.md.tmp"
        target.write_text("old", encoding="utf-8")
        tmp.write_text("new", encoding="utf-8")
        error = PermissionError(13, "File is locked")
        error.winerror = 32
        with patch("zemi.reporting.os.replace", side_effect=error), patch("zemi.reporting.time.sleep"):
            with self.assertRaises(PermissionError):
                _replace_report(tmp, target)
        self.assertEqual(target.read_text(encoding="utf-8"), "old")

    def test_safe_colliding_names_and_registered_relative_links(self):
        writer = self.writer
        first = writer.register_module("index", filename="a.ipynb")
        second = writer.register_module("index!", filename="b.ipynb")
        self.assertNotEqual(first.path.casefold(), second.path.casefold())
        self.assertNotEqual(first.path, "index.md")
        for ref in (first, second):
            self.assertTrue((writer.root / ref.path).is_file())
            self.assertNotIn("..", ref.path)
        sample = writer.register_sample("index", "../same")
        run = writer.register_run("index", "../same", sample_id="../same")
        self.assertTrue((writer.root / sample.path).is_file())
        self.assertTrue((writer.root / run.path).is_file())
        self.assertTrue(writer.href(sample, first).startswith("../"))
        self.assertIsNone(writer.artifact_href(first, "missing.ipynb"))

    def test_dynamic_parameters_metrics_and_run_anchor(self):
        writer = self.writer
        writer.register_module("detect", filename="detect.ipynb", optimized=True)
        writer.register_sample("detect", "s-1")
        writer.register_sample("detect", "s-2")
        writer.register_run("detect", "r-1", sample_id="s-1")
        samples = [
            {"id": "s-1", "params": {"temperature": 0}, "metrics": {"accuracy": .5},
             "score": .5, "status": "succeeded", "runs": [{"run_id": "r-1", "status": "succeeded"}]},
            {"id": "s-2", "params": {"temperature": 1}, "metrics": {"coverage": .8},
             "score": .8, "status": "failed", "runs": []},
        ]
        fragment = self.renderer.render_module_samples_summary(samples=samples,
            param_names=["temperature"], writer=writer, module_id="detect")
        self.assertIn("Parameters<br>temperature", fragment)
        self.assertIn("Metrics<br>accuracy / coverage", fragment)
        self.assertIn("0.500 / —", fragment)
        self.assertIn("— / 0.800", fragment)
        runs = self.renderer.render_module_runs_summary(samples=samples, writer=writer, module_id="detect")
        self.assertIn("### [Sample 1]", runs)
        self.assertIn("### [Sample 2]", runs)
        self.assertIn("#sample-1", fragment)
        self.assertIn("#sample-2", fragment)

    def test_runs_report_shows_selected_outputs_and_sample_parameters(self):
        writer = self.writer
        writer.register_module("detect", optimized=True)
        writer.register_sample("detect", "s-1")
        writer.register_run("detect", "r-1", sample_id="s-1")
        samples = [{"id": "s-1", "params": {"encoding_format": "cell_all"},
                    "runs": [{"run_id": "r-1", "status": "succeeded", "duration": "5.0 s",
                              "prediction": {"ranges": ["A1:B2"], "raw_response": "large",
                                             "lm_time": 3.2, "item_tokens": 20},
                              "report_output_keys": ["ranges", "lm_time", "item_tokens"]}]}]
        report = self.renderer.render_module_runs_summary(samples=samples, writer=writer, module_id="detect")
        self.assertIn("[Sample 1](samples/", report)
        self.assertIn("**Parameters:** encoding_format = cell_all", report)
        self.assertIn("Outputs<br>ranges / LM Time / Item Tokens", report)
        self.assertIn('["A1:B2"] / 3.200 / 20', report)
        self.assertNotIn("raw_response", report)
        self.assertNotIn("| Run | Sample |", report)

    def test_secret_redaction_and_custom_fragment(self):
        writer = self.writer
        writer.add_secrets(["token-123"])
        ref = writer.register_module("m")
        writer.write_module_results("m", "## Custom\n\nresult token-123")
        doc = (writer.root / ref.path).read_text(encoding="utf-8")
        self.assertIn("## Custom", doc)
        self.assertIn("***", doc)
        self.assertNotIn("token-123", doc)

    def test_running_run_is_replaced_by_error_and_linked_from_sample(self):
        writer = self.writer
        writer.register_module("m", optimized=True)
        sample = writer.register_sample("m", "s")
        run = writer.register_run("m", "r", sample_id="s")
        writer.write_run_report("m", "r", "**Status:** running", sample_id="s")
        writer.write_run_report("m", "r", "**Status:** failed\n\n## Errors\n\nmodel unavailable", sample_id="s")
        run_text = (writer.root / run.path).read_text(encoding="utf-8")
        sample_text = (writer.root / sample.path).read_text(encoding="utf-8")
        self.assertNotIn("**Status:** running", run_text)
        self.assertFalse((writer.root / 'runs').exists())
        self.assertIn("[Back to Module Report](m.md)", run_text)
        self.assertEqual(sample_text.splitlines()[2], "[Back to Module Report](../m.md)")

    def test_set_evaluation_permutation_duplicates_and_extra(self):
        item = {"id": "book::sheet", "input": {}, "ground_truth": ["A1:B2", "D1:E5"]}
        runs = [{"item": item, "prediction": {"ranges": ["D1:E5", "A1:B2", "A1:B2"]}},
                {"item": item, "prediction": {"ranges": ["A1:B2", "D1:E5", "F1:G2", "F1:G2"]}}]
        metrics, details = table_evaluator(type("Trial", (), {"runs": runs})(), params={})
        self.assertEqual((runs[0]["metrics"]["tp"], runs[0]["metrics"]["fp"], runs[0]["metrics"]["fn"]), (2, 0, 0))
        self.assertTrue(runs[0]["metrics"]["exact_match"])
        self.assertEqual((runs[1]["metrics"]["tp"], runs[1]["metrics"]["fp"]), (2, 1))
        self.assertFalse(runs[1]["metrics"]["exact_match"])
        self.assertEqual(metrics["tp"], 4)

    def test_mixed_job_single_and_optimizer_modes(self):
        writer = self.writer
        writer.register_module("single", filename="single.ipynb")
        writer.register_module("start", filename="start.ipynb", optimized=True)
        writer.register_module("search", filename="search.ipynb", optimized=True)
        modules = [
            {"id": "single", "name": "single.ipynb", "optimized": False, "status": "succeeded", "outputs": {}},
            {"id": "start", "name": "start.ipynb", "optimized": True, "mode": "start_only", "status": "succeeded", "samples": [], "runs": []},
            {"id": "search", "name": "search.ipynb", "optimized": True, "mode": "optimize", "status": "running", "samples": [], "runs": []},
        ]
        text = self.renderer.render_module_summary(modules=modules, writer=writer)
        self.assertIn("### Single Execution", text)
        self.assertIn("### Execution With Optimizer", text)
        self.assertIn("`start_only`", text)
        self.assertIn("`optimize`", text)
        self.assertIn("Samples<br>(OK / Total)", text)
        self.assertIn("Runs<br>(OK / Total)", text)


if __name__ == "__main__":
    unittest.main()

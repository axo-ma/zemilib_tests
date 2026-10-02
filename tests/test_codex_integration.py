import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
from zemi import env
from zemi.coding_agents.codex import server


class CodexIntegrationTests(unittest.TestCase):
    def setUp(self):
        env.path.tmp.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=env.path.tmp)
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.config_path=self.root/'settings.json'
        self.config_path.write_text(json.dumps({'component':str(self.root)}),encoding='utf-8')
        (self.root/'module.md').write_text(
            '[Sample](samples/one.md)\n\n| # | Item ID | Target | Matches | Sample 1 |\n'
            '|---|---|---|---|---|\n| 1 | [item](book.xlsx) | [] | 1 / 1 | ✅ |\n',encoding='utf-8')
        (self.root/'samples').mkdir()
        (self.root/'samples/one.md').write_text('[Back](../module.md#items)',encoding='utf-8')
        (self.root/'book.xlsx').write_bytes(b'test')
        (self.root/'module.chat.json').write_text(json.dumps({'rows':[{'item_id':'item','cells':[{'contexts':[
            {'source':{'component_root':str(self.root)},'request':{'messages':[{'role':'user','content':'worksheet'}]},
             'assistant':{'role':'assistant','content':'{"ranges":[]}'}}]}]}]}),encoding='utf-8')

    def action(self,href,report=None):
        return server.dispatch('tools/call',{'name':'zemi_report_action','arguments':{
            'report':str(report or self.root/'module.html'),'href':href}},self.config_path)['structuredContent']

    def test_real_report_resource_and_html_navigation(self):
        shown=server.dispatch('tools/call',{'name':'zemi_report_show','arguments':{'report':'module.html'}},self.config_path)
        self.assertTrue(shown['structuredContent']['ready'])
        resource=server.dispatch('resources/read',{'uri':'ui://zemi/report.html'},self.config_path)
        markup=resource['contents'][0]['text']
        self.assertIn('samples/one.html',markup)
        self.assertIn('zemi-chat:0:0',markup)
        self.assertIn('zemi_report_action',markup)
        sample=self.action('samples/one.html')
        self.assertIn('../module.html#items',sample['html'])
        back=self.action('../module.html#items',self.root/'samples/one.html')
        self.assertEqual(back['fragment'],'items')

    def test_excel_and_chat_actions_use_report_links(self):
        with patch('os.startfile',create=True) as opening:
            self.assertTrue(self.action('book.xlsx')['open_requested'])
            opening.assert_called_once_with(str(self.root/'book.xlsx'))
        with patch('subprocess.Popen') as launch:
            launch.return_value.pid=456
            result=self.action('zemi-chat:0:0')
            self.assertEqual(result['pid'],456)
            self.assertEqual(launch.call_args.args[0][-2:],['0','0'])

    def test_unlisted_links_and_outside_reports_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'not a link'):
            self.action('../other.xlsx')
        with self.assertRaisesRegex(ValueError,'inside'):
            self.action('book.xlsx',self.root.parent/'other.html')

    def config(self, settings):
        # Dispatch reads the configuration anew; no real file or process actions.
        return patch.object(Path, 'read_text', return_value=json.dumps(settings))

    def test_settings_are_reloaded_and_launch_uses_selected_cell(self):
        settings = {'component': 'component', 'manifest': 'manifest.json',
                    'excel': 'book.xlsx', 'item': 2, 'sample': 1}
        manifest = {'rows': [{'cells': [{}]}, {'cells': [{}]}]}
        original = Path.read_text
        def read(path, *args, **kwargs):
            if path.name == 'settings.json':
                return json.dumps(settings)
            if path.name == 'manifest.json':
                return json.dumps(manifest)
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', read), patch.object(Path, 'is_file', return_value=True), \
             patch('subprocess.Popen') as launch:
            launch.return_value.pid = 123
            result = server.dispatch('tools/call', {'name': 'zemi_probe_open_chat'}, 'settings.json')
            self.assertEqual(result['structuredContent']['item'], 2)
            self.assertEqual(launch.call_args.args[0][-2:], ['1', '0'])
            settings['item'] = 1
            server.dispatch('tools/call', {'name': 'zemi_probe_open_chat'}, 'settings.json')
            self.assertEqual(launch.call_args.args[0][-2:], ['0', '0'])

    def test_no_arbitrary_launch_arguments(self):
        with self.config({'component': 'component'}):
            with self.assertRaises(ValueError):
                server.dispatch('tools/call', {'name': 'zemi_probe_open_chat',
                                              'arguments': {'command': 'other'}}, 'settings.json')


if __name__ == '__main__':
    unittest.main()

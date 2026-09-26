import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zemi import env
from zemi.params import ParamSpace
from zemi.prompting import load_prompts, build_prompt, sample_names


class PromptingTests(unittest.TestCase):
    def setUp(self):
        env.path.tmp.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=env.path.tmp)
        self.root = Path(self.temp.name)
        self.paths = patch('zemi.prompting.zemi_path', side_effect=self.path)
        self.paths.start()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.paths.stop)
        self.file = self.root / 'prompts.md'
        self.file.write_text('# first\n\nExample:\n```text\n# literal\n```\n## Input\n{{item}}\n\n# second\n{{item}}\n', encoding='utf-8')
        (self.root / 'encoder.py').write_text('def encode(path, sheet, *, format):\n    return f"{path.name}:{sheet}:{format}"\n', encoding='utf-8')

    def path(self, ref):
        return self.root / str(ref).removeprefix('@comp/')

    def binding(self, name='first'):
        return dict(prompt_name=name, prompt_file='@comp/prompts.md',
                    encoder='@comp/encoder.py:encode', encoding_format='custom')

    def test_fenced_headings_preserved_and_bad_templates_rejected(self):
        prompts = load_prompts('@comp/prompts.md')
        self.assertEqual(list(prompts), ['first', 'second'])
        self.assertIn('# literal', prompts['first'])
        for text in ('# first\n{{item}}\n# first\n{{item}}', '# first\nmissing',
                     '# first\n{{item}}{{item}}', '# first\n```\n{{item}}'):
            self.file.write_text(text, encoding='utf-8')
            with self.assertRaises(ValueError):
                load_prompts('@comp/prompts.md')

    def test_encoder_receives_file_sheet_and_format(self):
        with patch('zemi.dataset.zemi_path', side_effect=self.path):
            item, prompt = build_prompt(self.binding(), '@comp/book.xlsx', 'Лист1')
        self.assertEqual(item, 'book.xlsx:Лист1:custom')
        self.assertIn(item, prompt)
        self.assertNotIn('{{item}}', prompt)
        with self.assertRaises(ValueError):
            build_prompt(self.binding('missing'), '@comp/book.xlsx', 'sheet')

    def test_paired_values_and_names_use_configured_grid_not_execution_order(self):
        bindings = [self.binding(), self.binding('second')]
        space = ParamSpace(config={'encoding_prompt': {'values': bindings, 'start': bindings[0]},
                                  'temperature': {'values': [0, 1], 'start': 0}})
        names = sample_names(space)
        self.assertEqual(list(names.values()), ['first-001', 'first-002', 'second-001', 'second-002'])
        self.assertEqual(len(space.grid()), 4)
        self.assertEqual(sample_names(space), names)

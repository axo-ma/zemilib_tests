from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path

from zemi import env
from zemi.component import ZemiComponent
from zemi.params import ParamSampler, ParamSpace, run_playbook_trial, validate_document


def document() -> dict:
    return {
        "system": {"version": "0.3", "params": {"locale": "en"}},
        "component": {"name": "params-03", "stop_on_error": True, "params": {}},
        "arsenals": [
            {"id": "local", "config_path": "@comp/zemi/example.toml", "lifecycle": "job", "params": {"device": "cpu"}},
            {"id": "remote", "lifecycle": "external", "params": {}},
        ],
        "playbooks": [
            {"id": "one", "path": "one.ipynb", "arsenal": "local", "params": {"temperature": {"values": [0.0, 0.2], "start": 0.2}}},
            {"id": "two", "path": "two.ipynb", "arsenal": "remote", "params": {}},
        ],
    }


class SchemaTests(unittest.TestCase):
    def test_complete_document_is_valid_and_copied(self) -> None:
        source = document()
        validated = validate_document(source)
        self.assertEqual(validated["system"]["version"], "0.3")
        self.assertIsNot(validated, source)

    def test_structural_sections_are_closed(self) -> None:
        source = document(); source["component"]["user_value"] = 1
        with self.assertRaisesRegex(ValueError, "unsupported structural keys: user_value"):
            validate_document(source)

    def test_parent_arsenal_must_exist_and_ids_are_unique(self) -> None:
        source = document(); source["playbooks"][0]["arsenal"] = "missing"
        with self.assertRaisesRegex(ValueError, "missing Arsenal"):
            validate_document(source)
        source = document(); source["arsenals"][1]["id"] = "local"
        with self.assertRaisesRegex(ValueError, "duplicate Arsenal id"):
            validate_document(source)

    def test_sampler_contract_is_validated(self) -> None:
        source = document()
        source["playbooks"][0]["sampler"] = {
            "strategy": "random", "max_samples": 2,
            "sample_trial": {
                "dataset": {"adapter": "jsonl", "path": "@comp/data/eval.jsonl"},
                "evaluator": {"adapter": "@comp/evaluate.py:evaluate"},
                "objective": {"metric": "score", "direction": "maximize"},
            },
        }
        validated = validate_document(source)
        self.assertEqual(validated["playbooks"][0]["sampler"]["strategy"], "random")


class ParamSpaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.space = ParamSpace.from_params({
            "fixed": "x",
            "temperature": {"values": [0.0, 0.2, 0.5], "start": 0.2},
            "seed": {"range": {"min": 1, "max": 2, "step": 1}, "start": 1},
        })

    def test_start_is_first_and_grid_is_unique(self) -> None:
        grid = self.space.grid()
        self.assertEqual(grid[0].values, {"fixed": "x", "temperature": 0.2, "seed": 1})
        self.assertEqual(len(grid), 6)
        self.assertEqual(len({sample.key() for sample in grid}), 6)

    def test_all_minimum_strategies_propose_without_duplicates(self) -> None:
        for strategy, options in (
            ("grid", {}), ("random", {"max_samples": 4, "seed": 7}),
            ("coordinate", {"max_samples": 4}),
            ("block_coordinate", {"max_samples": 4, "block_size": 2}),
        ):
            with self.subTest(strategy=strategy):
                sampler = ParamSampler(self.space, strategy, **options)
                self.assertEqual(sampler.candidates[0], self.space.start)
                self.assertEqual(len({sample.key() for sample in sampler.candidates}), len(sampler.candidates))

    def test_evaluator_observes_complete_sample_trial(self) -> None:
        events = []
        sampler = ParamSampler(self.space, "random", max_samples=2, seed=1, objective_metric="score")

        def run(sample, item):
            events.append(("run", sample.key(), item)); return item * 2

        def evaluate(sample, runs):
            events.append(("evaluate", sample.key(), tuple(runs))); return {"score": sum(runs)}, {"count": len(runs)}

        result = run_playbook_trial(sampler=sampler, dataset=[1, 2, 3], run=run, evaluator=evaluate, metric="score", direction="maximize")
        self.assertEqual(len(result.history), 2)
        self.assertTrue(all(len(item.runs) == 3 for item in result.history))
        self.assertEqual([event[0] for event in events], ["run", "run", "run", "evaluate"] * 2)


class CanonicalComponentTests(unittest.TestCase):
    def setUp(self) -> None:
        env.path.tmp.mkdir(parents=True, exist_ok=True)
        self.root = env.path.tmp / f"params-03-{os.getpid()}-{id(self)}"
        self.root.mkdir(); (self.root / ".zemicomp").write_text("", encoding="utf-8")
        (self.root / "params").mkdir()
        for name in ("one.ipynb", "two.ipynb"):
            (self.root / name).write_text(json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}), encoding="utf-8")
        self.cwd = Path.cwd(); os.chdir(self.root); env.path.comp._runid = None

    def tearDown(self) -> None:
        os.chdir(self.cwd); env.path.comp._runid = None; shutil.rmtree(self.root, ignore_errors=True)

    def test_canonical_document_resolves_named_arsenal_refs(self) -> None:
        content = '''
[system]
version = "0.3"
[system.params]
locale = "en"
[component]
name = "canonical"
stop_on_error = true
[component.params]
[[arsenals]]
id = "local"
config_path = "@comp/zemi/example.toml"
lifecycle = "job"
[arsenals.params]
device = "cpu"
[[playbooks]]
id = "one"
path = "one.ipynb"
arsenal = "local"
[playbooks.params]
locale = { ref = "system.params.locale" }
device = { ref = "arsenals.local.params.device" }
temperature = { values = [0.0, 0.2], start = 0.2 }
'''
        (self.root / "params" / "default_params.toml").write_text(content, encoding="utf-8")
        component = ZemiComponent()
        self.assertTrue(component.params_03)
        self.assertEqual(component.name, "canonical")
        self.assertEqual([p.params["temperature"] for p in component.playbooks], [0.2])
        self.assertEqual(component.playbooks[0].params["device"], "cpu")
        component.close()

    def test_canonical_refs_cannot_read_structural_fields(self) -> None:
        content = '''
[system]
version = "0.3"
[component]
[component.params]
[[arsenals]]
id = "local"
lifecycle = "external"
[[playbooks]]
id = "one"
path = "one.ipynb"
arsenal = "local"
[playbooks.params]
bad = { ref = "component.stop_on_error" }
'''
        (self.root / "params" / "default_params.toml").write_text(content, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "refs may target only params sections"):
            ZemiComponent()


if __name__ == "__main__":
    unittest.main()

import copy
import math
import unittest
import warnings

from zemi.params import (
    ParamSample,
    ParamSpace,
    PlaybookOptimizer,
    SampleTrialResult,
    validate_document,
)


def optimizer_config(strategy="grid"):
    result = {
        "strategy": strategy,
        "sample_trial": {
            "type": "@comp/zemi/sample_trial.py:TableDetectionSampleTrial",
            "dataset": "@comp/data/eval.json",
            "params": {},
        },
    }
    if strategy != "grid":
        result["max_trials"] = 4
    return result


def document():
    return {
        "system": {"version": "0.6", "params": {}},
        "component": {"name": "demo", "params": {}},
        "arsenals": [{"id": "local", "lifecycle": "external", "params": {}}],
        "modules": [{
            "id": "one",
            "kind": "playbook",
            "path": "@comp/one.ipynb",
            "arsenal": "local",
            "params": {"temperature": {"values": [0.0, 0.2], "start": 0.2}},
            "optimizer": optimizer_config(),
        }],
    }


class Params06SchemaTests(unittest.TestCase):
    def test_05_playbooks_migrate_to_06_modules_with_warning(self):
        source = document()
        source["system"]["version"] = "0.5"
        source["playbooks"] = source.pop("modules")
        source["playbooks"][0].pop("kind")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            migrated = validate_document(source)
        self.assertEqual(migrated["system"]["version"], "0.6")
        self.assertEqual(migrated["modules"][0]["kind"], "playbook")
        self.assertTrue(any("[[playbooks]]" in str(item.message) for item in caught))

    def test_only_implemented_module_kind_is_accepted(self):
        source = document()
        source["modules"][0]["kind"] = "python_script"
        with self.assertRaisesRegex(ValueError, "only 'playbook' is supported"):
            validate_document(source)

    def test_complete_document_is_valid_and_copied(self):
        source = document()
        validated = validate_document(source)
        self.assertEqual(validated["system"]["version"], "0.6")
        self.assertEqual(validated["modules"][0]["optimizer"]["strategy"], "grid")
        self.assertEqual(source, document())

    def test_variable_parameters_require_optimizer_and_sample_trial(self):
        source = document()
        del source["modules"][0]["optimizer"]
        with self.assertRaisesRegex(ValueError, "optimizer is required.*temperature"):
            validate_document(source)
        source = document()
        del source["modules"][0]["optimizer"]["sample_trial"]
        with self.assertRaisesRegex(ValueError, "optimizer.sample_trial is required"):
            validate_document(source)

    def test_fixed_parameters_forbid_optimizer(self):
        source = document()
        source["modules"][0]["params"] = {"temperature": 0.2}
        with self.assertRaisesRegex(ValueError, "optimizer is not allowed.*all fixed"):
            validate_document(source)
        del source["modules"][0]["optimizer"]
        self.assertNotIn("optimizer", validate_document(source)["modules"][0])

    def test_removed_structural_keys_are_rejected(self):
        for key, value in (("param_space_mode", "sampler"), ("sampler", {"strategy": "grid"})):
            with self.subTest(key=key):
                source = document()
                source["modules"][0][key] = value
                with self.assertRaisesRegex(ValueError, "unsupported structural keys"):
                    validate_document(source)
        source = document()
        source["modules"][0]["optimizer"]["objective"] = {"metric": "f1", "direction": "maximize"}
        with self.assertRaisesRegex(ValueError, "unsupported structural keys"):
            validate_document(source)

    def test_sample_trial_type_and_dataset_are_explicit(self):
        for trial, message in (
            ({"type": "table_detection", "dataset": "@comp/data/eval.json"}, "type must be"),
            ({"type": "@comp/trial.py:Trial"}, "dataset must be"),
            ({"type": "@comp/trial.py", "dataset": "@comp/data/eval.json"}, "type must be"),
        ):
            with self.subTest(trial=trial):
                source = document()
                source["modules"][0]["optimizer"]["sample_trial"] = trial
                with self.assertRaisesRegex(ValueError, message):
                    validate_document(source)

    def test_block_coordinate_blocks_are_preserved_and_validated(self):
        source = document()
        source["modules"][0]["optimizer"] = optimizer_config("block_coordinate")
        source["modules"][0]["optimizer"]["blocks"] = [["temperature"]]
        validated = validate_document(source)
        self.assertEqual(validated["modules"][0]["optimizer"]["blocks"], [["temperature"]])

    def test_03_migrates_with_warning_but_is_not_canonical(self):
        source = document()
        source["system"]["version"] = "0.3"
        source["playbooks"] = source.pop("modules")
        playbook = source["playbooks"][0]
        playbook.pop("kind")
        playbook["param_space_mode"] = "sampler"
        playbook["sampler"] = playbook.pop("optimizer")
        trial = playbook["sampler"]["sample_trial"]
        trial["implementation"] = "table_detection"
        trial["path"] = trial.pop("dataset")
        trial.pop("type")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            migrated = validate_document(source)
        self.assertEqual(migrated["system"]["version"], "0.6")
        self.assertIn("optimizer", migrated["modules"][0])
        self.assertTrue(any("deprecated" in str(item.message) for item in caught))


class ParamSpaceAndOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.space = ParamSpace(config={
            "fixed": "x",
            "temperature": {"values": [0.0, 0.2], "start": 0.2},
            "seed": {"range": {"min": 1, "max": 2, "step": 1}, "start": 1},
        })

    def result(self, sample, score):
        return SampleTrialResult(sample, [], {"quality": abs(score)}, score=score)

    def test_param_space_starts_first_and_keeps_fixed_values(self):
        optimizer = PlaybookOptimizer(config={"strategy": "grid"}, param_space=self.space)
        self.assertEqual(optimizer.next_param_sample([]), self.space.start)
        self.assertEqual(optimizer.candidates[0].values["fixed"], "x")

    def test_best_param_sample_always_maximizes_finite_score(self):
        optimizer = PlaybookOptimizer(config={"strategy": "grid"}, param_space=self.space)
        first, second = optimizer.candidates[:2]
        history = [self.result(first, -2.0), self.result(second, 3.0)]
        self.assertEqual(optimizer.best_param_sample(history), second)
        history.append(self.result(optimizer.candidates[2], math.nan))
        self.assertEqual(optimizer.best_param_sample(history), second)

    def test_coordinate_uses_best_score_as_anchor(self):
        space = ParamSpace(config={
            "x": {"values": [0, 1], "start": 0},
            "y": {"values": [0, 1], "start": 0},
        })
        optimizer = PlaybookOptimizer(
            config={"strategy": "coordinate", "max_samples": 4}, param_space=space
        )
        history = []
        while (sample := optimizer.next_param_sample(history)) is not None:
            history.append(self.result(sample, sample.values["x"] + sample.values["y"]))
        self.assertEqual(optimizer.best_param_sample(history).values, {"x": 1, "y": 1})

    def test_block_coordinate_preserves_joint_blocks(self):
        space = ParamSpace(config={
            "x": {"values": [0, 1], "start": 0},
            "y": {"values": [0, 1], "start": 0},
        })
        optimizer = PlaybookOptimizer(
            config={"strategy": "block_coordinate", "max_samples": 4, "blocks": [["x", "y"]]},
            param_space=space,
        )
        self.assertIn({"x": 1, "y": 1}, [sample.values for sample in optimizer.candidates])


if __name__ == "__main__":
    unittest.main()

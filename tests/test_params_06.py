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
            "params": {},
        },
        "trial_dataset": {"path": "@comp/data/eval.json"},
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
    def test_document_accepts_implicit_starts_and_rejects_invalid_domains(self):
        source = document()
        source["modules"][0]["params"] = {
            "temperature": {"values": [0.2, 0.0]},
            "seed": {"range": {"min": 1, "max": 3, "step": 1}},
        }
        validated = validate_document(source)
        self.assertEqual(ParamSpace(config=validated["modules"][0]["params"]).start.values,
                         {"temperature": 0.2, "seed": 1})
        for wrapper in ({"values": []}, {"values": [0], "start": 1}):
            source["modules"][0]["params"] = {"x": wrapper}
            with self.assertRaises(ValueError):
                validate_document(source)

    def test_kernel_reuse_defaults_to_true_and_can_be_disabled(self):
        source = document()
        self.assertIs(validate_document(source)['modules'][0]['optimizer']['reuse_kernel'], True)
        source['modules'][0]['optimizer']['reuse_kernel'] = False
        self.assertIs(validate_document(source)['modules'][0]['optimizer']['reuse_kernel'], False)
        source['modules'][0]['optimizer']['reuse_kernel'] = 'false'
        with self.assertRaisesRegex(ValueError, 'reuse_kernel must be boolean'):
            validate_document(source)

    def test_05_playbooks_migrate_to_06_modules_with_warning(self):
        source = document()
        source["system"]["version"] = "0.5"
        source["playbooks"] = source.pop("modules")
        source["playbooks"][0].pop("kind")
        optimizer = source["playbooks"][0]["optimizer"]
        optimizer["sample_trial"]["dataset"] = optimizer.pop("trial_dataset")["path"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            migrated = validate_document(source)
        self.assertEqual(migrated["system"]["version"], "0.6")
        self.assertEqual(migrated["modules"][0]["kind"], "playbook")
        self.assertEqual(migrated["modules"][0]["optimizer"]["trial_dataset"]["path"], "@comp/data/eval.json")
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

    def test_sample_trial_type_and_trial_dataset_path_are_explicit(self):
        for trial, message in (
            ({"type": "table_detection"}, "type must be"),
            ({"type": "@comp/trial.py"}, "type must be"),
        ):
            with self.subTest(trial=trial):
                source = document()
                source["modules"][0]["optimizer"]["sample_trial"] = trial
                with self.assertRaisesRegex(ValueError, message):
                    validate_document(source)
        source = document()
        source["modules"][0]["optimizer"].pop("trial_dataset")
        with self.assertRaisesRegex(ValueError, "optimizer.trial_dataset is required"):
            validate_document(source)
        source = document()
        source["modules"][0]["optimizer"]["trial_dataset"] = {"path": ""}
        with self.assertRaisesRegex(ValueError, "trial_dataset.path must be"):
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
        trial["path"] = playbook["sampler"].pop("trial_dataset")["path"]
        trial.pop("type")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            migrated = validate_document(source)
        self.assertEqual(migrated["system"]["version"], "0.6")
        self.assertIn("optimizer", migrated["modules"][0])
        self.assertTrue(any("deprecated" in str(item.message) for item in caught))


class ParamSpaceAndOptimizerTests(unittest.TestCase):
    def test_optional_start_defaults_and_explicit_override(self):
        space = ParamSpace(config={
            "temperature": {"values": [0.5, 0.0, 0.2]},
            "seed": {"range": {"min": 2, "max": 6, "step": 2}},
            "override": {"values": [1, 2], "start": 2},
            "range_override": {"range": {"min": 0, "max": 2, "step": 1}, "start": 1},
            "float_range": {"range": {"min": 0.0, "max": 1.0, "step": 0.2}},
        })
        self.assertEqual(space.start.values, {"temperature": 0.5, "seed": 2, "override": 2, "range_override": 1, "float_range": 0.0})

    def test_composite_default_is_complete_and_independent(self):
        binding = {"prompt_name": "first", "prompt_file": "@comp/params/prompts.md",
                   "encoder": "@comp/params/encoder.py:encode", "options": {"formats": ["cells"]}}
        space = ParamSpace(config={"encoding_prompt": {"values": [binding, {"prompt_name": "second"}]}})
        self.assertEqual(space.start.values["encoding_prompt"], binding)
        sample = space.start
        sample.values["encoding_prompt"]["options"]["formats"].append("rows")
        self.assertEqual(space.start.values["encoding_prompt"], binding)
        binding["prompt_name"] = "changed"
        self.assertEqual(space.start.values["encoding_prompt"]["prompt_name"], "first")

    def test_optional_start_validation(self):
        for wrapper, message in [
            ({"values": []}, "non-empty array"),
            ({"values": [], "start": 0}, "non-empty array"),
            ({"values": [0, 1], "start": 2}, "member of its domain"),
            ({"range": {"min": 0, "max": 4, "step": 2}, "start": 1}, "member of its domain"),
            ({"values": [{"name": "first"}], "start": {"name": "other"}}, "member of its domain"),
        ]:
            with self.subTest(wrapper=wrapper), self.assertRaisesRegex(ValueError, message):
                ParamSpace(config={"x": wrapper})

    def test_default_start_preserves_grid_and_all_optimizer_histories(self):
        implicit = ParamSpace(config={"x": {"values": [2, 0, 1]},
                                      "y": {"range": {"min": 0, "max": 2, "step": 1}}})
        explicit = ParamSpace(config={"x": {"values": [2, 0, 1], "start": 2},
                                      "y": {"range": {"min": 0, "max": 2, "step": 1}, "start": 0}})
        expected = [{"x": x, "y": y} for x in [2, 0, 1] for y in [0, 1, 2]]
        self.assertEqual([s.values for s in implicit.grid()], expected)
        self.assertEqual(implicit.grid(), explicit.grid())
        for strategy in ("grid", "random", "coordinate", "block_coordinate"):
            config = {"strategy": strategy, "max_trials": 9, "seed": 17}
            if strategy == "block_coordinate":
                config["blocks"] = [["x", "y"]]
            histories = []
            for space in (implicit, explicit):
                optimizer = PlaybookOptimizer(config=config, param_space=space)
                history = []
                while (sample := optimizer.next_param_sample(history)) is not None:
                    history.append(self.result(sample, sample.values["x"] + sample.values["y"]))
                    optimizer.observe(history, history[-1])
                keys = [r.sample.key() for r in history]
                self.assertEqual(len(keys), len(set(keys)))
                self.assertEqual(keys.count(space.start.key()), 1)
                histories.append((keys, optimizer.best_param_sample(history)))
            with self.subTest(strategy=strategy):
                self.assertEqual(histories[0], histories[1])

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

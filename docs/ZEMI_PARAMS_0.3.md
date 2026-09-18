# ZEMI Params 0.3

Status: proposed normative specification, awaiting implementation approval.

This document defines the configuration model and execution semantics for ZEMI
Params 0.3. Keywords **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are
normative.

## 1. Goals and boundaries

ZEMI Params 0.3 separates ZEMI-owned structure from user-owned values, makes
parameter reuse explicit, and gives parameter search a stable trial model. The
term `pipeline` is replaced by `system`.

ZEMI-owned fields live only in closed structural sections. Arbitrary user keys
are allowed only below a `params` section. The four user parameter scopes are:

- `system.params`
- `component.params`
- `arsenals[].params`
- `playbooks[].params`

There is no automatic inheritance between these scopes. Values move between
scopes only through `ref` or `__include__`.

The optimizer is an implementation detail of a sampler. It is not a separate
top-level configuration object. DSPy MAY be used inside a notebook/playbook or
a sampler adapter, but ZEMI MUST NOT require `dspy.Module`, `forward`, or any
other DSPy program shape.

## 2. Canonical document shape

The canonical top-level keys are `system`, `component`, `arsenals`, and
`playbooks`. A Params 0.3 document MUST contain `system.version = "0.3"`, one
`[component]`, one or more `[[arsenals]]`, and one or more `[[playbooks]]`.

Every structural table is closed: an unknown built-in key is an error. The
contents of its `params` child are open and JSON-compatible, subject to the
wrapper rules below.

### 2.1 Full example

```toml
[system]
version = "0.3"

[system.params]
locale = "en"

[system.params.defaults]
temperature = 0.2
max_tokens = 512

[component]
name = "two-arsenal-search"
stop_on_error = true

[component.params]
dataset_root = "@comp/data"

[component.params.generation]
__include__ = { ref = "system.params.defaults" }
max_tokens = 768

[[arsenals]]
id = "local"
config_path = "@comp/zemi/llm_curated_set_model_mode.toml"
lifecycle = "job"

[arsenals.params]
device = "cpu"
generation = { ref = "component.params.generation" }

[[arsenals]]
id = "remote"
config_path = "@comp/zemi/llm_external_providers.toml"
lifecycle = "external"

[arsenals.params]
device = "remote"

[[playbooks]]
id = "summarize-local"
path = "playbook.ipynb"
arsenal = "local"
enabled = true

[playbooks.params]
__include__ = { ref = "component.params.generation" }
locale = { ref = "system.params.locale" }
device = { ref = "arsenals.local.params.device" }
temperature = { values = [0.0, 0.2, 0.5], start = 0.2 }
prompt_style = { values = ["brief", "detailed"], start = "brief" }

[playbooks.sampler]
strategy = "grid"
max_samples = 6
seed = 17

[playbooks.sampler.sample_trial.dataset]
adapter = "jsonl"
path = "@comp/data/eval.jsonl"

[playbooks.sampler.sample_trial.evaluator]
adapter = "@comp/evaluators/summary.py:evaluate"

[playbooks.sampler.sample_trial.evaluator.params]
language = { ref = "system.params.locale" }

[playbooks.sampler.sample_trial.objective]
metric = "quality"
direction = "maximize"

[[playbooks]]
id = "extract-remote"
path = "playbook_gbnf.ipynb"
arsenal = "remote"
enabled = true

[playbooks.params]
__include__ = [
  { ref = "system.params.defaults" },
  { ref = "component.params.generation" },
]
device = { ref = "arsenals.remote.params.device" }
schema_mode = { values = ["strict", "repair"], start = "strict" }
temperature = { range = { min = 0.0, max = 0.4, step = 0.2 }, start = 0.0 }

[playbooks.sampler]
strategy = "coordinate"
max_samples = 5
seed = 23

[playbooks.sampler.sample_trial.dataset]
adapter = "csv"
path = "@comp/data/extraction.csv"

[playbooks.sampler.sample_trial.dataset.params]
input_column = "text"

[playbooks.sampler.sample_trial.evaluator]
adapter = "@comp/evaluators/extraction.py:evaluate"

[playbooks.sampler.sample_trial.objective]
metric = "f1"
direction = "maximize"
```

`arsenals.local.params.device` is a logical named lookup. Arrays remain the TOML
representation, but `arsenals.<id>` and `playbooks.<id>` are valid reference
namespaces. Duplicate ids are therefore forbidden.

## 3. Built-in fields

### 3.1 `system`

- `version` (required string): schema version; Params 0.3 accepts exactly
  `"0.3"`.
- `params` (optional table): system-wide user values. Its presence does not
  cause inheritance.

### 3.2 `component`

- `name` (optional non-empty string): logical component name; defaults to the
  component directory name.
- `stop_on_error` (optional boolean, default `true`): stop the enclosing job
  after the first failed trial.
- `params` (optional table): component user values.

### 3.3 `arsenals[]`

- `id` (required non-empty string): document-unique stable identifier. It MUST
  match `[A-Za-z][A-Za-z0-9_-]*`.
- `config_path` (optional ZEMI path string): Arsenal configuration. It is
  required when `lifecycle = "job"`.
- `lifecycle` (optional enum, default `"playbook"`): `job` starts once before
  the Arsenal's playbooks and stops once afterwards; `playbook` lets each run
  manage its own session; `external` never starts or stops Arsenal processes.
- `params` (optional table): Arsenal user values. Arsenal built-ins and user
  values MUST NOT share a table.

### 3.4 `playbooks[]`

- `id` (required non-empty string): document-unique stable identifier using the
  same syntax as Arsenal ids.
- `path` (required non-empty relative path or `@comp/...` path): source notebook.
- `arsenal` (required string): exact parent Arsenal id. The referenced Arsenal
  MUST exist. This explicit edge replaces containment-based or inferred parent
  selection.
- `enabled` (optional boolean or `select` wrapper, default `true`).
- `params` (optional table): user parameters and the playbook `ParamSpace`.
- `sampler` (optional table): sampling policy and SampleTrial definition. If
  absent, ZEMI executes the single start/fixed sample once.

### 3.5 `sampler`

- `strategy` (required enum): `grid`, `random`, `coordinate`, or
  `block_coordinate`.
- `max_samples` (optional positive integer): hard proposal limit. It is required
  for `random`, `coordinate`, and `block_coordinate`; for `grid` omission means
  the complete finite Cartesian product.
- `seed` (optional integer): reproducibility seed.
- `block_size` (required positive integer only for `block_coordinate`).
- `sample_trial` (required table): dataset, evaluator, and objective contract.

An implementation MAY expose strategy-specific adapters, including one backed
by DSPy, but they obey the same `propose(history)` / `observe(...)` contract.

### 3.6 `sample_trial`

- `dataset` (required table): `adapter` identifies the dataset loader; `path` is
  an optional `@comp/...` or `@inst/...` source; `params` contains only adapter
  user values.
- `evaluator` (required table): `adapter` identifies a callable or registered
  evaluator; `params` contains only evaluator user values. The evaluator accepts
  the completed SampleTrial (all PlaybookRuns and their outputs) and returns a
  finite numeric metric map plus optional JSON-compatible feedback.
- `objective` (required table): `metric` is the exact evaluator metric key and
  `direction` is exactly `maximize` or `minimize`.

Dataset and evaluator adapters are deliberately small interfaces; Params 0.3
does not prescribe a machine-learning framework.

## 4. Parameter values and ParamSpace

A key below any `params` table is a user parameter. A plain JSON-compatible
value is fixed. In `playbooks[].params`, either of these exact wrappers declares
a variable dimension:

```toml
x = { values = [1, 2, 3], start = 2 }
y = { range = { min = 0.0, max = 1.0, step = 0.1 }, start = 0.5 }
```

`values` MUST be a non-empty array of unique JSON-compatible values and `start`
MUST equal one member with type-sensitive equality. `range` MUST contain exactly
`min`, `max`, and `step`; all are finite numbers, `step > 0`, `max >= min`, and
`start` MUST lie on the inclusive generated grid. The wrapper MUST contain
exactly its domain key and `start`. This keeps the start value beside the domain
description.

`ParamSpace` is the ordered collection of variable dimensions after reference
resolution. TOML declaration order is preserved. `ParamSample` is one immutable
mapping of every resolved playbook parameter to a concrete value. Fixed values
are included in every ParamSample.

The existing interactive wrappers remain distinct:

- `{ select = [...] }` chooses one value once while loading the job and does not
  create a search dimension.
- `{ input = ... }` obtains one typed value once while loading the job.

They MAY appear in any `params` table. They are resolved before construction of
the ParamSpace.

## 5. Explicit references and includes

`{ ref = "dotted.path" }` copies one scalar, array, or table. `__include__`
accepts one ref wrapper or a non-empty ordered array of ref wrappers; every ref
MUST resolve to a table. References MAY target only a `params` table or a value
below one. Structural fields such as `component.stop_on_error` and
`arsenals.local.config_path` cannot be referenced or included.

Resolution is deterministic:

1. Parse TOML and validate the closed structural shape and unique ids.
2. Build named `arsenals.<id>` and `playbooks.<id>` lookup namespaces.
3. Resolve `system.params`.
4. Resolve `component.params`.
5. Resolve each `arsenals[].params` in document order.
6. Resolve each `playbooks[].params`, dataset params, and evaluator params in
   document order.
7. Within one table, apply `__include__` entries left-to-right; later includes
   replace earlier keys, then local keys replace all included keys.
8. Resolve `select` and `input` once, validate variable wrappers, and construct
   ParamSpace and its start ParamSample.
9. For each proposal, overlay only the sampled dimension values onto the fixed
   resolved playbook params.

References are deep-copied. Missing paths, traversal through non-tables, cycles,
references to structural fields, and malformed wrappers are errors. No parent
scope is merged implicitly; the same key in two scopes is unrelated unless a
ref/include explicitly connects them.

## 6. Trials and execution lifecycle

The single normative hierarchy is:

`JobTrial → PlaybookTrial → SampleTrial → PlaybookRun`

- `JobTrial`: one execution of a component parameter document.
- `PlaybookTrial`: one enabled playbook traversing its ParamSpace.
- `SampleTrial`: one ParamSample evaluated against the complete dataset.
- `PlaybookRun`: one execution of that playbook for one dataset item.

For each PlaybookTrial the conceptual outer loop is:

```text
sample = sampler.propose(history)
run playbook once for every dataset item using sample
result = evaluator.evaluate(completed_sample_trial)
sampler.observe(history, result)
```

The evaluator runs only after every required PlaybookRun for that SampleTrial
has been collected. Group metrics that need multiple runs are computed at that
point. A sampler cannot claim the objective has been reached before evaluator
metrics exist. `observe` receives the ParamSample, run records, metrics,
feedback, and failure state; `history` is the ordered sequence of observed
SampleTrial results.

Strategy minimum semantics:

- `grid`: deterministic Cartesian product in parameter declaration order, with
  each domain's declared order; the start sample is proposed first.
- `random`: seeded sampling without replacement for finite discrete spaces; the
  start sample is first.
- `coordinate`: start sample first, then vary one dimension at a time around the
  best observed sample; ties keep the earlier sample.
- `block_coordinate`: as coordinate, but varies consecutive declaration-order
  blocks of at most `block_size` dimensions.

The sampler owns any optimizer state. A trial stops when the finite grid is
exhausted, `max_samples` is reached, or the sampler reports exhaustion. Objective
comparison uses only the configured metric and direction. Missing, boolean,
NaN, or infinite objective metrics fail the SampleTrial and are not valid
objective observations.

## 7. Validation and reporting

Validation MUST happen before starting an Arsenal or executing a notebook.
Errors MUST name the full logical path and, for array items, the source index or
id. Besides the field rules above, ZEMI MUST reject:

- unknown top-level or structural keys;
- arbitrary keys outside `params`;
- duplicate Arsenal or playbook ids;
- missing parent Arsenal references;
- absolute filesystem paths in configuration;
- unsupported strategy, direction, adapter shape, or parameter wrapper;
- duplicate ParamSamples proposed by a sampler;
- evaluator results without the objective metric.

Reports MUST preserve the hierarchy and stable ids. Every SampleTrial records
its ParamSample, proposal ordinal, PlaybookRuns, evaluator metrics and feedback,
objective value, status, timestamps, and errors. Existing notebook artifacts
remain attached to the corresponding PlaybookRun.

## 8. Migration from the current implementation

The implementation preceding 0.3 already supports `ref`, `__include__`,
`select`, `input`, `each`, `pipeline_params`, `component_params`, top-level
`playbooks_params`, nested `arsenals.playbooks_params`, and
`playbook_params`. Migration MUST preserve the proven resolver behavior while
moving to the canonical names and trial hierarchy.

Canonical mappings are:

- `pipeline_params` → `system.params`
- user entries in `component_params` → `component.params`; known lifecycle
  fields → `component`
- `arsenals[].name` → `arsenals[].id`
- `arsenal_config_path` → `config_path`
- `arsenal_start_and_stop_at_job_level = true` → `lifecycle = "job"`
- `playbooks_params` / `arsenals[].playbooks_params` → top-level `playbooks`
- `playbook_name` → `path` (and a generated stable `id` only in a migration
  tool, never silently at canonical validation time)
- `playbook_params` → `params`
- `{ each = [...] }` → `{ values = [...], start = <explicit value> }`

The 0.3 loader SHOULD accept the old complete document shape during one
compatibility window, normalize it before canonical validation, and emit a
clear deprecation warning. A document MUST NOT mix canonical and legacy shapes.
Because legacy `each` lacks a start value, automatic runtime normalization MUST
use its first element and warn; a source migration tool MUST write that choice
explicitly.

## 9. Explicit unresolved schema reminder

The current TOML schema mixes ZEMI built-in fields and arbitrary user values.
This boundary must be formalized through closed structural sections and open
`params` sections, not left implicit. Params 0.3 specifies that boundary; code,
examples, templates, and future extensions must not weaken it by accepting
unknown structural keys as user parameters.

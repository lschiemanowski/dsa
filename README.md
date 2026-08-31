# dsa

`dsa` is a small, auditable agent for answering one completely specified data
science question about a supplied DuckDB database. The caller supplies the
database path, question, exact JSON Schema for the answer, model configuration,
and host-enforced run policy.

The project is a clean rewrite of the existing dsagent behavior. It is not a
conversational task-definition system.

## Current milestone

Milestone 6A adds isolated, resumable benchmark orchestration over the content-pinned
Hugging Face packs and native Databricks MLflow evaluation path:

- strict, serializable request and policy models
- JSON Schema Draft 2020-12 validation
- a Pydantic AI structured-output episode
- schema feedback and bounded answer retries
- typed terminal success and failure outcomes
- native Pydantic AI messages and usage in one terminal record
- atomic private record retention with one SHA-256 digest
- deterministic schema inspection with unambiguous quoted DuckDB relation names
- one structured-identity and bound-plan checked SQL statement per query tool call
- complete small query results inline
- automatic full-result Parquet retention with at most five preview rows
- run-private artifact handles with same-descriptor integrity checks
- transactional publication for complete multi-output Python batches
- an injected Python executor protocol using managed database, input, and output paths
- one private database copy per run with retained source and final SHA-256 identities
- transactional per-call database mutation with rollback on every failed Python call
- a digest-pinned, never-pull, non-root, networkless Docker backend with read-only binds
- Docker memory, CPU, process, elapsed-time, writable-tmpfs, and diagnostic limits
- disabled daemon logging and bounded host recovery of database and output bytes
- trusted process quiescence and WAL checkpointing before database recovery
- default cleanup of private working databases with an operator-only debug override
- direct final answers or same-run retained JSON final answers
- an operator-only `report_to_mlflow` flag with context-local Pydantic AI tracing
- exact terminal-record and artifact-manifest metadata export to Databricks
- native MLflow Evaluation Datasets with lossless host-only expectations and pack scorers
- one immutable public Online Retail II pack containing twenty evaluation cases
- exact Hugging Face repository revision, manifest, case-export, and database identities
- immutable content-addressed benchmark studies and deterministic matrix expansion
- fresh subprocesses and private attempts for every pack-model-repetition cell
- separate explicit MLflow case-worker and benchmark cell-worker bounds
- atomic no-overwrite local receipts correlated to tagged Databricks evaluation runs
- conservative explicit resume that never guesses about ambiguous remote work
- runtime-only workspace paths, Databricks dataset names, and secrets outside study identity

Model-authored Python is never executed on the host. The `run_python` tool is
registered only when the host injects an executor. MLflow reporting uses Databricks
only; a local MLflow server or tracking database is not part of the design.

## Public boundary

```python
from pathlib import Path

from dsa import ModelConfiguration, RunPolicy, RunRequest

request = RunRequest(
    database_path=Path("data/example.duckdb"),
    question="How many rows are in the events relation?",
    answer_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"count": {"type": "integer", "minimum": 0}},
        "required": ["count"],
        "additionalProperties": False,
    },
    model=ModelConfiguration(
        name="openrouter:example/model",
        settings={"temperature": 0, "seed": 7},
    ),
    policy=RunPolicy(),
)
```

Credentials, provider endpoints, and evaluator expectations are deliberately
excluded from this request and from the terminal record. Provider credentials
remain runtime environment configuration.

Query transport is host-selected. A response marked `inline` is complete. A response
marked `artifact` identifies a complete retained Parquet table and includes only a
bounded orientation preview. The preview is not an analytical substitute for the
artifact. Python receives selected artifacts through `DSAGENT_INPUTS` rather than
through copied transcript content.

The Docker image must be built or obtained before a run and supplied by immutable
SHA-256 identity. Runtime execution never pulls or builds an image:

```text
docker build --tag dsa-python:m3 docker
docker image inspect --format '{{.Id}}' dsa-python:m3
```

The resulting raw `sha256:...` image ID is valid for that local Docker store. A
registry deployment should use its `repository@sha256:...` digest instead.

```python
from dsa import DockerPythonExecutor, default_docker_configuration, run_analysis

executor = DockerPythonExecutor(default_docker_configuration("sha256:<64 hex digits>"))
completion = await run_analysis(request, python_executor=executor)
```

Install the optional dependencies and provide operator-owned Databricks runtime
configuration to resolve packs and report any ordinary run:

```text
uv sync --all-groups --extra huggingface --extra mlflow --frozen
export MLFLOW_TRACKING_URI=databricks
export MLFLOW_EXPERIMENT_ID=<experiment-id>
export DATABRICKS_HOST=<workspace-url>
export DATABRICKS_TOKEN=<token>
```

```python
completion = await run_analysis(
    request,
    runs_directory=Path("runs"),
    report_to_mlflow=True,
)
```

The repository pins Online Retail II 1.0.0 through
`evaluation-packs/online-retail-ii-1.0.0.json`. Load it from its exact public Hugging
Face commit, then bind the machine-local model and policy only when evaluating:

```python
from pathlib import Path

from dsa import (
    HuggingFacePackReference,
    load_huggingface_evaluation_pack,
    run_mlflow_evaluation,
)

reference = HuggingFacePackReference.model_validate_json(
    Path("evaluation-packs/online-retail-ii-1.0.0.json").read_bytes()
)
pack = load_huggingface_evaluation_pack(reference)
result = run_mlflow_evaluation(
    pack,
    dataset_name="<databricks-unity-catalog-dataset>",
    runs_directory=Path("runs"),
    model_configuration=request.model,
    policy=request.policy,
)
```

Only case identity/version, database identity/digest, question, and answer schema enter
MLflow dataset inputs. Reference answers and scorer policies remain evaluator-only
expectations, encoded as canonical JSON strings so managed-dataset number coercion cannot
change their meaning. Exact packs keep integers and floats exact; a pack may explicitly
opt into bounded relative/absolute tolerance for expected floating-point leaves while
integers remain type-exact. Hugging Face cache paths, model configuration, run policy,
and executor configuration remain host runtime bindings.

Benchmark studies bind exact pack references, model configurations, policy, Docker image,
concurrency, repetitions, and the exact agent Git revision. Runtime files bind only a new
local workspace and one distinct `catalog.schema.table` Databricks dataset per pack. Both
inputs must be canonical JSON. Preflight resolves every pack, checks the already-present
immutable Docker image, requires Databricks configuration, and verifies the running Git
revision before starting any cell. Tracked or untracked implementation changes under
`src/dsa`, `pyproject.toml`, or `uv.lock` are rejected because they are not represented by
that revision:

```text
dsa-benchmark plan --study study.json --runtime runtime.json
dsa-benchmark run --study study.json --runtime runtime.json
dsa-benchmark run --study study.json --runtime runtime.json --resume
dsa-benchmark report --study study.json --runtime runtime.json --output reports
```

`plan` performs the same complete preflight as `run` but makes no model call or MLflow
write. `--resume` skips only a locally verified completed receipt. A partial attempt or
unverifiable receipt is reported as ambiguous and preserved for operator inspection.
`report` requires a complete matrix of verified receipts, reads only their exact
Databricks MLflow runs, recomputes the existing scorers against the pinned packs, and
atomically publishes canonical JSON plus deterministic Markdown without running a
model or modifying remote state. Literal exactness and pack-policy matches are retained
as separate metrics, so an allowed floating-point tolerance never inflates exact
accuracy. The execution and publication contracts are recorded
in `docs/milestone-6a.md` and `docs/milestone-6b.md`.

`completion.reporting` is `disabled`, `reported`, or `failed`. A reporting failure does
not change `completion.outcome` or the canonical local `terminal.json`. Enabled reporting
uploads the native Pydantic AI trace, the exact terminal record, and artifact metadata;
it does not upload retained artifact contents, DuckDB files, or private workspaces.

Each Python call receives a size-limited tmpfs copy of the private attempt database,
explicitly selected read-only artifact inputs, and a size-limited tmpfs output directory.
Only the database seed and selected inputs are host bind-mounted, both read-only; the
host recovers database and output bytes through bounded streams. Model code resolves
`DSAGENT_DATABASE`, `DSAGENT_INPUTS`, and `DSAGENT_OUTPUTS` through `os.environ`.
Declared outputs must be `.json` or `.parquet`; `expected_outputs=[]` is valid for a
database-only call. Successful database changes become visible to later tools in the
same run. Any failed call is discarded. The source database is never mounted and is
never modified.

Once a valid run starts, it produces exactly one terminal outcome. Success
contains the answer validated against the caller's schema. Failure contains a
stage, stable code, safe message, and bounded diagnostics. Cancellation is
recorded and then propagated to the caller.

## Development

```text
uv sync --all-groups --frozen
uv run ruff check .
uv run pyright
uv run pytest
```

The opt-in real Docker tier requires an already available immutable image:

```text
DSA_DOCKER_TEST_IMAGE=sha256:<64 hex digits> uv run pytest -m integration
```

The real Databricks acceptance test additionally requires the configuration above, a
Unity Catalog dataset name in `DSA_MLFLOW_DATASET_NAME`, and explicit opt-in:

```text
DSA_DATABRICKS_TEST=1 uv run pytest -m databricks
```

The full benchmark-cell acceptance also needs the already-present Docker image and exact
public pack access. It uses Pydantic AI's deterministic test model rather than a paid
provider:

```text
DSA_BENCHMARK_DATABRICKS_TEST=1 \
DSA_DOCKER_TEST_IMAGE=sha256:<64 hex digits> \
uv run pytest tests/test_benchmark_databricks.py
```

The exact public Hugging Face boundary has a separate opt-in acceptance test:

```text
DSA_HUGGINGFACE_TEST=1 uv run pytest tests/test_online_retail_pack.py
```

Python 3.12 is the development and CI baseline. Dependencies are resolved in
`uv.lock`.

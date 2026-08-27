# dsa

`dsa` is a small, auditable agent for answering one completely specified data
science question about a supplied DuckDB database. The caller supplies the
database path, question, exact JSON Schema for the answer, model configuration,
and host-enforced run policy.

The project is a clean rewrite of the existing dsagent behavior. It is not a
conversational task-definition system.

## Current milestone

Milestone 3 adds a private writable database lifecycle and the production Docker
executor to the deterministic run boundary:

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
- a digest-pinned, non-root, networkless Docker backend with narrow mounts
- Docker memory, CPU, process, elapsed-time, scratch, and diagnostic limits
- default cleanup of private working databases with an operator-only debug override
- direct final answers or same-run retained JSON final answers

Model-authored Python is never executed on the host. The `run_python` tool is
registered only when the host injects an executor. Databricks Free Edition MLflow
integration remains a later milestone. A local MLflow server is not part of the design.

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

Each Python call receives only the private attempt database, explicitly selected
artifact inputs, and an empty output directory. Model code resolves
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

Python 3.12 is the development and CI baseline. Dependencies are resolved in
`uv.lock`.

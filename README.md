# dsa

`dsa` is a small, auditable agent for answering one completely specified data
science question about a supplied DuckDB database. The caller supplies the
database path, question, exact JSON Schema for the answer, model configuration,
and host-enforced run policy.

The project is a clean rewrite of the existing dsagent behavior. It is not a
conversational task-definition system.

## Current milestone

Milestone 2 adds bounded database investigation and automatic artifact routing to
the deterministic run boundary:

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
- an injected Python executor protocol using managed input and output paths
- direct final answers or same-run retained JSON final answers

Model-authored Python is never executed on the host by this milestone. The
`run_python` tool is registered only when the host injects an executor. Milestone 3
will provide the production Docker executor, run-private database copies, and
transactional mutation behavior. Databricks Free Edition MLflow integration remains
a later milestone. A local MLflow server is not part of the design.

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

Python 3.12 is the development and CI baseline. Dependencies are resolved in
`uv.lock`.

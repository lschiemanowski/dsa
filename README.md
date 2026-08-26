# dsa

`dsa` is a small, auditable agent for answering one completely specified data
science question about a supplied DuckDB database. The caller supplies the
database path, question, exact JSON Schema for the answer, model configuration,
and host-enforced run policy.

The project is a clean rewrite of the existing dsagent behavior. It is not a
conversational task-definition system.

## Current milestone

Milestone 1 establishes the deterministic run boundary:

- strict, serializable request and policy models
- JSON Schema Draft 2020-12 validation
- a Pydantic AI structured-output episode
- schema feedback and bounded answer retries
- typed terminal success and failure outcomes
- native Pydantic AI messages and usage in one terminal record
- atomic private record retention with one SHA-256 digest

Database inspection, SQL, artifact externalization, Docker, and Databricks
MLflow arrive in later milestones. The current code verifies that the supplied
database is an available file but does not open it yet.

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

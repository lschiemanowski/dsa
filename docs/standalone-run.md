# Standalone task execution

`dsa-run` executes exactly one caller-supplied analysis task. It uses the same
`RunRequest`, hardened Docker Python executor, terminal record, and optional Databricks
MLflow reporting path as benchmark cases, without creating an evaluation dataset or
requiring a pack, expected answer, scorer, benchmark receipt, or study report.

The request file is canonical JSON. Provider credentials remain in the process
environment and must not appear in the request. The following Python snippet creates a
valid request file with the default host-enforced policy:

```python
import json
from pathlib import Path

from dsa import ModelConfiguration, RunPolicy, RunRequest

request = RunRequest(
    database_path=Path("data/source.duckdb"),
    question="How many distinct invoices are recorded in the database?",
    answer_schema={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "invoice_count": {"type": "integer", "minimum": 0},
        },
        "required": ["invoice_count"],
        "additionalProperties": False,
    },
    model=ModelConfiguration(
        name="openrouter:provider/model",
        settings={"temperature": 0},
    ),
    policy=RunPolicy(),
)
content = json.dumps(
    request.model_dump(mode="json"),
    ensure_ascii=False,
    allow_nan=False,
    sort_keys=True,
    separators=(",", ":"),
) + "\n"
Path("task.json").write_text(content)
```

A relative `database_path` is resolved from the request file's directory. Run the task
with an immutable image already present in the local Docker store:

```text
dsa-run \
  --request task.json \
  --runs-directory runs \
  --docker-image sha256:<64 lowercase hex digits>
```

Add `--report-to-mlflow` to project the ordinary run to the configured Databricks
experiment. Reporting failure does not change the local analysis outcome.

The command writes one canonical JSON object to standard output. A successful result
contains the schema-valid answer when its canonical encoding is at most 1 MiB. Larger
answers remain available in the retained terminal record and are reported with
`answer_inline: false`. A failed analysis exposes only its stable stage and code, not raw
provider text. Every started run reports the exact terminal-record path, SHA-256 digest,
and byte length.

Exit status is `0` for analysis success, `1` for a terminal analysis failure or an
unexpected execution failure, `2` for invalid CLI input or request configuration, and
`130` for operator interruption. Standalone tasks are deliberately not eligible for
benchmark scoring or publication; put a task in a versioned evaluation pack when an
expected answer and scorer policy are required.

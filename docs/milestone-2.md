# Milestone 2: SQL and automatic artifact routing

## Objective

One deterministic agent episode can inspect a supplied DuckDB database, execute
bounded read-only SQL, retain complete larger tables as Parquet, pass selected
artifacts through managed Python locators, and finish either directly or from a
same-run JSON artifact.

The public answer remains exactly the caller-defined JSON value. The terminal record
is canonical run state. Artifact files are integrity-addressed run-private data that
the terminal record references by metadata.

## Database tools

`inspect_database` returns a sorted catalog of schema-qualified tables and views when
called without a relation. When given `schema.name`, it returns ordered column names,
DuckDB types, and nullability. It never returns rows. When the catalog or column list
does not fit its byte limit, it returns the bounded prefix with `complete` set to
`false`.

`query_database` accepts exactly one DuckDB-parsed `SELECT`, `WITH`, or `VALUES`
statement. The connection is read-only and external access is disabled. Writes,
multiple statements, external paths, and external reader or scanner functions are
rejected before execution.

The host enforces elapsed time, DuckDB memory, row-count, materialized-result byte,
artifact, per-tool response, and cumulative model-visible response limits.

## Result transport

A result is inline only when the complete table has no more rows than the preview
limit and its complete canonical JSON representation fits the per-tool byte limit.

Every other accepted result is written in full as Parquet. The model receives:

- ordered column names and DuckDB types
- the total row count
- a run-private artifact handle
- a managed `DSAGENT_INPUTS` path
- no more than five orientation rows
- an explicit indication that the preview is incomplete

The implementation converts only the bounded preview to transcript JSON. It does not
convert the complete retained table into Python row lists or place it in model
messages.

If the complete table exceeds query or artifact limits, the call fails with guidance
to filter or aggregate further. No partial artifact is published and no truncated
table is represented as complete.

## Artifact boundary

Artifact handles are sequential within one run. Publication copies bytes to a private
temporary file, flushes them, publishes without overwriting, and records byte size,
SHA-256 digest, media type, managed relative path, and producing tool call.

Every artifact is checked against its retained size and digest before consumption.
Tabular artifacts use Parquet. Final-answer artifacts use finite UTF-8 JSON.

## Python seam

Milestone 2 defines an asynchronous `PythonExecutor` protocol. The runner registers
`run_python` only when an executor is explicitly injected. There is no default local
executor and no use of host `exec`, `eval`, subprocesses, or notebooks.

The executor receives:

- model-authored source text
- the supplied database path
- a private directory containing copies of only the selected input artifacts
- a private output directory
- `DSAGENT_DATABASE`, `DSAGENT_INPUTS`, and `DSAGENT_OUTPUTS` locators
- the exact expected output file names

The host accepts only the declared regular `.json` and `.parquet` files, then publishes
them through the same artifact boundary. Milestone 3 will implement this protocol with
Docker and a run-private database copy.

## Final output

The model can call `final_answer` with the caller-requested JSON directly or call
`answer_from_artifact` with a same-run JSON handle. Both paths pass through the same
Draft 2020-12 validator and the same global validation-attempt budget.

Unknown, unavailable, modified, non-JSON, oversized, malformed, non-finite, or
schema-invalid answer artifacts receive bounded retry feedback.

## Acceptance evidence

The milestone suite covers real DuckDB catalog inspection, query safety, source
preservation, inline transport, automatic Parquet transport, full retained row
content, bounded previews, query rejection without partial publication, artifact
integrity, managed Python inputs and outputs, direct and artifact-backed final output,
cumulative tool-result limits, terminal-record safety, and all Milestone 1 behavior.

Docker, writable database copies, rollback, Databricks MLflow, live providers, a CLI,
and a conversational demo remain outside this milestone.

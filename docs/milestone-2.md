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
called without a relation. Each relation is represented as a canonical quoted DuckDB
identifier such as `"analytics"."events"`, preserving dots and quotes inside either
identifier component. Passing that exact returned name yields ordered column names,
DuckDB types, and nullability. It never returns rows. When the catalog or column list
does not fit its byte limit, the host stops reading the sorted cursor and returns the
bounded prefix with `complete` set to `false`. Neither catalog nor column inspection
materializes metadata beyond the model-visible byte budget.

`query_database` accepts exactly one DuckDB-parsed `SELECT`, `WITH`, or `VALUES`
statement. The host checks the first significant keyword while correctly skipping line,
block, and nested block comments, rather than accepting every syntax DuckDB classifies
as a semantic `SELECT`. The connection is read-only and external access is disabled.
Before execution, DuckDB binds the query inside a host-generated parameterized row-limit
wrapper with optimization disabled. The host validates scalar and table-function
identities from DuckDB's structured serialized SQL tree, rejecting dynamic and other
non-allowlisted table functions before expansion while recursively validating stored
table macros and referenced views. Relation references are resolved against their actual
lexical CTE scope, including ordered and recursive definitions, so nested names cannot
suppress unrelated view validation. Stored view bodies beginning with either `SELECT` or
`WITH` pass through the same structured parser. This preserves function calls
independently of projection aliases and distinguishes calls from string literals. The
pre-optimization bound plan separately enforces the table scan allowlist. Together these
checks reject writes, multiple statements, external scans, host metadata access,
side-effecting functions, and stored definitions that expand to those operations. Normal
optimization is restored before an accepted query executes.

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
messages. When a value cannot be represented as finite JSON, such as a non-finite
floating-point value, the complete table is still retained as Parquet and the preview
is omitted.

If the complete table exceeds query or artifact limits, the call fails with guidance
to filter or aggregate further. No partial artifact is published and no truncated
table is represented as complete.

## Artifact boundary

Artifact handles are sequential within one run. Publication copies bytes to private
temporary files, flushes them, and publishes without overwriting. A multi-output
Python call publishes its complete batch or rolls the complete batch back. Only after
publication succeeds does the store record byte size, SHA-256 digest, media type,
managed relative path, and producing tool call.

Every consumer opens a retained artifact once, verifies the size and digest from that
descriptor, and consumes or copies the bytes from the same descriptor. Tabular
artifacts use Parquet. Declared Parquet output validation reads every data page, not
only footer metadata. Final-answer artifacts use finite UTF-8 JSON.

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
the complete declared output set transactionally through the same artifact boundary.
Each executor output is first copied to host-private artifact staging. The staged bytes
are fully validated, then those same staged inodes are published as one batch. Milestone
3 will implement this protocol with Docker and a run-private database copy.

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
integrity and replacement races, transactional multi-output rollback, complete
Parquet validation, executor-output replacement races, ambiguous quoted relation names,
constant-folded, aliased, macro-expanded, and view-contained forbidden SQL calls,
function-like string literals, dynamic table functions, safe table macros, exact
top-level query syntax, lexical nested CTEs, `WITH`-based stored views, streamed catalog
and column limits, bounded targeted catalog lookup, managed Python inputs and outputs,
direct and artifact-backed final output, per-result and cumulative
retry-feedback limits, terminal-record safety, and all Milestone 1 behavior.

Docker, writable database copies, rollback, Databricks MLflow, live providers, a CLI,
and a conversational demo remain outside this milestone.

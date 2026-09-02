# Milestone 4: opt-in MLflow observability

## Objective

Any analysis run may opt into MLflow observability through the operator-level
`report_to_mlflow=True` argument. Reporting is a projection of an already canonical DSA
run: it never changes the request seen by the model, the analysis outcome, or the retained
`terminal.json` bytes.

The same reporting path supports one small native MLflow Evaluation Dataset and exact
scoring slice. Observability is therefore useful during ordinary runs as well as during
evaluation, without turning evaluation concerns into agent inputs.

## Runtime boundary

MLflow support is an optional `dsa[mlflow]` dependency pinned to the version exercised by
this milestone. Disabled runs do not require MLflow. An enabled run reads only these
operator-owned environment variables:

- `MLFLOW_TRACKING_URI`, either exactly `databricks`, a loopback HTTP tracking server,
  or an HTTPS tracking server
- `MLFLOW_EXPERIMENT_ID`
- `DATABRICKS_HOST`, required only for the Databricks backend
- `DATABRICKS_TOKEN`, required only for the Databricks backend

There is no implicit experiment creation or fallback tracking destination. Direct
filesystem and SQLite tracking URIs are rejected: local operation goes through an HTTP
tracking server so concurrent benchmark workers share one service boundary. Plain HTTP
is accepted only for `localhost`, `127.0.0.1`, or `::1`; remote tracking servers require
HTTPS. Missing dependencies, invalid configuration, backend unavailability, and export
failures produce a stable safe reporting failure code. They do not convert a completed
DSA analysis success into an analysis failure.

Pydantic AI autologging is initialized once per process. Every analysis enters an
async-context-local MLflow tracing policy from its beginning, including disabled runs, so
concurrent enabled and disabled runs cannot change one another's tracing choice. Enabled
runs create one explicit tracking run through `MlflowClient`, use an experiment-local root
trace span, and allow Pydantic AI's native instrumentation to create its model, agent, and
tool child spans. The implementation does not use MLflow's process-global active-run API.

## Reporting projection

`RunCompletion` includes one operator-visible reporting result:

- `disabled`
- `reported`, with the MLflow tracking run ID and trace ID
- `failed`, with a stable safe failure code

This result is deliberately absent from `RunRequest`, the model prompt and tools,
`RunPolicy`, and `TerminalRecord`. The canonical local terminal file remains the source of
truth even when MLflow reporting fails.

An enabled successful export includes:

- the native Pydantic AI trace under one DSA root span
- the exact retained `terminal.json` as an MLflow run artifact
- a canonical JSON artifact-manifest projection containing metadata only
- bounded safe run tags and numeric metrics

It never uploads retained artifact contents, the DuckDB database, the private run
workspace, provider credentials, backend credentials, or raw reporting exceptions. The
flag constitutes an explicit operator choice to send the trace and terminal record to the
configured MLflow destination.

## Native evaluation slice

An evaluation case contains a normal `RunRequest` plus a host-only expected answer. The
expectation is validated against the caller's answer schema and copied into the native
MLflow Evaluation Dataset `expectations` field. Only the dataset `inputs` are passed to the
prediction function. Expectations never enter the DSA request, model context, tools,
Docker environment, or terminal record.

The evaluation invokes the ordinary `run_analysis(..., report_to_mlflow=True)` path and
uses six explicit scorers:

- `end_to_end_exact_success`: false for every non-exact result and every failure; this is
  the primary all-case denominator
- `conditional_exact_json`: exact canonical JSON equality for accepted answers and invalid
  otherwise
- `end_to_end_policy_success`: accepted, successfully reported answers matching the
  pack's declared comparison policy
- `conditional_policy_match`: policy matches among accepted answers and invalid otherwise
- `agent_failure`: true only for a valid model or analysis failure under the pack policy
- `infrastructure_failure`: true for provider, host-runtime, cancellation, or MLflow
  reporting failure

Pack-backed evaluation transports the expected answer and comparison policy as canonical
JSON strings inside `expectations`, preventing a dataset transport from changing integer JSON
leaves into floating-point values. The default policy remains exact canonical JSON. An
explicit numeric-tolerance policy applies only to expected floating-point leaves;
expected integers, booleans, strings, nulls, object keys, and array structure remain
exact. Policy tolerance never changes either exact scorer.

MLflow's prediction trace preflight is disabled only around the native evaluation call
and its prior environment setting is restored afterward. The DSA prediction already emits
an explicit trace, while the preflight would otherwise execute the first side-effecting
analysis twice. Native MLflow evaluation is itself documented as not thread-safe. Host
run-timeout, model-usage-limit, and tool-result-limit terminal codes count as
infrastructure failures alongside internal orchestration errors.

The live acceptance cases use a generated tiny DuckDB database and a deterministic
scripted Pydantic AI model. They prove native dataset creation, one evaluation, one DSA
tracking run in addition to the native evaluation run, the expected trace shape, exact
scoring, and exact terminal-record upload. Each backend test is opt-in and is reported as
skipped when its explicit environment is unavailable.

## Acceptance evidence

Deterministic tests cover strict flag validation before side effects, request and terminal
isolation, reporting-result invariants, stable safe reporting failures, canonical terminal
and manifest projection, concurrency-local tracing choice, expectation separation, exact
JSON comparison, and failure classification. Existing tests continue to prove the local
canonical lifecycle.

The separately invoked Databricks and local-server tiers are the evidence for their real
backend boundaries. A skipped live test is not reported as backend evidence. The Pydantic
AI trace is inspected explicitly because the MLflow integration's published compatibility
range may lag the pinned Pydantic AI version.

Hugging Face evaluation packs, Online Retail II, live model providers, benchmark matrices,
CLI orchestration, and custom tracing instrumentation remain outside this milestone.

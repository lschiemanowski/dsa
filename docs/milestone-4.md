# Milestone 4: opt-in Databricks MLflow observability

## Objective

Any analysis run may opt into Databricks MLflow observability through the operator-level
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

- `MLFLOW_TRACKING_URI`, which must be exactly `databricks`
- `MLFLOW_EXPERIMENT_ID`
- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`

There is no local tracking server, local MLflow database, implicit experiment creation,
or fallback tracking destination. Missing dependencies, invalid configuration, Databricks
unavailability, and export failures produce a stable safe reporting failure code. They do
not convert a completed DSA analysis success into an analysis failure.

Pydantic AI autologging is initialized once per process. Every analysis enters an
async-context-local MLflow tracing policy from its beginning, including disabled runs, so
concurrent enabled and disabled runs cannot change one another's tracing choice. Enabled
runs create one explicit tracking run through `MlflowClient`, use an experiment-local root
trace span, and allow Pydantic AI's native instrumentation to create its model, agent, and
tool child spans. The implementation does not use MLflow's process-global active-run API.

## Reporting projection

`RunCompletion` includes one operator-visible reporting result:

- `disabled`
- `reported`, with the Databricks tracking run ID and trace ID
- `failed`, with a stable safe failure code

This result is deliberately absent from `RunRequest`, the model prompt and tools,
`RunPolicy`, and `TerminalRecord`. The canonical local terminal file remains the source of
truth even when Databricks reporting fails.

An enabled successful export includes:

- the native Pydantic AI trace under one DSA root span
- the exact retained `terminal.json` as an MLflow run artifact
- a canonical JSON artifact-manifest projection containing metadata only
- bounded safe run tags and numeric metrics

It never uploads retained artifact contents, the DuckDB database, the private run
workspace, provider credentials, Databricks credentials, or raw reporting exceptions.
The flag constitutes an explicit operator choice to send the trace and terminal record to
the configured Databricks workspace.

## Native evaluation slice

An evaluation case contains a normal `RunRequest` plus a host-only expected answer. The
expectation is validated against the caller's answer schema and copied into the native
MLflow Evaluation Dataset `expectations` field. Only the dataset `inputs` are passed to the
prediction function. Expectations never enter the DSA request, model context, tools,
Docker environment, or terminal record.

The evaluation invokes the ordinary `run_analysis(..., report_to_mlflow=True)` path and
uses four explicit scorers:

- `end_to_end_exact_success`: false for every non-exact result and every failure; this is
  the primary all-case denominator
- `conditional_exact_json`: exact canonical JSON equality for accepted answers and invalid
  otherwise
- `agent_failure`: true only for a valid model or analysis failure
- `infrastructure_failure`: true for provider, host-runtime, cancellation, or MLflow
  reporting failure

The live acceptance case uses a generated tiny DuckDB database and a deterministic
scripted Pydantic AI model. It proves native Databricks dataset creation, one evaluation,
one DSA tracking run in addition to the native evaluation run, the expected trace shape,
exact scoring, and exact terminal-record upload. It is opt-in and is reported as skipped
when credentials are unavailable.

## Acceptance evidence

Deterministic tests cover strict flag validation before side effects, request and terminal
isolation, reporting-result invariants, stable safe reporting failures, canonical terminal
and manifest projection, concurrency-local tracing choice, expectation separation, exact
JSON comparison, and failure classification. Existing tests continue to prove the local
canonical lifecycle.

The separately invoked Databricks tier is the only evidence for the real remote boundary.
A skipped live test is not reported as Databricks evidence. The Pydantic AI trace is
inspected explicitly because the MLflow integration's published compatibility range may
lag the pinned Pydantic AI version.

Hugging Face evaluation packs, Online Retail II, live model providers, benchmark matrices,
CLI orchestration, local MLflow backends, and custom tracing instrumentation remain outside
this milestone.

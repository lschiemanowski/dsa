# Milestone 6A: isolated benchmark orchestration

## Objective

Add a thin benchmark runner that expands one immutable study into independent
pack-model-repetition cells, executes each cell through the existing native Databricks
MLflow evaluation path, and retains only the correlation needed for safe resume and
later read-only publication.

This milestone does not add another evaluation engine. Pack loading remains owned by
the Hugging Face boundary, case execution remains owned by `run_analysis`, deterministic
assessment remains owned by MLflow GenAI Evaluation, and model-authored Python remains
owned by the digest-pinned Docker executor.

## Portable study contract

`dsa-benchmark-study/v1` contains a safe study identity and semantic version, an exact
agent Git revision, one or more ordered exact Hugging Face pack references, one or more
ordered model configurations, one shared run policy, one digest-pinned Docker image,
explicit MLflow case and benchmark-cell concurrency, and a positive repetition count.

The canonical compact recursively key-sorted JSON representation is the study identity.
Changing a pack, model setting, policy, image, concurrency value, repetition count, or
agent revision therefore creates a different study digest and different cell identities.
Typed study values are revalidated and isolated at every execution boundary.

One cell is the tuple of study digest, pack identity, model identity, and zero-based
repetition index. Cells are ordered by pack identity, model identity, then repetition.
Their IDs are derived from that canonical tuple rather than host paths or remote run IDs.

## Runtime and secret boundary

`dsa-benchmark-runtime/v1` contains only a new workspace root and an exact mapping from
each study pack identity to its Databricks MLflow dataset name. Databricks credentials,
experiment identity, provider credentials, provider endpoints, and Hugging Face cache
locations remain process environment or library-owned state and never enter study,
cell, or receipt identity.

Before any model call or MLflow write, preflight revalidates both contracts, verifies the
current agent revision, resolves and verifies every exact pack, requires exact runtime
coverage, confirms the digest-pinned Docker image, validates Databricks configuration,
confirms the tracked agent implementation is clean at that revision, and derives the
complete cell plan. A failed preflight starts no cells.

## Execution lifecycle

Each cell runs in a fresh Python subprocess with one private workspace. The worker sets
MLflow's internal prediction concurrency explicitly, constructs the existing Docker
executor from the exact image, and invokes `run_mlflow_evaluation` once for the complete
pack. The parent bounds simultaneous cell subprocesses separately. Both concurrency
values default to one and are retained in study identity.

The worker accepts one canonical payload on standard input and emits one framed strict
result on standard output. Other output is bounded diagnostic material. Cancellation
terminates, force-kills when necessary, and reaps workers without deleting completed run
records. Model failures, wrong accepted answers, policy limits, and provider failures
inside a returned evaluation are benchmark observations rather than worker crashes and
are never retried automatically.

Every attempt has a new private directory. A successful evaluation atomically publishes
one no-overwrite canonical `cell.json` containing study and cell identities, the exact
pack/model/repetition/environment identities, MLflow dataset and evaluation run IDs, and
the returned per-case run and terminal-record identities. It contains no expectations,
credentials, endpoint, or raw backend diagnostics.

Explicit resume skips only a completely verified completed receipt. Existing incomplete
state is preserved. A remotely finished evaluation without a corresponding verified
local receipt is ambiguous and must not be rerun automatically.

## CLI and acceptance

`dsa-benchmark plan` performs validation and complete preflight, then emits canonical
JSON describing the study digest and ordered cells without model calls or MLflow writes.
`dsa-benchmark run` executes that same plan and accepts an explicit `--resume` flag.
Usage or preflight rejection exits two, incomplete execution exits one, complete or
safely resumed execution exits zero, and cancellation exits 130.

Deterministic tests cover strict models, canonical identity, mutation resistance, exact
coverage, zero-work preflight rejection, process framing, both concurrency bounds,
cancellation, atomic no-overwrite receipts, safe resume, and failure classification. A
real subprocess test covers the production framing boundary. Existing real Docker tests
continue to cover the sandbox lifecycle. One opt-in Databricks test may use Pydantic AI's
built-in deterministic test model without paid provider calls.

Read-only aggregate publication, Markdown tables, cost estimation, difficulty
aggregation, additional released packs, paid model runs, automatic publication, local
MLflow, and user interfaces remain outside Milestone 6A.

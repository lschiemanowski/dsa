# Milestone 3: run lifecycle and isolated Python

## Objective

Every valid run operates on one private writable DuckDB copy. Model-authored Python
executes only in a digest-pinned Docker image. Each Python call either commits its
database mutation and complete declared artifact batch or leaves no visible change.

The source database is never mounted into a container and is never reopened after the
private copy is established. SQL and later Python calls observe the current private
database state. The terminal record remains canonical run state.

## Run workspace

A valid source must be a regular DuckDB file without an accompanying write-ahead log.
The host copies it once through one open descriptor into a private `0700` workspace and
calculates the source SHA-256 from the exact copied bytes. The copy is flushed before it
becomes the run database and is opened through the existing locked-down DuckDB boundary.

The run layout is:

```text
<run-id>/
  work/
    database.duckdb
    attempts/
  artifacts/
  terminal.json
```

All analysis tools share one per-run lock. Inspection and query open only the current
working database read-only. Python holds the same lock while preparing and committing a
mutable attempt, so no query can observe a partially committed state.

The terminal record retains the source and final working-database SHA-256 identities.
After the final digest is calculated, `work/` is removed by default before terminal
publication. An operator-only `keep_workdir` argument may retain it for debugging. That
argument is not part of the public request, model context, or `RunPolicy`.

## Transactional Python calls

Each call receives a fresh private attempt directory containing:

- a copy of the current working database in its own writable mount directory
- read-only copies of only the selected retained input artifacts
- a fresh output staging directory

The container receives fixed locators through the environment-variable names
`DSAGENT_DATABASE`, `DSAGENT_INPUTS`, and `DSAGENT_OUTPUTS`; model code resolves their
values through `os.environ`. The database directory is isolated from the input and
output directories so DuckDB can manage its private write-ahead log without exposing a
broader host directory. Source code arrives on standard input rather than through a host
mount. The declared output list may be empty for a database-only call.

After the container has stopped and been removed, the host validates the complete attempt
database and the complete declared output set. Artifact publication and replacement of
the working database form one host-owned logical commit while the tool lock is held. If
any validation, publication, replacement, or cleanup step fails, the attempt database
and staged outputs are discarded and the prior working database remains current.

Successful mutations persist for later SQL and Python calls in the same run. Timeout,
out-of-memory termination, nonzero exit, invalid or undeclared output, database
corruption, artifact-limit failure, and container-cleanup failure all roll back.

## Docker boundary

`DockerPythonExecutor` is the production implementation of the Milestone 2 executor
protocol. Its operator-owned configuration requires an immutable image reference pinned
by SHA-256. It never pulls or builds during a run.

Every invocation uses a fresh container with:

- a non-root UID and GID
- no network
- a read-only root filesystem
- all Linux capabilities dropped
- `no-new-privileges`
- no host PID namespace and a private IPC namespace
- explicit memory, CPU, process, elapsed-time, and tmpfs limits
- deterministic single-thread analytical-library environment variables
- only the attempt database, selected inputs, and output staging mounts
- no checkout, home directory, credentials, Docker socket, or inherited container
  environment

The host drains stdout and stderr while retaining only bounded prefixes. Timeout kills
the container. OOM state and nonzero exit are classified separately. Container removal
is mandatory before an invocation can succeed. The Docker server, configured image
reference, and inspected image identity are retained as safe runtime evidence.

The repository image recipe uses a digest-pinned Python 3.12 base, a fully pinned
analytical package set derived from the proven prototype environment, and a non-root
runtime user.

## Policy

The existing flat `RunPolicy` supplies Python time, memory, output, scratch, and artifact
limits. Milestone 3 adds positive CPU and process limits. Every effective limit is
enforced by the host or Docker runtime and retained in the terminal request snapshot.

Docker executable location, image identity, container UID and GID, control-operation
timeout, and working-directory retention are operator configuration rather than task
inputs.

## Acceptance evidence

Deterministic tests cover source copying and hashing, WAL rejection, source immutability,
tool serialization, successful mutation persistence, rollback for every failure class,
logical database-and-artifact commit, terminal digests, default cleanup, debugging
retention, strict Docker configuration, exact shell-free Docker arguments, bounded
diagnostics, timeout killing, OOM classification, mandatory removal, and all earlier
milestone behavior.

A separately invoked real-Docker tier proves non-root execution, network denial, narrow
mount visibility, read-only selected inputs, writable private database and outputs,
successful mutation persistence, failed mutation rollback, resource enforcement, source
immutability, and container cleanup. A skipped integration test is not reported as real
Docker evidence.

Databricks MLflow, Hugging Face packs, live providers, prototype revision graphs, a
trusted host-Python backend, CLI orchestration, and benchmark comparisons remain outside
this milestone.

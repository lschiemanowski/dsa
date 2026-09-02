# Local MLflow for DSA benchmarks

DSA can use the pinned MLflow tracking server as the complete observability and evaluation
backend. Model inference, DuckDB analysis, Docker execution, MLflow state, artifacts, and
the UI can therefore remain on one machine.

## Start the server

Install the pinned optional backend and create persistent state once:

```bash
cd /home/lothar/workspace/dsa
UV_CACHE_DIR=/tmp/dsa-benchmark-uv-cache uv sync --extra mlflow --frozen
mkdir -p /home/lothar/workspace/dsa/.mlflow/artifacts
```

Keep this command running in its own terminal:

```bash
cd /home/lothar/workspace/dsa
UV_CACHE_DIR=/tmp/dsa-benchmark-uv-cache uv run mlflow server \
  --backend-store-uri sqlite:////home/lothar/workspace/dsa/.mlflow/mlflow.db \
  --artifacts-destination /home/lothar/workspace/dsa/.mlflow/artifacts \
  --host 127.0.0.1 \
  --port 5000 \
  --workers 1
```

The single worker matches SQLite's intended local use. The server remains loopback-only;
do not bind it to `0.0.0.0` without adding authentication and explicit MLflow host/CORS
controls.

## Configure the benchmark shell

In the terminal used for `dsa-benchmark`, select the server and create or reuse one
experiment:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"

export MLFLOW_EXPERIMENT_ID="$(
  UV_CACHE_DIR=/tmp/dsa-benchmark-uv-cache uv run python - <<'PY'
from mlflow import MlflowClient

client = MlflowClient()
name = "dsa-benchmarks"
existing = client.get_experiment_by_name(name)
print(existing.experiment_id if existing else client.create_experiment(name))
PY
)"
```

`DATABRICKS_HOST` and `DATABRICKS_TOKEN` are not used by this backend. Existing model
provider variables, such as a local `OPENAI_BASE_URL`, remain unchanged.

Use a new safe dataset name in the benchmark runtime, for example:

```json
{"datasets":[{"dataset_name":"online_retail_ii_qwen35_4b_local_v1","pack_id":"online-retail-ii"}],"format":"dsa-benchmark-runtime/v1","workspace_root":"/absolute/new/workspace"}
```

Runtime JSON must still use canonical compact bytes. Run `dsa-benchmark plan`, `run`, and
`report` exactly as for Databricks. Reports retain and verify the same dataset digests,
completed MLflow runs, scoring metrics, terminal digests, and local terminal records.

Open <http://127.0.0.1:5000> to inspect experiments, evaluation runs, traces, and artifacts.

## Acceptance test

With the server running and the environment above configured, the opt-in live boundary is:

```bash
DSA_LOCAL_MLFLOW_TEST=1 \
UV_CACHE_DIR=/tmp/dsa-benchmark-uv-cache \
uv run pytest tests/test_mlflow_local.py
```

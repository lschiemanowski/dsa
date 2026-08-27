"""Opt-in acceptance for the complete benchmark-to-Databricks boundary."""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from dsa import (
    BenchmarkRuntime,
    BenchmarkStudy,
    HuggingFacePackReference,
    read_benchmark_cell_receipt,
)
from dsa.benchmark import prepare_benchmark, run_prepared_benchmark


def _benchmark_databricks_acceptance_enabled() -> bool:
    required = (
        "MLFLOW_EXPERIMENT_ID",
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
        "DSA_MLFLOW_DATASET_NAME",
        "DSA_DOCKER_TEST_IMAGE",
    )
    return (
        os.environ.get("DSA_BENCHMARK_DATABRICKS_TEST") == "1"
        and os.environ.get("MLFLOW_TRACKING_URI") == "databricks"
        and all(os.environ.get(key, "").strip() for key in required)
    )


@pytest.mark.integration
@pytest.mark.databricks
@pytest.mark.skipif(
    not _benchmark_databricks_acceptance_enabled(),
    reason="set DSA_BENCHMARK_DATABRICKS_TEST=1 and the benchmark environment",
)
def test_public_pack_benchmark_cell_reaches_databricks_once(
    tmp_path: Path,
) -> None:
    from mlflow import MlflowClient

    repository = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        timeout=5,
    ).stdout.decode().strip()
    reference = HuggingFacePackReference.model_validate_json(
        (repository / "evaluation-packs" / "online-retail-ii-1.0.0.json").read_bytes()
    )
    study = BenchmarkStudy.model_validate(
        {
            "format": "dsa-benchmark-study/v1",
            "study_id": "online-retail-databricks-acceptance",
            "version": "1.0.0",
            "agent_revision": revision,
            "packs": ({"pack_id": "online-retail-ii", "reference": reference},),
            "models": (
                {
                    "model_id": "pydantic-test",
                    "configuration": {"name": "test", "settings": {}},
                },
            ),
            "policy": {},
            "execution": {
                "docker_image": os.environ["DSA_DOCKER_TEST_IMAGE"],
                "case_workers": 1,
                "cell_workers": 1,
            },
            "repetitions": 1,
        }
    )
    runtime = BenchmarkRuntime.model_validate(
        {
            "format": "dsa-benchmark-runtime/v1",
            "workspace_root": tmp_path / "benchmark",
            "datasets": (
                {
                    "pack_id": "online-retail-ii",
                    "dataset_name": os.environ["DSA_MLFLOW_DATASET_NAME"],
                },
            ),
        }
    )

    prepared = asyncio.run(
        prepare_benchmark(study, runtime, current_revision=revision)
    )
    results = asyncio.run(run_prepared_benchmark(prepared))

    assert len(results) == 1
    assert results[0].status == "completed"
    receipt = read_benchmark_cell_receipt(
        runtime.workspace_root / prepared.cells[0].cell_id / "cell.json"
    )
    assert len(receipt.predictions) == prepared.packs[0][1].manifest.cases.case_count
    run: Any = MlflowClient(tracking_uri="databricks").get_run(receipt.evaluation_run_id)
    tags = cast(dict[str, str], run.data.tags)
    assert tags["dsa.benchmark.cell_id"] == receipt.cell_id
    assert tags["dsa.benchmark.study_sha256"] == receipt.study_sha256

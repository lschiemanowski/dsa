"""Opt-in acceptance test for the real local MLflow server boundary."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic_ai.models.test import TestModel

from dsa.benchmark_report import MlflowBenchmarkEvidenceReader
from dsa.evaluation import run_mlflow_evaluation
from dsa.pack import EvaluationPackCase

from .test_episode import valid_request
from .test_mlflow_evaluation import case_metadata, evaluation_pack


def _local_mlflow_acceptance_enabled() -> bool:
    return (
        os.environ.get("DSA_LOCAL_MLFLOW_TEST") == "1"
        and os.environ.get("MLFLOW_TRACKING_URI", "").startswith(
            ("http://127.0.0.1:", "http://localhost:", "http://[::1]:")
        )
        and bool(os.environ.get("MLFLOW_EXPERIMENT_ID", "").strip())
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not _local_mlflow_acceptance_enabled(),
    reason="set DSA_LOCAL_MLFLOW_TEST=1 and the local MLflow environment",
)
def test_tiny_native_local_evaluation_and_evidence(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    datasets = pytest.importorskip("mlflow.genai.datasets")
    request = valid_request(tmp_path)
    case = EvaluationPackCase(
        case_id="local-mlflow-count",
        case_version="1",
        question=request.question,
        answer_schema=request.answer_schema,
        expected_answer={"count": 3},
        metadata=case_metadata(),
    )

    result = run_mlflow_evaluation(
        evaluation_pack(tmp_path, case),
        dataset_name=f"dsa_local_acceptance_{uuid4().hex}",
        runs_directory=tmp_path / "runs",
        model_configuration=request.model,
        policy=request.policy,
        model_factory=lambda _case: TestModel(
            call_tools=[], custom_output_args={"count": 3}
        ),
    )

    assert result.evaluation_run_id is not None
    assert len(result.predictions) == 1
    prediction = result.predictions[0]
    assert prediction.reporting.status == "reported"
    assert datasets.get_dataset(dataset_id=result.dataset_id).digest == result.dataset_digest
    client = mlflow.MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    reader = MlflowBenchmarkEvidenceReader(client)
    evaluation_run = reader.get_run(result.evaluation_run_id)
    assert evaluation_run.status == "FINISHED"
    assert evaluation_run.metrics == {
        "agent_failure/mean": 0.0,
        "conditional_exact_json/mean": 1.0,
        "conditional_policy_match/mean": 1.0,
        "end_to_end_exact_success/mean": 1.0,
        "end_to_end_policy_success/mean": 1.0,
        "infrastructure_failure/mean": 0.0,
    }
    assert prediction.reporting.tracking_run_id is not None
    assert reader.get_run(prediction.reporting.tracking_run_id).status == "FINISHED"
    artifacts = {
        item.path
        for item in client.list_artifacts(prediction.reporting.tracking_run_id, "dsa")
    }
    assert artifacts == {"dsa/artifacts.json", "dsa/terminal.json"}

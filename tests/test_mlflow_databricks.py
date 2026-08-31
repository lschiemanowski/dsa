from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any

import duckdb
import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from dsa import run_mlflow_evaluation
from dsa.pack import EvaluationPackCase

from .test_episode import valid_request
from .test_mlflow_evaluation import case_metadata, evaluation_pack


def _databricks_acceptance_enabled() -> bool:
    required = (
        "MLFLOW_EXPERIMENT_ID",
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
        "DSA_MLFLOW_DATASET_NAME",
    )
    return (
        os.environ.get("DSA_DATABRICKS_TEST") == "1"
        and os.environ.get("MLFLOW_TRACKING_URI") == "databricks"
        and all(os.environ.get(key, "").strip() for key in required)
    )


@pytest.mark.integration
@pytest.mark.databricks
@pytest.mark.skipif(
    not _databricks_acceptance_enabled(),
    reason="set DSA_DATABRICKS_TEST=1 and the Databricks MLflow environment",
)
def test_tiny_native_databricks_evaluation_and_pydantic_ai_trace(
    tmp_path: Path,
) -> None:
    from mlflow import MlflowClient

    request = valid_request(tmp_path)
    connection = duckdb.connect(str(request.database_path))
    connection.execute("create table events(value integer)")
    connection.execute("insert into events values (1), (2), (3)")
    connection.close()

    def model_factory(_case: EvaluationPackCase) -> FunctionModel:
        calls = 0

        async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "query_database",
                            {"sql": "select count(*) as count from events"},
                            "query-1",
                        )
                    ]
                )
            return ModelResponse(
                parts=[ToolCallPart("final_answer", {"count": 3}, "answer-1")]
            )

        return FunctionModel(respond, model_name="deterministic-databricks-test")

    case = EvaluationPackCase(
        case_id="tiny-event-count",
        case_version="1",
        question=request.question,
        answer_schema=request.answer_schema,
        expected_answer={"count": 3},
        metadata=case_metadata(),
    )
    pack = evaluation_pack(tmp_path, case)
    result = run_mlflow_evaluation(
        pack,
        dataset_name=os.environ["DSA_MLFLOW_DATASET_NAME"],
        runs_directory=tmp_path / "runs",
        model_configuration=request.model,
        policy=request.policy,
        model_factory=model_factory,
    )

    assert result.dataset_id
    assert result.dataset_digest
    assert result.evaluation_run_id
    assert len(result.predictions) == 1
    prediction = result.predictions[0]
    assert prediction.accepted is True
    assert prediction.answer == {"count": 3}
    assert prediction.reporting.status == "reported"
    assert result.metrics["end_to_end_exact_success/mean"] == 1.0
    assert result.metrics["conditional_exact_json/mean"] == 1.0
    assert result.metrics["end_to_end_policy_success/mean"] == 1.0
    assert result.metrics["conditional_policy_match/mean"] == 1.0
    assert result.metrics["agent_failure/mean"] == 0.0
    assert result.metrics["infrastructure_failure/mean"] == 0.0

    assert prediction.reporting.tracking_run_id is not None
    assert prediction.reporting.trace_id is not None
    client = MlflowClient(tracking_uri="databricks")
    terminal_path = Path(
        client.download_artifacts(
            prediction.reporting.tracking_run_id,
            "dsa/terminal.json",
            dst_path=str(tmp_path / "download"),
        )
    )
    terminal_bytes = terminal_path.read_bytes()
    assert sha256(terminal_bytes).hexdigest() == prediction.terminal_sha256
    terminal = json.loads(terminal_bytes)
    assert terminal["outcome"] == {"status": "succeeded", "answer": {"count": 3}}
    assert "reporting" not in terminal
    assert "expectations" not in terminal["request"]

    trace = client.get_trace(prediction.reporting.trace_id, flush=True)
    span_names = {span.name for span in trace.data.spans}
    span_types = {span.span_type for span in trace.data.spans}
    assert "dsa.run" in span_names
    assert "query_database" in span_names
    assert "TOOL" in span_types
    assert len(trace.data.spans) >= 3

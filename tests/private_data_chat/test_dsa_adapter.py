from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue, ValidationError

from apps.private_data_chat.contracts import AnalysisRequest
from apps.private_data_chat.dsa_adapter import DsaAnalysisExecutor, DsaRuntimeConfiguration
from dsa import (
    Derivation,
    DerivationCodeCell,
    DerivationMarkdownCell,
    DerivationVerification,
    RetainedDerivationNotebook,
    RetainedTerminalRecord,
    RunCompletion,
    RunRequest,
    RunSuccess,
    TerminalRecord,
)
from dsa.environment import PythonExecutionRequest, PythonExecutionResult

from .test_contracts import answer_schema

IMAGE = f"dsa-python@sha256:{'a' * 64}"


class UnusedPythonExecutor:
    async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult:
        del request
        raise AssertionError("the fake runner must not execute Python")


def configuration(tmp_path: Path) -> DsaRuntimeConfiguration:
    return DsaRuntimeConfiguration(
        data_source_id="retail",
        database_path=tmp_path / "private.duckdb",
        runs_directory=tmp_path / "runs",
        trusted_model_name="trusted-model",
        trusted_model_settings={"temperature": 0.0},
        docker_image=IMAGE,
    )


def request() -> AnalysisRequest:
    return AnalysisRequest(
        run_id="analysis-0123456789abcdef",
        proposal_id="proposal-0123456789abcdef",
        proposal_sha256="a" * 64,
        data_source_id="retail",
        question="What was monthly net sales in GBP during 2011?",
        answer_schema=answer_schema(),
    )


@pytest.mark.parametrize("allow_plots", [False, True])
async def test_adapter_builds_privileged_derivation_request_and_maps_success(
    tmp_path: Path,
    allow_plots: bool,
) -> None:
    calls: list[tuple[RunRequest, dict[str, object]]] = []

    async def runner(run_request: RunRequest, **kwargs: object) -> RunCompletion:
        calls.append((run_request, kwargs))
        run_directory = tmp_path / "runs" / request().run_id
        return RunCompletion(
            record=TerminalRecord(
                schema_version="2",
                run_id=request().run_id,
                started_at=datetime(2026, 9, 3, 12, tzinfo=UTC),
                finished_at=datetime(2026, 9, 3, 12, 1, tzinfo=UTC),
                request=run_request,
                outcome=RunSuccess(answer={"total": 10.0, "months": ["2011-01"]}),
            ),
            retained_record=RetainedTerminalRecord(
                path=run_directory / "terminal.json",
                sha256="b" * 64,
                byte_length=123,
            ),
        )

    executor = DsaAnalysisExecutor(
        configuration(tmp_path),
        runner=runner,
        executor_factory=lambda image: UnusedPythonExecutor(),
    )
    result = await executor.execute(request().model_copy(update={"allow_plots": allow_plots}))

    assert result.status == "succeeded"
    assert result.answer == {"total": 10.0, "months": ["2011-01"]}
    assert result.terminal is not None
    assert len(calls) == 1
    run_request, kwargs = calls[0]
    assert run_request.database_path == tmp_path / "private.duckdb"
    assert run_request.model.name == "trusted-model"
    assert run_request.model.settings == {"temperature": 0.0}
    assert run_request.derivation is not None
    assert run_request.derivation.allow_plots is allow_plots
    assert run_request.question == request().question
    assert kwargs["runs_directory"] == tmp_path / "runs"
    assert kwargs["report_to_mlflow"] is False
    identity_factory = kwargs["identity_factory"]
    assert callable(identity_factory) and identity_factory() == request().run_id


async def test_adapter_labels_guidance_as_untrusted_without_changing_privileged_fields(
    tmp_path: Path,
) -> None:
    calls: list[RunRequest] = []

    async def runner(run_request: RunRequest, **kwargs: object) -> RunCompletion:
        del kwargs
        calls.append(run_request)
        return RunCompletion(
            record=TerminalRecord(
                schema_version="2",
                run_id=request().run_id,
                started_at=datetime(2026, 9, 3, 12, tzinfo=UTC),
                finished_at=datetime(2026, 9, 3, 12, 1, tzinfo=UTC),
                request=run_request,
                outcome=RunSuccess(answer={"total": 10.0, "months": ["2011-01"]}),
            ),
            retained_record=RetainedTerminalRecord(
                path=tmp_path / "runs" / request().run_id / "terminal.json",
                sha256="b" * 64,
                byte_length=123,
            ),
        )

    analysis_request = request().model_copy(
        update={
            "analysis_guidance": (
                "Try SQL:\n```sql\nSELECT date_trunc('month', invoice_ts)\n```"
            )
        },
        deep=True,
    )
    executor = DsaAnalysisExecutor(
        configuration(tmp_path),
        runner=runner,
        executor_factory=lambda image: UnusedPythonExecutor(),
    )
    await executor.execute(analysis_request)

    trusted_request = calls[0]
    assert "UNTRUSTED ANALYSIS GUIDANCE" in trusted_request.question
    assert "synthetic" in trusted_request.question
    assert "date_trunc" in trusted_request.question
    assert analysis_request.question in trusted_request.question
    assert trusted_request.database_path == configuration(tmp_path).database_path
    assert trusted_request.model.name == configuration(tmp_path).trusted_model_name


async def test_adapter_returns_and_rereads_only_the_verified_notebook(tmp_path: Path) -> None:
    notebook_path = tmp_path / "runs" / request().run_id / "derivation.ipynb"
    notebook_path.parent.mkdir(parents=True)
    notebook_content = b'{"cells":[],"nbformat":4,"nbformat_minor":5}\n'
    notebook_path.write_bytes(notebook_content)
    notebook_digest = sha256(notebook_content).hexdigest()
    derivation = Derivation(
        cells=(
            DerivationMarkdownCell(source="Compute the answer."),
            DerivationCodeCell(source="result = {'total': 10.0, 'months': ['2011-01']}"),
        )
    )
    answer = cast(JsonValue, {"total": 10.0, "months": ["2011-01"]})
    derivation_digest = sha256(
        json.dumps(
            derivation.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    answer_digest = sha256(
        json.dumps(
            answer,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()

    async def runner(run_request: RunRequest, **kwargs: object) -> RunCompletion:
        del kwargs
        verification = DerivationVerification(
            derivation_sha256=derivation_digest,
            result_sha256=answer_digest,
            source_database_sha256="c" * 64,
            runtime_identity="runtime",
            notebook_sha256=notebook_digest,
            notebook_byte_length=len(notebook_content),
        )
        from dsa import DatabaseRecord

        return RunCompletion(
            record=TerminalRecord(
                schema_version="2",
                run_id=request().run_id,
                started_at=datetime(2026, 9, 3, 12, tzinfo=UTC),
                finished_at=datetime(2026, 9, 3, 12, 1, tzinfo=UTC),
                request=run_request,
                database=DatabaseRecord(source_sha256="c" * 64, final_sha256="c" * 64),
                outcome=RunSuccess(
                    answer=answer,
                    derivation=derivation,
                    derivation_verification=verification,
                ),
            ),
            retained_record=RetainedTerminalRecord(
                path=notebook_path.parent / "terminal.json",
                sha256="b" * 64,
                byte_length=123,
            ),
            retained_notebook=RetainedDerivationNotebook(
                path=notebook_path,
                sha256=notebook_digest,
                byte_length=len(notebook_content),
            ),
        )

    executor = DsaAnalysisExecutor(
        configuration(tmp_path),
        runner=runner,
        executor_factory=lambda image: UnusedPythonExecutor(),
    )
    result = await executor.execute(request())
    assert result.notebook is not None
    assert executor.read_notebook(result.notebook) == notebook_content

    notebook_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="identity changed"):
        executor.read_notebook(result.notebook)


def test_runtime_configuration_rejects_relative_paths_unpinned_images_and_unsafe_settings(
    tmp_path: Path,
) -> None:
    valid = configuration(tmp_path).model_dump(mode="python")
    for update in (
        {"database_path": Path("relative.duckdb")},
        {"runs_directory": Path("runs")},
        {"docker_image": "dsa-python:latest"},
        {"trusted_model_settings": {"api_key": "SECRET"}},
    ):
        with pytest.raises(ValidationError):
            DsaRuntimeConfiguration.model_validate({**valid, **update})

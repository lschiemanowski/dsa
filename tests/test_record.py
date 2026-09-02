from __future__ import annotations

import json
import stat
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from threading import Barrier

import pytest
from pydantic import ValidationError

from dsa import (
    ArtifactRecord,
    DatabaseRecord,
    DerivationVerification,
    Failure,
    RetainedTerminalRecord,
    RunFailure,
    RunRequest,
    RunSuccess,
    TerminalRecord,
    write_terminal_record,
)

from .test_contract import request_value
from .test_derivation import sample_derivation


def request(tmp_path: Path) -> RunRequest:
    return RunRequest.model_validate(request_value(tmp_path / "source.duckdb"))


def terminal_record(tmp_path: Path) -> TerminalRecord:
    return TerminalRecord(
        run_id="run-001",
        started_at=datetime(2026, 8, 26, 8, 0, tzinfo=UTC),
        finished_at=datetime(2026, 8, 26, 8, 1, tzinfo=UTC),
        request=request(tmp_path),
        messages=(
            {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": "q"}]},
            {"kind": "response", "parts": [{"part_kind": "tool-call", "args": {}}]},
        ),
        usage={"requests": 1, "input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        outcome=RunSuccess(answer={"count": 3}),
    )


def test_terminal_record_contains_exactly_one_answer_or_failure(tmp_path: Path) -> None:
    success = terminal_record(tmp_path).model_dump(mode="json")
    failure = terminal_record(tmp_path).model_copy(
        update={
            "outcome": RunFailure(
                failure=Failure(
                    stage="answer_validation",
                    code="attempts_exhausted",
                    message="The model did not produce a valid answer",
                    diagnostics={"attempts": 3},
                )
            )
        }
    ).model_dump(mode="json")

    assert success["outcome"] == {"status": "succeeded", "answer": {"count": 3}}
    assert "failure" not in success["outcome"]
    assert failure["outcome"]["status"] == "failed"
    assert "answer" not in failure["outcome"]
    assert failure["outcome"]["failure"]["stage"] == "answer_validation"


def test_terminal_record_retains_artifact_metadata_without_artifact_bytes(
    tmp_path: Path,
) -> None:
    record = terminal_record(tmp_path).model_copy(
        update={
            "artifacts": (
                ArtifactRecord(
                    handle="a1",
                    relative_path="artifacts/a1.parquet",
                    media_type="application/vnd.apache.parquet",
                    size_bytes=1024,
                    sha256="a" * 64,
                    producer_tool_call_id="query-1",
                ),
            )
        }
    )

    dumped = record.model_dump(mode="json")

    assert dumped["artifacts"] == [
        {
            "handle": "a1",
            "relative_path": "artifacts/a1.parquet",
            "media_type": "application/vnd.apache.parquet",
            "size_bytes": 1024,
            "sha256": "a" * 64,
            "producer_tool_call_id": "query-1",
        }
    ]


def test_terminal_record_independently_rejects_invalid_success_answer(tmp_path: Path) -> None:
    raw = terminal_record(tmp_path).model_dump()
    raw["outcome"] = {"status": "succeeded", "answer": {"count": -1}}

    with pytest.raises(ValidationError, match="violates the caller schema"):
        TerminalRecord.model_validate(raw)


def test_terminal_record_requires_derivation_exactly_when_requested(
    tmp_path: Path,
) -> None:
    derivation = sample_derivation()
    request_raw = request_value(tmp_path / "source.duckdb")
    request_raw["derivation"] = {"format": "dsa-derivation/v1"}
    selected_request = RunRequest.model_validate(request_raw)
    derivation_bytes = json.dumps(
        derivation.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    answer_bytes = b'{"count":3}'
    source_digest = "a" * 64
    verification = DerivationVerification(
        derivation_sha256=sha256(derivation_bytes).hexdigest(),
        result_sha256=sha256(answer_bytes).hexdigest(),
        source_database_sha256=source_digest,
        runtime_identity="docker/test",
        notebook_sha256="b" * 64,
        notebook_byte_length=100,
    )
    derived = TerminalRecord(
        schema_version="2",
        run_id="run-derived",
        started_at=datetime(2026, 8, 26, 8, 0, tzinfo=UTC),
        finished_at=datetime(2026, 8, 26, 8, 1, tzinfo=UTC),
        request=selected_request,
        database=DatabaseRecord(
            source_sha256=source_digest,
            final_sha256=source_digest,
        ),
        outcome=RunSuccess(
            answer={"count": 3},
            derivation=derivation,
            derivation_verification=verification,
        ),
    )

    assert derived.schema_version == "2"
    with pytest.raises(ValidationError, match="schema version"):
        TerminalRecord.model_validate(
            {**derived.model_dump(mode="python"), "schema_version": "1"}
        )
    with pytest.raises(ValidationError, match="must match the request contract"):
        TerminalRecord.model_validate(
            {
                **derived.model_dump(mode="python"),
                "outcome": {"status": "succeeded", "answer": {"count": 3}},
            }
        )
    with pytest.raises(ValidationError, match="digest contradicts"):
        TerminalRecord.model_validate(
            {
                **derived.model_dump(mode="python"),
                "outcome": {
                    **derived.outcome.model_dump(mode="python"),
                    "derivation_verification": {
                        **verification.model_dump(mode="python"),
                        "derivation_sha256": "0" * 64,
                    },
                },
            }
        )


def test_answer_only_terminal_rejects_unrequested_derivation(tmp_path: Path) -> None:
    raw = terminal_record(tmp_path).model_dump(mode="python")
    raw["outcome"] = {
        "status": "succeeded",
        "answer": {"count": 3},
        "derivation": sample_derivation().model_dump(mode="python"),
    }

    with pytest.raises(ValidationError, match="must match the request contract"):
        TerminalRecord.model_validate(raw)

    raw = terminal_record(tmp_path).model_dump(mode="python")
    raw["schema_version"] = "2"
    with pytest.raises(ValidationError, match="schema version"):
        TerminalRecord.model_validate(raw)


def test_terminal_record_is_written_atomically_with_digest_and_private_mode(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run-001"
    run_directory.mkdir(mode=0o700)

    retained = write_terminal_record(terminal_record(tmp_path), run_directory)
    content = retained.path.read_bytes()

    assert retained.path == run_directory / "terminal.json"
    assert retained.sha256 == sha256(content).hexdigest()
    assert retained.byte_length == len(content)
    assert content.endswith(b"\n")
    assert json.loads(content)["run_id"] == "run-001"
    assert stat.S_IMODE(retained.path.stat().st_mode) == 0o600
    assert list(run_directory.iterdir()) == [retained.path]


def test_equivalent_record_mappings_have_identical_retained_bytes(tmp_path: Path) -> None:
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first_directory.mkdir()
    second_directory.mkdir()
    first = terminal_record(tmp_path)
    second = first.model_copy(
        update={
            "usage": {
                "total_tokens": 12,
                "output_tokens": 2,
                "input_tokens": 10,
                "requests": 1,
            }
        }
    )

    first_retained = write_terminal_record(first, first_directory)
    second_retained = write_terminal_record(second, second_directory)

    assert first_retained.sha256 == second_retained.sha256
    assert first_retained.path.read_bytes() == second_retained.path.read_bytes()


def test_failed_atomic_publication_leaves_no_terminal_or_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = tmp_path / "run-001"
    run_directory.mkdir()

    def fail_link(source: Path, destination: Path) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr("dsa.record.os.link", fail_link)

    with pytest.raises(OSError, match="simulated"):
        write_terminal_record(terminal_record(tmp_path), run_directory)

    assert list(run_directory.iterdir()) == []


def test_concurrent_writers_publish_exactly_one_matching_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = tmp_path / "run-001"
    run_directory.mkdir()
    destination = run_directory / "terminal.json"
    first = terminal_record(tmp_path)
    second = first.model_copy(update={"usage": {"requests": 2}})
    exists_barrier = Barrier(2)
    original_exists = Path.exists

    def synchronized_exists(path: Path) -> bool:
        result = original_exists(path)
        if path == destination:
            exists_barrier.wait()
        return result

    monkeypatch.setattr(Path, "exists", synchronized_exists)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures: list[Future[RetainedTerminalRecord]] = [
            executor.submit(write_terminal_record, record, run_directory)
            for record in (first, second)
        ]

    retained: list[RetainedTerminalRecord] = []
    errors: list[BaseException] = []
    for future in futures:
        try:
            retained.append(future.result())
        except BaseException as error:
            errors.append(error)

    assert len(retained) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], FileExistsError)
    content = destination.read_bytes()
    assert retained[0].sha256 == sha256(content).hexdigest()
    assert retained[0].byte_length == len(content)
    assert list(run_directory.iterdir()) == [destination]


@pytest.mark.parametrize("run_id", ["../escape", "contains/slash", "", " two"])
def test_run_identity_cannot_escape_its_managed_directory(
    tmp_path: Path,
    run_id: str,
) -> None:
    with pytest.raises(ValidationError):
        TerminalRecord.model_validate(
            {**terminal_record(tmp_path).model_dump(), "run_id": run_id}
        )

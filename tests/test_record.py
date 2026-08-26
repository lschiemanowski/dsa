from __future__ import annotations

import json
import stat
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from dsa import (
    Failure,
    RunFailure,
    RunRequest,
    RunSuccess,
    TerminalRecord,
    write_terminal_record,
)

from .test_contract import request_value


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


def test_terminal_record_independently_rejects_invalid_success_answer(tmp_path: Path) -> None:
    raw = terminal_record(tmp_path).model_dump()
    raw["outcome"] = {"status": "succeeded", "answer": {"count": -1}}

    with pytest.raises(ValidationError, match="violates the caller schema"):
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


def test_failed_atomic_replace_leaves_no_terminal_or_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = tmp_path / "run-001"
    run_directory.mkdir()

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated replacement failure")

    monkeypatch.setattr("dsa.record.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated"):
        write_terminal_record(terminal_record(tmp_path), run_directory)

    assert list(run_directory.iterdir()) == []


@pytest.mark.parametrize("run_id", ["../escape", "contains/slash", "", " two"])
def test_run_identity_cannot_escape_its_managed_directory(
    tmp_path: Path,
    run_id: str,
) -> None:
    with pytest.raises(ValidationError):
        TerminalRecord.model_validate(
            {**terminal_record(tmp_path).model_dump(), "run_id": run_id}
        )

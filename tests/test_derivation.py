from __future__ import annotations

import json
import os
import stat
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import duckdb
import nbformat
import pytest
from pydantic import JsonValue

from dsa import Derivation, RunPolicy
from dsa.derivation import DerivationError, verify_derivation
from dsa.docker import DockerExecutorConfiguration, DockerPythonExecutor
from dsa.environment import (
    PythonExecutionError,
    PythonExecutionRequest,
    PythonExecutionResult,
)


def sample_derivation() -> Derivation:
    return Derivation.model_validate(
        {
            "format": "dsa-derivation/v1",
            "cells": [
                {
                    "type": "markdown",
                    "source": "Count the rows in the source table.",
                },
                {
                    "type": "code",
                    "source": (
                        "import duckdb\n"
                        "connection = duckdb.connect(str(database_path), read_only=True)\n"
                        "count = connection.execute('select count(*) from events').fetchone()[0]"
                    ),
                },
                {
                    "type": "markdown",
                    "source": "Return the count using the requested shape.",
                },
                {"type": "code", "source": "result = {'count': int(count)}"},
            ],
        }
    )


def database(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "source.duckdb"
    connection = duckdb.connect(str(path))
    try:
        connection.execute("create table events as select * from range(3)")
    finally:
        connection.close()
    return path, sha256(path.read_bytes()).hexdigest()


class ResultExecutor:
    def __init__(self, result: JsonValue) -> None:
        self.result = result
        self.requests: list[PythonExecutionRequest] = []

    async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult:
        self.requests.append(request)
        assert "database_path" in request.source
        assert "dsa-derivation-cell" in request.source
        (request.output_directory / "result.json").write_text(
            json.dumps(self.result, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return PythonExecutionResult(
            runtime_identity="docker/test|sha256:" + "1" * 64
        )


class BackendFailureExecutor:
    async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult:
        del request
        raise PythonExecutionError(
            "python_backend_unavailable",
            "backend unavailable",
        )


async def test_verified_derivation_retains_a_deterministic_valid_notebook(
    tmp_path: Path,
) -> None:
    source, source_digest = database(tmp_path)
    run_directory = tmp_path / "run"
    (run_directory / "work").mkdir(parents=True)
    executor = ResultExecutor({"count": 3})

    verified = await verify_derivation(
        sample_derivation(),
        {"count": 3},
        question="How many rows are in events?",
        source_database=source,
        source_database_sha256=source_digest,
        run_directory=run_directory,
        policy=RunPolicy(),
        python_executor=executor,
    )

    assert verified.verification.status == "verified"
    assert verified.verification.source_database_sha256 == source_digest
    assert verified.verification.runtime_identity == "docker/test|sha256:" + "1" * 64
    assert verified.notebook.path == run_directory / "derivation.ipynb"
    content = verified.notebook.path.read_bytes()
    assert verified.notebook.sha256 == sha256(content).hexdigest()
    assert stat.S_IMODE(verified.notebook.path.stat().st_mode) == 0o600
    notebook_module = cast(Any, nbformat)
    notebook = notebook_module.reads(content.decode(), as_version=4)
    notebook_module.validate(notebook)
    assert notebook.cells[0].cell_type == "markdown"
    assert notebook.cells[2].source == "Count the rows in the source table."
    assert notebook.cells[-1].source == "result"
    assert notebook.metadata.dsa.source_database_sha256 == source_digest
    assert len(executor.requests) == 1
    assert not list((run_directory / "work").glob("derivation-*"))


@pytest.mark.parametrize("replayed", [{"count": 4}, {"count": 3.0}])
async def test_derivation_result_must_match_exact_canonical_json(
    tmp_path: Path,
    replayed: object,
) -> None:
    source, source_digest = database(tmp_path)
    run_directory = tmp_path / "run"
    (run_directory / "work").mkdir(parents=True)

    with pytest.raises(DerivationError) as captured:
        await verify_derivation(
            sample_derivation(),
            {"count": 3},
            question="How many rows are in events?",
            source_database=source,
            source_database_sha256=source_digest,
            run_directory=run_directory,
            policy=RunPolicy(),
            python_executor=ResultExecutor(cast(JsonValue, replayed)),
        )

    assert captured.value.code == "derivation_result_mismatch"
    assert captured.value.infrastructure is False
    assert not (run_directory / "derivation.ipynb").exists()
    assert not list((run_directory / "work").glob("derivation-*"))


async def test_derivation_requires_an_executor_and_classifies_backend_failure(
    tmp_path: Path,
) -> None:
    source, source_digest = database(tmp_path)
    run_directory = tmp_path / "run"
    (run_directory / "work").mkdir(parents=True)
    with pytest.raises(DerivationError) as missing:
        await verify_derivation(
            sample_derivation(),
            {"count": 3},
            question="How many rows are in events?",
            source_database=source,
            source_database_sha256=source_digest,
            run_directory=run_directory,
            policy=RunPolicy(),
            python_executor=None,
        )
    assert missing.value.code == "derivation_executor_unavailable"
    assert missing.value.infrastructure is True

    with pytest.raises(DerivationError) as backend:
        await verify_derivation(
            sample_derivation(),
            {"count": 3},
            question="How many rows are in events?",
            source_database=source,
            source_database_sha256=source_digest,
            run_directory=run_directory,
            policy=RunPolicy(),
            python_executor=BackendFailureExecutor(),
        )
    assert backend.value.code == "python_backend_unavailable"
    assert backend.value.infrastructure is True


@pytest.mark.integration
async def test_real_docker_replays_derivation_and_retains_notebook(
    tmp_path: Path,
) -> None:
    image = os.environ.get("DSA_DOCKER_TEST_IMAGE")
    if image is None:
        pytest.skip("DSA_DOCKER_TEST_IMAGE is required for the real Docker tier")
    source, source_digest = database(tmp_path)
    run_directory = tmp_path / "run"
    (run_directory / "work").mkdir(parents=True)

    verified = await verify_derivation(
        sample_derivation(),
        {"count": 3},
        question="How many rows are in events?",
        source_database=source,
        source_database_sha256=source_digest,
        run_directory=run_directory,
        policy=RunPolicy(
            max_python_seconds=20,
            max_python_memory_bytes=512 * 1024 * 1024,
        ),
        python_executor=DockerPythonExecutor(
            DockerExecutorConfiguration(
                image=image,
                user_id=os.getuid(),
                group_id=os.getgid(),
            ),
            container_name_factory=lambda: "dsa-derivation-real-smoke",
        ),
    )

    assert verified.verification.status == "verified"
    assert verified.notebook.path.is_file()

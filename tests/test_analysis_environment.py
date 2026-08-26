from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import duckdb
import pyarrow.parquet as parquet
import pytest

from dsa import RunPolicy
from dsa.environment import (
    AnalysisEnvironment,
    ArtifactError,
    PythonExecutionRequest,
    PythonExecutionResult,
    PythonExecutor,
)


def database_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "source.duckdb"
    connection = duckdb.connect(str(path))
    try:
        connection.execute("create schema analytics")
        connection.execute(
            "create table analytics.events as "
            "select i::integer as event_id, ('event-' || i)::varchar as label "
            "from range(12) values(i)"
        )
        connection.execute(
            "create view analytics.event_count as "
            "select count(*) as count from analytics.events"
        )
    finally:
        connection.close()
    return path


def environment(
    tmp_path: Path,
    *,
    policy: RunPolicy | None = None,
    python_executor: PythonExecutor | None = None,
) -> AnalysisEnvironment:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    return AnalysisEnvironment(
        database_path=database_fixture(tmp_path),
        run_directory=run_directory,
        policy=policy or RunPolicy(),
        python_executor=python_executor,
    )


async def test_inspection_is_sorted_schema_qualified_and_schema_only(tmp_path: Path) -> None:
    runtime = environment(tmp_path)

    catalog = json.loads(await runtime.inspect_database(None, tool_call_id="inspect-1"))
    relation = json.loads(
        await runtime.inspect_database("analytics.events", tool_call_id="inspect-2")
    )

    assert catalog == {
        "ok": True,
        "complete": True,
        "relations": [
            {"name": "analytics.event_count", "type": "VIEW"},
            {"name": "analytics.events", "type": "BASE TABLE"},
        ],
    }
    assert relation == {
        "ok": True,
        "complete": True,
        "relation": "analytics.events",
        "columns": [
            {"name": "event_id", "type": "INTEGER", "nullable": True},
            {"name": "label", "type": "VARCHAR", "nullable": True},
        ],
    }
    assert "event-0" not in json.dumps(relation)


async def test_inspection_truncation_is_explicit_and_byte_bounded(tmp_path: Path) -> None:
    runtime = environment(
        tmp_path,
        policy=RunPolicy(max_inspection_result_bytes=100),
    )

    encoded = await runtime.inspect_database(None, tool_call_id="inspect-bounded")
    catalog = json.loads(encoded)

    assert catalog["ok"] is True
    assert catalog["complete"] is False
    assert len(catalog["relations"]) < 2
    assert len(encoded.encode("utf-8")) <= 100


async def test_query_rejects_writes_multiple_statements_and_external_sources(
    tmp_path: Path,
) -> None:
    runtime = environment(tmp_path)
    before = sha256(runtime.database_path.read_bytes()).hexdigest()

    write = json.loads(
        await runtime.query_database(
            "delete from analytics.events", tool_call_id="query-write"
        )
    )
    multiple = json.loads(
        await runtime.query_database("select 1; select 2", tool_call_id="query-multiple")
    )
    external = json.loads(
        await runtime.query_database(
            "select * from read_csv_auto('/tmp/private.csv')",
            tool_call_id="query-external",
        )
    )
    secret_catalog = json.loads(
        await runtime.query_database(
            "select * from duckdb_secrets()",
            tool_call_id="query-secret-catalog",
        )
    )

    assert write["error"]["code"] == "query_not_read_only"
    assert multiple["error"]["code"] == "query_statement_count"
    assert external["error"]["code"] == "query_external_access"
    assert secret_catalog["error"]["code"] == "query_external_access"
    assert sha256(runtime.database_path.read_bytes()).hexdigest() == before
    assert runtime.artifact_records == ()


async def test_small_complete_query_result_stays_inline(tmp_path: Path) -> None:
    runtime = environment(tmp_path)

    result = json.loads(
        await runtime.query_database(
            "select event_id, label from analytics.events where event_id < 2 order by event_id",
            tool_call_id="query-small",
        )
    )

    assert result == {
        "ok": True,
        "transport": "inline",
        "complete": True,
        "columns": [
            {"name": "event_id", "type": "INTEGER"},
            {"name": "label", "type": "VARCHAR"},
        ],
        "row_count": 2,
        "rows": [[0, "event-0"], [1, "event-1"]],
    }
    assert runtime.artifact_records == ()


async def test_inline_query_uses_lossless_tagged_json_values(tmp_path: Path) -> None:
    runtime = environment(tmp_path)

    result = json.loads(
        await runtime.query_database(
            "select 1.25::decimal(4, 2) as amount, "
            "date '2026-01-02' as day, 'hi'::blob as payload",
            tool_call_id="query-tagged-values",
        )
    )

    assert result["rows"] == [
        [
            {"$type": "decimal", "value": "1.25"},
            {"$type": "date", "value": "2026-01-02"},
            {"$type": "bytes", "base64": "aGk="},
        ]
    ]


async def test_larger_result_is_automatically_retained_in_full_as_parquet(
    tmp_path: Path,
) -> None:
    runtime = environment(tmp_path, policy=RunPolicy(max_preview_rows=5))

    result = json.loads(
        await runtime.query_database(
            "select event_id, label from analytics.events order by event_id",
            tool_call_id="query-large",
        )
    )

    assert result["transport"] == "artifact"
    assert result["complete"] is True
    assert result["row_count"] == 12
    assert result["artifact_handle"] == "a1"
    assert result["artifact_path"] == "DSAGENT_INPUTS/a1.parquet"
    assert result["preview_complete"] is False
    assert len(result["preview_rows"]) == 5
    assert len(result["preview_rows"]) < result["row_count"]
    assert len(json.dumps(result).encode()) <= runtime.policy.max_tool_result_bytes

    records = runtime.artifact_records
    assert len(records) == 1
    record = records[0]
    assert record.handle == "a1"
    assert record.producer_tool_call_id == "query-large"
    retained = runtime.run_directory / record.relative_path
    assert retained.is_file()
    assert record.sha256 == sha256(retained.read_bytes()).hexdigest()
    table = parquet.read_table(  # pyright: ignore[reportUnknownMemberType]
        retained
    )
    assert table.num_rows == 12
    assert table.column("event_id").to_pylist() == list(range(12))


async def test_query_over_materialization_limit_fails_without_partial_artifact(
    tmp_path: Path,
) -> None:
    runtime = environment(tmp_path, policy=RunPolicy(max_query_rows=3))

    result = json.loads(
        await runtime.query_database(
            "select * from analytics.events order by event_id",
            tool_call_id="query-too-large",
        )
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "query_row_limit"
    assert "aggregate" in result["error"]["message"]
    assert runtime.artifact_records == ()
    assert not (runtime.run_directory / "artifacts").exists()


class FakePythonExecutor:
    def __init__(self) -> None:
        self.request: PythonExecutionRequest | None = None

    async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult:
        self.request = request
        assert request.environment["DSAGENT_DATABASE"] == str(request.database_path)
        assert request.environment["DSAGENT_INPUTS"] == str(request.inputs_directory)
        assert request.environment["DSAGENT_OUTPUTS"] == str(request.output_directory)
        table = parquet.read_table(  # pyright: ignore[reportUnknownMemberType]
            request.inputs_directory / "a1.parquet"
        )
        (request.output_directory / "answer.json").write_text(
            json.dumps({"count": table.num_rows}), encoding="utf-8"
        )
        return PythonExecutionResult(stdout="computed 12 rows\n", stderr="")


async def test_injected_python_executor_consumes_managed_parquet_and_publishes_json(
    tmp_path: Path,
) -> None:
    executor = FakePythonExecutor()
    runtime = environment(tmp_path, python_executor=executor)
    query = json.loads(
        await runtime.query_database(
            "select * from analytics.events order by event_id",
            tool_call_id="query-for-python",
        )
    )
    assert query["artifact_handle"] == "a1"

    result = json.loads(
        await runtime.run_python(
            "# model-authored source is passed only to the injected executor",
            inputs=["a1"],
            expected_outputs=["answer.json"],
            tool_call_id="python-1",
        )
    )

    assert executor.request is not None
    assert executor.request.source.startswith("# model-authored")
    assert result["ok"] is True
    assert result["stdout"] == "computed 12 rows\n"
    assert result["outputs"] == [
        {
            "handle": "a2",
            "path": "DSAGENT_INPUTS/a2.json",
            "media_type": "application/json",
            "size_bytes": 13,
        }
    ]
    assert runtime.load_json_artifact("a2") == {"count": 12}
    assert [record.producer_tool_call_id for record in runtime.artifact_records] == [
        "query-for-python",
        "python-1",
    ]


async def test_artifact_integrity_is_checked_before_consumption(tmp_path: Path) -> None:
    executor = FakePythonExecutor()
    runtime = environment(tmp_path, python_executor=executor)
    result = json.loads(
        await runtime.query_database(
            "select * from analytics.events order by event_id",
            tool_call_id="query-integrity",
        )
    )
    record = runtime.artifact_records[0]
    (runtime.run_directory / record.relative_path).write_bytes(b"tampered")

    with pytest.raises(ArtifactError) as captured:
        runtime.load_json_artifact(result["artifact_handle"])

    assert captured.value.code == "artifact_integrity"


class InvalidOutputExecutor:
    async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult:
        (request.output_directory / "answer.json").write_text(
            "not JSON", encoding="utf-8"
        )
        return PythonExecutionResult()


async def test_invalid_python_output_is_not_published(tmp_path: Path) -> None:
    runtime = environment(tmp_path, python_executor=InvalidOutputExecutor())
    await runtime.query_database(
        "select * from analytics.events order by event_id",
        tool_call_id="query-before-invalid-output",
    )

    result = json.loads(
        await runtime.run_python(
            "# write invalid declared output",
            inputs=["a1"],
            expected_outputs=["answer.json"],
            tool_call_id="python-invalid-output",
        )
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "python_output_invalid"
    assert [record.handle for record in runtime.artifact_records] == ["a1"]

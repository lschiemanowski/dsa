from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterator
from hashlib import sha256
from pathlib import Path
from typing import cast

import duckdb
import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

import dsa.environment as environment_module
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
        connection.execute("create macro unsafe_log() as write_log('audit')")
        connection.execute(
            "create macro unsafe_settings() as table select * from duckdb_settings()"
        )
        connection.execute(
            "create macro unsafe_secret_directory() "
            "as current_setting('secret_directory')"
        )
        connection.execute(
            "create view analytics.unsafe_secret_view as "
            "select current_setting('secret_directory') as benign"
        )
        connection.execute(
            "create view unsafe_v as "
            "select current_setting('secret_directory') as benign"
        )
        connection.execute(
            "create view analytics.safe_with_view as "
            "with q as (select 7 as value) select value from q"
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
            {"name": '"analytics"."event_count"', "type": "VIEW"},
            {"name": '"analytics"."events"', "type": "BASE TABLE"},
            {"name": '"analytics"."safe_with_view"', "type": "VIEW"},
            {"name": '"analytics"."unsafe_secret_view"', "type": "VIEW"},
            {"name": '"main"."unsafe_v"', "type": "VIEW"},
        ],
    }
    assert relation == {
        "ok": True,
        "complete": True,
        "relation": '"analytics"."events"',
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
    commented_write = json.loads(
        await runtime.query_database(
            "select write_log/**/('hello')",
            tool_call_id="query-commented-write",
        )
    )
    settings = json.loads(
        await runtime.query_database(
            "select * from duckdb_settings()",
            tool_call_id="query-settings",
        )
    )
    macro_write = json.loads(
        await runtime.query_database(
            "select unsafe_log()",
            tool_call_id="query-macro-write",
        )
    )
    macro_settings = json.loads(
        await runtime.query_database(
            "select * from unsafe_settings()",
            tool_call_id="query-macro-settings",
        )
    )
    secret_directory = json.loads(
        await runtime.query_database(
            "select current_setting('secret_directory')",
            tool_call_id="query-secret-directory",
        )
    )
    macro_secret_directory = json.loads(
        await runtime.query_database(
            "select unsafe_secret_directory()",
            tool_call_id="query-macro-secret-directory",
        )
    )
    aliased_write = json.loads(
        await runtime.query_database(
            "select write_log('hello') as benign",
            tool_call_id="query-aliased-write",
        )
    )
    aliased_secret_directory = json.loads(
        await runtime.query_database(
            "select current_setting('secret_directory') as benign",
            tool_call_id="query-aliased-secret-directory",
        )
    )
    aliased_macro_write = json.loads(
        await runtime.query_database(
            "select unsafe_log() as benign",
            tool_call_id="query-aliased-macro-write",
        )
    )
    harmless_literal = json.loads(
        await runtime.query_database(
            "select 1 as value where 'write_log(' = 'write_log('",
            tool_call_id="query-harmless-function-literal",
        )
    )
    unsafe_view = json.loads(
        await runtime.query_database(
            "select * from analytics.unsafe_secret_view",
            tool_call_id="query-unsafe-view",
        )
    )
    safe_cte_shadow = json.loads(
        await runtime.query_database(
            "with unsafe_secret_view as (select 1 as value) "
            "select value from unsafe_secret_view",
            tool_call_id="query-safe-cte-shadow",
        )
    )
    safe_view = json.loads(
        await runtime.query_database(
            "select count from analytics.event_count",
            tool_call_id="query-safe-view",
        )
    )
    nested_cte_name = json.loads(
        await runtime.query_database(
            "select * from unsafe_v where exists ("
            "with unsafe_v as (select 1 as value) select 1)",
            tool_call_id="query-nested-cte-name",
        )
    )
    safe_with_view = json.loads(
        await runtime.query_database(
            "select value from analytics.safe_with_view",
            tool_call_id="query-safe-with-view",
        )
    )

    assert write["error"]["code"] == "query_not_read_only"
    assert multiple["error"]["code"] == "query_statement_count"
    assert external["error"]["code"] == "query_external_access"
    assert secret_catalog["error"]["code"] == "query_external_access"
    assert commented_write["error"]["code"] == "query_external_access"
    assert settings["error"]["code"] == "query_external_access"
    assert macro_write["error"]["code"] == "query_external_access"
    assert macro_settings["error"]["code"] == "query_external_access"
    assert secret_directory["error"]["code"] == "query_external_access"
    assert macro_secret_directory["error"]["code"] == "query_external_access"
    assert aliased_write["error"]["code"] == "query_external_access"
    assert aliased_secret_directory["error"]["code"] == "query_external_access"
    assert aliased_macro_write["error"]["code"] == "query_external_access"
    assert harmless_literal["rows"] == [[1]]
    assert unsafe_view["error"]["code"] == "query_external_access"
    assert safe_cte_shadow["rows"] == [[1]]
    assert safe_view["rows"] == [[12]]
    assert nested_cte_name["error"]["code"] == "query_external_access"
    assert safe_with_view["rows"] == [[7]]
    assert "stored_secrets" not in json.dumps(secret_catalog)
    assert "stored_secrets" not in json.dumps(secret_directory)
    assert "stored_secrets" not in json.dumps(macro_secret_directory)
    assert "stored_secrets" not in json.dumps(unsafe_view)
    assert "stored_secrets" not in json.dumps(nested_cte_name)
    assert "temp_directory" not in json.dumps(settings)
    assert sha256(runtime.database_path.read_bytes()).hexdigest() == before
    assert runtime.artifact_records == ()


async def test_catalog_preserves_ambiguous_identifier_boundaries(tmp_path: Path) -> None:
    database = database_fixture(tmp_path)
    connection = duckdb.connect(str(database))
    try:
        connection.execute('create schema "a.b"')
        connection.execute('create table "a.b".t(dotted_schema integer)')
        connection.execute("create schema a")
        connection.execute('create table a."b.t"(dotted_relation varchar)')
        connection.execute('create schema "q""uote"')
        connection.execute('create table "q""uote"."n""ame"(quoted_identifier boolean)')
    finally:
        connection.close()
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    runtime = AnalysisEnvironment(
        database_path=database,
        run_directory=run_directory,
        policy=RunPolicy(),
    )

    catalog = json.loads(await runtime.inspect_database(None, tool_call_id="inspect-catalog"))
    names = [relation["name"] for relation in catalog["relations"]]
    dotted_schema = json.loads(
        await runtime.inspect_database('"a.b"."t"', tool_call_id="inspect-dotted-schema")
    )
    dotted_relation = json.loads(
        await runtime.inspect_database('"a"."b.t"', tool_call_id="inspect-dotted-relation")
    )
    quoted_identifier = json.loads(
        await runtime.inspect_database(
            '"q""uote"."n""ame"',
            tool_call_id="inspect-quoted-identifier",
        )
    )

    assert '"a.b"."t"' in names
    assert '"a"."b.t"' in names
    assert '"q""uote"."n""ame"' in names
    assert dotted_schema["columns"][0]["name"] == "dotted_schema"
    assert dotted_relation["columns"][0]["name"] == "dotted_relation"
    assert quoted_identifier["columns"][0]["name"] == "quoted_identifier"


async def test_targeted_inspection_does_not_materialize_the_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Cursor:
        def __init__(
            self,
            *,
            row: tuple[str, str] | None = None,
            rows: list[tuple[str, str, str]] | None = None,
            forbid_fetchall: bool = False,
        ) -> None:
            self.row = row
            self.rows = rows or []
            self.forbid_fetchall = forbid_fetchall
            self.row_consumed = False

        def fetchone(self) -> tuple[str, ...] | None:
            if self.row is not None and not self.row_consumed:
                self.row_consumed = True
                return self.row
            return self.rows.pop(0) if self.rows else None

        def fetchall(self) -> list[tuple[str, str, str]]:
            if self.forbid_fetchall:
                raise AssertionError("targeted inspection materialized the catalog")
            return self.rows

    class Connection:
        def execute(
            self,
            sql: str,
            parameters: list[str] | None = None,
        ) -> Cursor:
            if "select table_schema, table_name" in sql:
                assert parameters == ["analytics", "events"]
                assert "limit 1" in sql.lower()
                return Cursor(row=("analytics", "events"), forbid_fetchall=True)
            assert "information_schema.columns" in sql
            assert parameters == ["analytics", "events"]
            return Cursor(rows=[("event_id", "INTEGER", "YES")])

        def interrupt(self) -> None:
            pass

        def close(self) -> None:
            pass

    connection = Connection()

    def open_connection(_path: Path, _memory_bytes: int) -> Connection:
        return connection

    monkeypatch.setattr(
        environment_module,
        "_open_connection",
        open_connection,
    )
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    runtime = AnalysisEnvironment(
        database_path=tmp_path / "unused.duckdb",
        run_directory=run_directory,
        policy=RunPolicy(),
    )

    result = json.loads(
        await runtime.inspect_database('"analytics"."events"', tool_call_id="inspect-one")
    )

    assert result["relation"] == '"analytics"."events"'
    assert result["columns"] == [
        {"name": "event_id", "type": "INTEGER", "nullable": True}
    ]


async def test_catalog_inspection_stops_reading_at_its_byte_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Cursor:
        def __init__(self) -> None:
            self.reads = 0

        def fetchone(self) -> tuple[str, str, str] | None:
            self.reads += 1
            return ("analytics", f"relation_{self.reads:04d}", "BASE TABLE")

        def fetchall(self) -> list[tuple[str, str, str]]:
            raise AssertionError("catalog inspection materialized every relation")

    class Connection:
        def __init__(self) -> None:
            self.cursor = Cursor()

        def execute(self, sql: str) -> Cursor:
            assert "information_schema.tables" in sql
            return self.cursor

        def interrupt(self) -> None:
            pass

        def close(self) -> None:
            pass

    connection = Connection()

    def open_connection(_path: Path, _memory_bytes: int) -> Connection:
        return connection

    monkeypatch.setattr(environment_module, "_open_connection", open_connection)
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    runtime = AnalysisEnvironment(
        database_path=tmp_path / "unused.duckdb",
        run_directory=run_directory,
        policy=RunPolicy(max_inspection_result_bytes=160),
    )

    result = json.loads(await runtime.inspect_database(None, tool_call_id="inspect-bounded"))

    assert result["ok"] is True
    assert result["complete"] is False
    assert 0 < len(result["relations"]) < connection.cursor.reads
    assert connection.cursor.reads < 10


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


async def test_leading_and_trailing_comments_remain_valid_single_queries(
    tmp_path: Path,
) -> None:
    runtime = environment(tmp_path)

    leading = json.loads(
        await runtime.query_database(
            "/* orientation */ select 1 as value",
            tool_call_id="query-leading-comment",
        )
    )
    trailing = json.loads(
        await runtime.query_database(
            "select 2 as value -- trailing comment",
            tool_call_id="query-trailing-comment",
        )
    )

    assert leading["rows"] == [[1]]
    assert trailing["rows"] == [[2]]


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


async def test_nonfinite_inline_value_is_retained_as_parquet_without_preview(
    tmp_path: Path,
) -> None:
    runtime = environment(tmp_path)

    result = json.loads(
        await runtime.query_database(
            "select 'NaN'::double as value",
            tool_call_id="query-nan",
        )
    )

    assert result["ok"] is True
    assert result["transport"] == "artifact"
    assert result["row_count"] == 1
    assert result["preview_rows"] == []
    record = runtime.artifact_records[0]
    table = parquet.read_table(  # pyright: ignore[reportUnknownMemberType]
        runtime.run_directory / record.relative_path
    )
    assert math.isnan(table.column("value")[0].as_py())


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
    await runtime.query_database(
        "select * from analytics.events order by event_id",
        tool_call_id="query-before-integrity",
    )
    result = json.loads(
        await runtime.run_python(
            "# produce a JSON answer",
            inputs=["a1"],
            expected_outputs=["answer.json"],
            tool_call_id="python-integrity",
        )
    )
    record = runtime.artifact_records[1]
    (runtime.run_directory / record.relative_path).write_bytes(b"tampered")

    with pytest.raises(ArtifactError) as captured:
        runtime.load_json_artifact(result["outputs"][0]["handle"])

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


class TwoOutputExecutor:
    async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult:
        (request.output_directory / "first.json").write_text("{\"value\":1}", encoding="utf-8")
        (request.output_directory / "second.json").write_text("{\"value\":2}", encoding="utf-8")
        return PythonExecutionResult()


async def test_multi_output_publication_rolls_back_the_complete_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = environment(tmp_path, python_executor=TwoOutputExecutor())
    original_link = os.link
    calls = 0

    def fail_second_link(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second publication failure")
        original_link(source, destination)

    monkeypatch.setattr("dsa.environment.os.link", fail_second_link)

    result = json.loads(
        await runtime.run_python(
            "# produce two declared outputs",
            inputs=[],
            expected_outputs=["first.json", "second.json"],
            tool_call_id="python-two-outputs",
        )
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "artifact_publication_failed"
    assert runtime.artifact_records == ()
    artifacts = runtime.run_directory / "artifacts"
    assert not artifacts.exists() or list(artifacts.iterdir()) == []


class JsonAnswerExecutor:
    async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult:
        (request.output_directory / "answer.json").write_text(
            '{"count": 12}', encoding="utf-8"
        )
        return PythonExecutionResult()


class ReplaceableJsonOutputExecutor:
    def __init__(self) -> None:
        self.output: Path | None = None

    async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult:
        self.output = request.output_directory / "answer.json"
        self.output.write_text('{"value":1}', encoding="utf-8")
        return PythonExecutionResult()


async def test_python_publishes_the_exact_output_bytes_that_were_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = ReplaceableJsonOutputExecutor()
    runtime = environment(tmp_path, python_executor=executor)
    original_validate = cast(
        Callable[[Path, str, int, int], None],
        vars(environment_module)["_validate_executor_output"],
    )

    def replace_executor_source_after_validation(
        path: Path,
        media_type: str,
        max_bytes: int,
        max_json_bytes: int,
    ) -> None:
        original_validate(path, media_type, max_bytes, max_json_bytes)
        assert executor.output is not None
        executor.output.write_text("not-json!!!", encoding="utf-8")

    monkeypatch.setattr(
        environment_module,
        "_validate_executor_output",
        replace_executor_source_after_validation,
    )

    result = json.loads(
        await runtime.run_python(
            "# produce one replaceable JSON output",
            inputs=[],
            expected_outputs=["answer.json"],
            tool_call_id="python-replace-output",
        )
    )

    assert result["ok"] is True
    assert runtime.load_json_artifact(result["outputs"][0]["handle"]) == {"value": 1}


async def test_json_consumer_uses_the_same_bytes_it_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = environment(tmp_path, python_executor=JsonAnswerExecutor())
    await runtime.run_python(
        "# produce JSON answer",
        inputs=[],
        expected_outputs=["answer.json"],
        tool_call_id="python-json-answer",
    )
    record = runtime.artifact_records[0]
    retained = runtime.run_directory / record.relative_path
    original_fdopen = os.fdopen
    replaced = False

    def replace_after_open(descriptor: int, mode: str):
        nonlocal replaced
        if mode == "rb" and not replaced:
            replacement = retained.with_name("replacement.json")
            replacement.write_text('{"count": 99}', encoding="utf-8")
            os.replace(replacement, retained)
            replaced = True
        return original_fdopen(descriptor, mode)

    monkeypatch.setattr("dsa.environment.os.fdopen", replace_after_open)

    assert runtime.load_json_artifact("a1") == {"count": 12}
    assert replaced is True


class ParquetOutputExecutor:
    async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult:
        parquet.write_table(  # pyright: ignore[reportUnknownMemberType]
            pa.table({"value": [1, 2, 3]}),  # pyright: ignore[reportUnknownMemberType]
            request.output_directory / "result.parquet",
        )
        return PythonExecutionResult()


async def test_parquet_output_validation_reads_data_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = environment(tmp_path, python_executor=ParquetOutputExecutor())

    class CorruptParquet:
        def iter_batches(self, *args: object, **kwargs: object) -> Iterator[object]:
            del args, kwargs
            raise OSError("corrupt data page")

    def parquet_factory(_path: object) -> CorruptParquet:
        return CorruptParquet()

    monkeypatch.setattr("dsa.environment.parquet.ParquetFile", parquet_factory)

    result = json.loads(
        await runtime.run_python(
            "# produce Parquet output",
            inputs=[],
            expected_outputs=["result.parquet"],
            tool_call_id="python-parquet-output",
        )
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "python_output_invalid"
    assert runtime.artifact_records == ()

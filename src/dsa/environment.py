"""Bounded DuckDB tools, run-private artifacts, and the Python executor seam."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
import stat
import tempfile
from base64 import b64encode
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from threading import Event, Lock, Timer
from typing import Any, Protocol, cast
from uuid import UUID, uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as parquet
from pydantic import JsonValue

from dsa.contract import RunPolicy
from dsa.record import ArtifactRecord

_PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
_JSON_MEDIA_TYPE = "application/json"
_MEDIA_EXTENSIONS = {
    _JSON_MEDIA_TYPE: ".json",
    _PARQUET_MEDIA_TYPE: ".parquet",
}
_OUTPUT_MEDIA_TYPES = {
    ".json": _JSON_MEDIA_TYPE,
    ".parquet": _PARQUET_MEDIA_TYPE,
}
_SAFE_OUTPUT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_ALLOWED_BOUND_TABLE_FUNCTIONS = frozenset(
    {
        "generate_series",
        "json_each",
        "json_tree",
        "range",
        "repeat",
        "repeat_row",
        "unnest",
    }
)
_FORBIDDEN_BOUND_SCALAR_FUNCTIONS = frozenset(
    {"current_setting", "getvariable", "setvariable"}
)
_VIEW_DEFINITION = re.compile(
    r'^\s*CREATE\s+VIEW\s+(?:"(?:[^"]|"")*"|[A-Z_][A-Z0-9_$]*)'
    r'(?:\s*\.\s*(?:"(?:[^"]|"")*"|[A-Z_][A-Z0-9_$]*))?'
    r'(?:\s*\((?:[^()"]|"(?:[^"]|"")*")*\))?\s+AS\s+(.+)\s*;\s*$',
    re.IGNORECASE | re.DOTALL,
)


class ArtifactError(Exception):
    """Stable artifact-boundary rejection suitable for model retry feedback."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ToolResultLimitExceeded(Exception):
    """A model-visible tool result exceeded a host-enforced byte limit."""


@dataclass(frozen=True)
class PythonExecutionRequest:
    """Managed paths and source passed to an injected Python executor."""

    source: str
    database_path: Path
    inputs_directory: Path
    output_directory: Path
    expected_outputs: tuple[str, ...]
    environment: Mapping[str, str]


@dataclass(frozen=True)
class PythonExecutionResult:
    """Bounded textual diagnostics returned by an injected Python executor."""

    stdout: str = ""
    stderr: str = ""


class PythonExecutor(Protocol):
    """Executor implemented by the Docker milestone, or by deterministic tests."""

    async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult: ...


@dataclass(frozen=True)
class _QueryMaterialization:
    columns: tuple[dict[str, JsonValue], ...]
    table: pa.Table


class _ArtifactStore:
    def __init__(self, run_directory: Path, policy: RunPolicy) -> None:
        self._run_directory = run_directory
        self._directory = run_directory / "artifacts"
        self._policy = policy
        self._records: list[ArtifactRecord] = []
        self._lock = Lock()

    @property
    def directory(self) -> Path:
        self._directory.mkdir(mode=0o700, exist_ok=True)
        return self._directory

    @property
    def records(self) -> tuple[ArtifactRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def _record_for(self, handle: str) -> ArtifactRecord:
        with self._lock:
            record = next((item for item in self._records if item.handle == handle), None)
        if record is None:
            raise ArtifactError("artifact_unknown", "No same-run artifact has that handle")
        return record

    def read_verified_bytes(
        self,
        handle: str,
        *,
        max_bytes: int,
        expected_media_type: str | None = None,
        media_type_error: tuple[str, str] | None = None,
    ) -> tuple[ArtifactRecord, bytes]:
        record = self._record_for(handle)
        if expected_media_type is not None and record.media_type != expected_media_type:
            code, message = media_type_error or (
                "artifact_media_type",
                "The retained artifact has an unsupported media type",
            )
            raise ArtifactError(code, message)
        if record.size_bytes > max_bytes:
            raise ArtifactError(
                "artifact_size_limit",
                "The retained artifact exceeds the consumption byte limit",
            )
        descriptor = self._open_verified_descriptor(record)
        digest = sha256()
        content = bytearray()
        with os.fdopen(descriptor, "rb") as source:
            while chunk := source.read(1024 * 1024):
                content.extend(chunk)
                digest.update(chunk)
                if len(content) > max_bytes:
                    raise ArtifactError(
                        "artifact_size_limit",
                        "The retained artifact exceeds the consumption byte limit",
                    )
        self._verify_consumed(record, len(content), digest.hexdigest())
        return record, bytes(content)

    def copy_verified(self, handle: str, directory: Path) -> tuple[ArtifactRecord, Path]:
        record = self._record_for(handle)
        destination = directory / Path(record.relative_path).name
        source_descriptor = self._open_verified_descriptor(record)
        digest = sha256()
        copied = 0
        destination_descriptor: int | None = None
        try:
            destination_descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o400,
            )
            with os.fdopen(destination_descriptor, "wb") as target:
                destination_descriptor = None
                with os.fdopen(source_descriptor, "rb") as source:
                    source_descriptor = -1
                    while chunk := source.read(1024 * 1024):
                        copied += len(chunk)
                        digest.update(chunk)
                        target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            self._verify_consumed(record, copied, digest.hexdigest())
            return record, destination
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        finally:
            if source_descriptor >= 0:
                os.close(source_descriptor)
            if destination_descriptor is not None:
                os.close(destination_descriptor)

    def publish_file(
        self,
        source: Path,
        *,
        media_type: str,
        producer_tool_call_id: str,
    ) -> ArtifactRecord:
        return self.publish_files(
            [(source, media_type)],
            producer_tool_call_id=producer_tool_call_id,
        )[0]

    def publish_files(
        self,
        sources: Sequence[tuple[Path, str]],
        *,
        producer_tool_call_id: str,
        validator: Callable[[Path, str], None] | None = None,
    ) -> tuple[ArtifactRecord, ...]:
        if not sources:
            raise ArtifactError("artifact_batch_empty", "An artifact batch must not be empty")
        if not producer_tool_call_id or len(producer_tool_call_id) > 256:
            raise ArtifactError("artifact_producer", "The artifact producer identity is invalid")

        with self._lock:
            source_sizes: list[int] = []
            extensions: list[str] = []
            for source, media_type in sources:
                extension = _MEDIA_EXTENSIONS.get(media_type)
                if extension is None:
                    raise ArtifactError(
                        "artifact_media_type",
                        "The artifact media type is unsupported",
                    )
                if source.is_symlink() or not source.is_file():
                    raise ArtifactError(
                        "artifact_source",
                        "The artifact source must be a regular file",
                    )
                source_sizes.append(source.stat().st_size)
                extensions.append(extension)
            total_size = sum(item.size_bytes for item in self._records)
            if len(self._records) + len(sources) > self._policy.max_artifact_count:
                raise ArtifactError(
                    "artifact_count_limit",
                    "The run artifact count limit was reached",
                )
            if any(size > self._policy.max_artifact_bytes for size in source_sizes):
                raise ArtifactError(
                    "artifact_size_limit",
                    "The complete result is too large to retain. Aggregate or filter it further",
                )
            if total_size + sum(source_sizes) > self._policy.max_total_artifact_bytes:
                raise ArtifactError(
                    "artifact_total_limit",
                    "The run artifact byte limit was reached. Reuse or reduce existing results",
                )

            directory = self.directory
            prepared: list[tuple[Path, Path, ArtifactRecord]] = []
            linked: list[Path] = []
            published = False
            try:
                for offset, ((source, media_type), extension, expected_size) in enumerate(
                    zip(sources, extensions, source_sizes, strict=True),
                    start=1,
                ):
                    handle = f"a{len(self._records) + offset}"
                    destination = directory / f"{handle}{extension}"
                    temporary = directory / f".artifact.{uuid4().hex}.tmp"
                    digest, copied = self._copy_artifact_source(source, temporary)
                    try:
                        if copied != expected_size:
                            raise ArtifactError(
                                "artifact_source_changed",
                                "The artifact source changed while it was retained",
                            )
                        if validator is not None:
                            validator(temporary, media_type)
                    except BaseException:
                        temporary.unlink(missing_ok=True)
                        raise
                    record = ArtifactRecord(
                        handle=handle,
                        relative_path=destination.relative_to(self._run_directory).as_posix(),
                        media_type=media_type,
                        size_bytes=copied,
                        sha256=digest,
                        producer_tool_call_id=producer_tool_call_id,
                    )
                    prepared.append((temporary, destination, record))
                for temporary, destination, _record in prepared:
                    os.link(temporary, destination)
                    linked.append(destination)
                _fsync_directory(directory)
                published = True
            except ArtifactError:
                raise
            except OSError as error:
                raise ArtifactError(
                    "artifact_publication_failed",
                    "The artifact batch could not be published",
                ) from error
            finally:
                for temporary, _destination, _record in prepared:
                    temporary.unlink(missing_ok=True)
                if not published:
                    for destination in linked:
                        destination.unlink(missing_ok=True)
                    if directory.exists():
                        _fsync_directory(directory)
            records = tuple(record for _temporary, _destination, record in prepared)
            self._records.extend(records)
            return records

    def _copy_artifact_source(self, source: Path, temporary: Path) -> tuple[str, int]:
        source_descriptor = os.open(
            source,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        destination_descriptor: int | None = None
        digest = sha256()
        copied = 0
        try:
            destination_descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(destination_descriptor, "wb") as target:
                destination_descriptor = None
                with os.fdopen(source_descriptor, "rb") as origin:
                    source_descriptor = -1
                    while chunk := origin.read(1024 * 1024):
                        copied += len(chunk)
                        if copied > self._policy.max_artifact_bytes:
                            raise ArtifactError(
                                "artifact_size_limit",
                                "The complete result is too large to retain. "
                                "Aggregate or filter it further",
                            )
                        digest.update(chunk)
                        target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            return digest.hexdigest(), copied
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            if source_descriptor >= 0:
                os.close(source_descriptor)
            if destination_descriptor is not None:
                os.close(destination_descriptor)

    def _open_verified_descriptor(self, record: ArtifactRecord) -> int:
        path = self._run_directory / record.relative_path
        descriptor: int | None = None
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError("artifact is not a regular file")
            return descriptor
        except OSError as error:
            if descriptor is not None:
                os.close(descriptor)
            raise ArtifactError(
                "artifact_unavailable",
                "The retained artifact is unavailable",
            ) from error

    @staticmethod
    def _verify_consumed(record: ArtifactRecord, size: int, digest: str) -> None:
        if size != record.size_bytes or digest != record.sha256:
            raise ArtifactError(
                "artifact_integrity",
                "The retained artifact no longer matches its integrity record",
            )


class AnalysisEnvironment:
    """Run-scoped implementation of model-visible analysis tools."""

    def __init__(
        self,
        *,
        database_path: Path,
        run_directory: Path,
        policy: RunPolicy,
        python_executor: PythonExecutor | None = None,
    ) -> None:
        self.database_path = database_path
        self.run_directory = run_directory
        self.policy = policy
        self.python_executor = python_executor
        self._artifacts = _ArtifactStore(run_directory, policy)

    @property
    def artifact_records(self) -> tuple[ArtifactRecord, ...]:
        return self._artifacts.records

    async def check_database(self) -> None:
        """Open the supplied file through the same locked-down DuckDB boundary."""
        async with asyncio.timeout(self.policy.max_inspection_seconds):
            await asyncio.to_thread(self._check_database_sync)

    async def inspect_database(self, relation: str | None, *, tool_call_id: str) -> str:
        try:
            result = await asyncio.to_thread(self._inspect_database_sync, relation)
        except TimeoutError:
            return self._tool_error("inspection_timeout", "Database inspection timed out")
        except duckdb.Error as error:
            return self._tool_error(
                "inspection_failed",
                _safe_duckdb_message(error, self.database_path),
            )
        limit = min(
            self.policy.max_inspection_result_bytes,
            self.policy.max_tool_result_bytes,
        )
        return _bound_inspection_result(result, limit, self._tool_error)

    async def query_database(self, sql: str, *, tool_call_id: str) -> str:
        issue = _query_issue(sql)
        if issue is not None:
            return self._tool_error(*issue)
        try:
            materialization = await asyncio.to_thread(self._query_database_sync, sql)
        except ArtifactError as error:
            return self._tool_error(error.code, error.message)
        except TimeoutError:
            return self._tool_error(
                "query_timeout",
                "The query timed out. Filter or aggregate the result further",
            )
        except MemoryError:
            return self._tool_error(
                "query_memory_limit",
                "The query exceeded its memory limit. Filter or aggregate it further",
            )
        except duckdb.Error as error:
            return self._tool_error(
                "query_failed",
                _safe_duckdb_message(error, self.database_path),
            )

        columns = list(materialization.columns)
        row_count = materialization.table.num_rows
        preview_count = min(self.policy.max_preview_rows, row_count)
        preview_rows: list[list[JsonValue]] = []
        if row_count <= self.policy.max_preview_rows:
            try:
                inline = {
                    "ok": True,
                    "transport": "inline",
                    "complete": True,
                    "columns": columns,
                    "row_count": row_count,
                    "rows": _table_rows(materialization.table),
                }
                encoded_inline = _canonical_json(inline)
                if len(encoded_inline.encode("utf-8")) <= self.policy.max_tool_result_bytes:
                    return encoded_inline
            except (TypeError, ValueError):
                preview_count = 0
        if preview_count:
            try:
                preview_rows = _table_rows(
                    materialization.table.slice(0, preview_count)
                )
            except (TypeError, ValueError):
                preview_count = 0
                preview_rows = []

        temporary = self.run_directory / f".query.{uuid4().hex}.parquet"
        try:
            parquet.write_table(  # pyright: ignore[reportUnknownMemberType]
                materialization.table,
                temporary,
            )
            record = self._artifacts.publish_file(
                temporary,
                media_type=_PARQUET_MEDIA_TYPE,
                producer_tool_call_id=tool_call_id,
            )
        except ArtifactError as error:
            return self._tool_error(error.code, error.message)
        finally:
            temporary.unlink(missing_ok=True)

        while preview_count >= 0:
            result = {
                "ok": True,
                "transport": "artifact",
                "complete": True,
                "columns": columns,
                "row_count": row_count,
                "artifact_handle": record.handle,
                "artifact_path": f"DSAGENT_INPUTS/{Path(record.relative_path).name}",
                "preview_complete": False,
                "preview_rows": preview_rows[:preview_count],
            }
            encoded = _canonical_json(result)
            if len(encoded.encode("utf-8")) <= self.policy.max_tool_result_bytes:
                return encoded
            preview_count -= 1
        raise ToolResultLimitExceeded("query metadata exceeds the per-result byte limit")

    async def run_python(
        self,
        source: str,
        inputs: Sequence[str],
        expected_outputs: Sequence[str],
        *,
        tool_call_id: str,
    ) -> str:
        if self.python_executor is None:
            return self._tool_error(
                "python_unavailable",
                "No isolated Python executor is configured for this run",
            )
        issue = _python_request_issue(source, inputs, expected_outputs)
        if issue is not None:
            return self._tool_error(*issue)
        staging = Path(tempfile.mkdtemp(prefix=".python-", dir=self.run_directory))
        inputs_directory = staging / "inputs"
        inputs_directory.mkdir(mode=0o700)
        output_directory = staging / "outputs"
        output_directory.mkdir(mode=0o700)
        try:
            for handle in inputs:
                self._artifacts.copy_verified(handle, inputs_directory)
            request = PythonExecutionRequest(
                source=source,
                database_path=self.database_path,
                inputs_directory=inputs_directory,
                output_directory=output_directory,
                expected_outputs=tuple(expected_outputs),
                environment={
                    "DSAGENT_DATABASE": str(self.database_path),
                    "DSAGENT_INPUTS": str(inputs_directory),
                    "DSAGENT_OUTPUTS": str(output_directory),
                },
            )
            try:
                async with asyncio.timeout(self.policy.max_python_seconds):
                    execution = await self.python_executor.execute(request)
            except TimeoutError:
                return self._tool_error("python_timeout", "Python execution timed out")
            except Exception as error:
                return self._tool_error(
                    "python_failed",
                    f"The isolated Python executor failed ({type(error).__name__})",
                )

            if output_directory.is_symlink() or not output_directory.is_dir():
                return self._tool_error(
                    "python_output_directory_invalid",
                    "The managed Python output directory was replaced",
                )
            actual_outputs = sorted(path.name for path in output_directory.iterdir())
            if actual_outputs != sorted(expected_outputs):
                return self._tool_error(
                    "python_outputs_mismatch",
                    "Python must create exactly the declared output files",
                )
            output_paths = [output_directory / name for name in expected_outputs]
            records = self._artifacts.publish_files(
                [
                    (path, _OUTPUT_MEDIA_TYPES[Path(name).suffix])
                    for name, path in zip(expected_outputs, output_paths, strict=True)
                ],
                producer_tool_call_id=tool_call_id,
                validator=lambda path, media_type: _validate_executor_output(
                    path,
                    media_type,
                    self.policy.max_artifact_bytes,
                    self.policy.max_tool_result_bytes,
                ),
            )
            return self._python_result(execution, records)
        except ArtifactError as error:
            return self._tool_error(error.code, error.message)
        finally:
            shutil.rmtree(staging)

    def load_json_artifact(self, handle: str) -> JsonValue:
        record, content = self._artifacts.read_verified_bytes(
            handle,
            max_bytes=self.policy.max_tool_result_bytes,
            expected_media_type=_JSON_MEDIA_TYPE,
            media_type_error=(
                "answer_artifact_media_type",
                "The final answer artifact must be JSON",
            ),
        )
        if record.size_bytes > self.policy.max_tool_result_bytes:
            raise ArtifactError(
                "answer_artifact_size_limit",
                "The final answer artifact exceeds the answer byte limit",
            )
        try:
            return cast(
                JsonValue,
                json.loads(
                    content,
                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ArtifactError(
                "answer_artifact_invalid_json",
                "The final answer artifact must contain finite UTF-8 JSON",
            ) from error

    def _inspect_database_sync(self, relation: str | None) -> dict[str, JsonValue]:
        connection = _open_connection(self.database_path, self.policy.max_query_memory_bytes)
        timed_out = Event()
        timer = _interrupt_after(connection, self.policy.max_inspection_seconds, timed_out)
        try:
            if relation is None:
                cursor = connection.execute(
                    "select table_schema, table_name, table_type "
                    "from information_schema.tables "
                    "where table_schema not in ('information_schema', 'pg_catalog') "
                    "order by table_schema, table_name"
                )
                return _stream_catalog(cursor, self._inspection_result_limit())
            selected = _parse_qualified_relation(relation)
            if selected is None:
                return _error_value(
                    "relation_not_found",
                    "Use an exact relation name returned by catalog inspection",
                )
            schema, name = selected
            matched = connection.execute(
                "select table_schema, table_name from information_schema.tables "
                "where table_schema = ? and table_name = ? "
                "and table_schema not in ('information_schema', 'pg_catalog') limit 1",
                [schema, name],
            ).fetchone()
            if matched is None:
                return _error_value(
                    "relation_not_found",
                    "Use an exact relation name returned by catalog inspection",
                )
            schema, name = matched
            cursor = connection.execute(
                "select column_name, data_type, is_nullable "
                "from information_schema.columns "
                "where table_schema = ? and table_name = ? order by ordinal_position",
                [schema, name],
            )
            return _stream_columns(
                cursor,
                _qualified_relation(schema, name),
                self._inspection_result_limit(),
            )
        except duckdb.Error as error:
            if timed_out.is_set():
                raise TimeoutError from error
            raise
        finally:
            timer.cancel()
            connection.close()

    def _inspection_result_limit(self) -> int:
        return min(
            self.policy.max_inspection_result_bytes,
            self.policy.max_tool_result_bytes,
        )

    def _check_database_sync(self) -> None:
        connection = _open_connection(self.database_path, self.policy.max_query_memory_bytes)
        try:
            connection.execute("select 1").fetchone()
        finally:
            connection.close()

    def _query_database_sync(self, sql: str) -> _QueryMaterialization:
        connection = _open_connection(self.database_path, self.policy.max_query_memory_bytes)
        timed_out = Event()
        timer = _interrupt_after(connection, self.policy.max_query_seconds, timed_out)
        try:
            bounded_sql = (
                "select * from query(?) as __dsa_query "
                f"limit {self.policy.max_query_rows + 1}"
            )
            _validate_bound_query(connection, bounded_sql, sql)
            cursor = connection.execute(bounded_sql, [sql])
            columns: tuple[dict[str, JsonValue], ...] = tuple(
                {"name": item[0], "type": str(item[1])}
                for item in cursor.description
            )
            reader = cursor.to_arrow_reader(batch_size=8_192)
            batches: list[pa.RecordBatch] = []
            row_count = 0
            result_bytes = 0
            for batch in reader:
                row_count += batch.num_rows
                result_bytes += batch.nbytes
                if row_count > self.policy.max_query_rows:
                    raise ArtifactError(
                        "query_row_limit",
                        "The complete result has too many rows. Filter or aggregate it further",
                    )
                if result_bytes > self.policy.max_query_result_bytes:
                    raise ArtifactError(
                        "query_result_size_limit",
                        "The complete result is too large. Filter or aggregate it further",
                    )
                batches.append(batch)
            table = pa.Table.from_batches(batches, schema=reader.schema)
            return _QueryMaterialization(columns=columns, table=table)
        except duckdb.Error as error:
            message = str(error).lower()
            if timed_out.is_set() or "interrupt" in message:
                raise TimeoutError from error
            if "memory" in message:
                raise MemoryError from error
            raise
        finally:
            timer.cancel()
            connection.close()

    def _tool_error(self, code: str, message: str) -> str:
        encoded = _canonical_json(_error_value(code, message))
        if len(encoded.encode("utf-8")) > self.policy.max_tool_result_bytes:
            raise ToolResultLimitExceeded("tool error exceeds the per-result byte limit")
        return encoded

    def _python_result(
        self,
        execution: PythonExecutionResult,
        records: Sequence[ArtifactRecord],
    ) -> str:
        diagnostic_budget = min(
            self.policy.max_python_output_bytes,
            self.policy.max_tool_result_bytes,
        )
        while True:
            stdout, stdout_truncated = _bounded_utf8(execution.stdout, diagnostic_budget)
            stderr_budget = max(0, diagnostic_budget - len(stdout.encode("utf-8")))
            stderr, stderr_truncated = _bounded_utf8(execution.stderr, stderr_budget)
            result = {
                "ok": True,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "outputs": [
                    {
                        "handle": record.handle,
                        "path": f"DSAGENT_INPUTS/{Path(record.relative_path).name}",
                        "media_type": record.media_type,
                        "size_bytes": record.size_bytes,
                    }
                    for record in records
                ],
            }
            encoded = _canonical_json(result)
            if len(encoded.encode("utf-8")) <= self.policy.max_tool_result_bytes:
                return encoded
            if diagnostic_budget == 0:
                raise ToolResultLimitExceeded(
                    "python result metadata exceeds the per-result byte limit"
                )
            diagnostic_budget //= 2


def _query_issue(sql: str) -> tuple[str, str] | None:
    if not sql.strip():
        return "query_blank", "SQL must not be blank"
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.Error:
        return "query_invalid", "SQL must be valid DuckDB syntax"
    if len(statements) != 1:
        return "query_statement_count", "Submit exactly one SQL statement"
    if statements[0].type != duckdb.StatementType.SELECT:
        return "query_not_read_only", "Only read-only tabular queries are allowed"
    return None


def _validate_bound_query(
    connection: duckdb.DuckDBPyConnection,
    bounded_sql: str,
    submitted_sql: str,
) -> None:
    _validate_function_identities(connection, submitted_sql)
    try:
        connection.execute("pragma disable_optimizer")
        try:
            row = connection.execute(
                f"explain (format json) {bounded_sql}",
                [submitted_sql],
            ).fetchone()
        finally:
            connection.execute("pragma enable_optimizer")
    except duckdb.PermissionException as error:
        raise ArtifactError(
            "query_external_access",
            "Queries may read only from the supplied database",
        ) from error
    if row is None or not isinstance(row[1], str):
        raise ArtifactError("query_plan_unavailable", "DuckDB could not bind the query plan")
    plan = cast(JsonValue, json.loads(row[1]))
    table_functions = _bound_table_functions(plan)
    if table_functions - _ALLOWED_BOUND_TABLE_FUNCTIONS:
        raise ArtifactError(
            "query_external_access",
            "Queries may scan only supplied database relations and pure generated tables",
        )


def _validate_function_identities(
    connection: duckdb.DuckDBPyConnection,
    submitted_sql: str,
) -> None:
    pending_asts = [_serialized_sql_ast(connection, submitted_sql)]
    expanded_macros: set[tuple[str, str, str]] = set()
    expanded_views: set[tuple[str, str]] = set()
    while pending_asts:
        ast = pending_asts.pop()
        for requested_schema, name in _function_identities(ast):
            function_conditions = ["lower(function_name) = ?"]
            function_parameters = [name]
            if requested_schema:
                function_conditions.append("lower(schema_name) = ?")
                function_parameters.append(requested_schema)
            side_effect_row = connection.execute(
                "select coalesce(bool_or(has_side_effects), false) "
                "from duckdb_functions() where " + " and ".join(function_conditions),
                function_parameters,
            ).fetchone()
            if name in _FORBIDDEN_BOUND_SCALAR_FUNCTIONS or (
                side_effect_row is not None and side_effect_row[0] is True
            ):
                raise ArtifactError(
                    "query_external_access",
                    "Queries may not call host metadata or side-effecting functions",
                )
            macro_conditions = [
                *function_conditions,
                "function_type in ('macro', 'table_macro')",
            ]
            macro_rows = connection.execute(
                "select lower(schema_name), function_type, macro_definition "
                "from duckdb_functions() where "
                + " and ".join(macro_conditions)
                + " order by database_name, schema_name, function_oid limit 101",
                function_parameters,
            ).fetchall()
            if len(macro_rows) > 100:
                raise ArtifactError(
                    "query_policy_complexity",
                    "The query references too many same-named macros to validate safely",
                )
            for schema, function_type, definition in macro_rows:
                identity = (cast(str, schema), name, cast(str, function_type))
                if identity in expanded_macros:
                    continue
                _check_policy_expansion(len(expanded_macros) + len(expanded_views), definition)
                expanded_macros.add(identity)
                macro_sql = (
                    definition if function_type == "table_macro" else f"select {definition}"
                )
                pending_asts.append(_serialized_sql_ast(connection, macro_sql))

        for requested_schema, name in _persistent_relation_identities(ast):
            conditions = ["lower(table_name) = ?"]
            parameters = [name]
            if requested_schema:
                conditions.append("lower(table_schema) = ?")
                parameters.append(requested_schema)
            view_rows = connection.execute(
                "select lower(table_schema), lower(table_name), view_definition "
                "from information_schema.views where "
                + " and ".join(conditions)
                + " order by table_schema, table_name limit 101",
                parameters,
            ).fetchall()
            if len(view_rows) > 100:
                raise ArtifactError(
                    "query_policy_complexity",
                    "The query references too many same-named views to validate safely",
                )
            for schema, view_name, definition in view_rows:
                identity = (cast(str, schema), cast(str, view_name))
                if identity in expanded_views:
                    continue
                _check_policy_expansion(len(expanded_macros) + len(expanded_views), definition)
                view_sql = _view_select_sql(cast(str, definition))
                if view_sql is None:
                    raise ArtifactError(
                        "query_policy_unavailable",
                        "DuckDB could not serialize a referenced view for policy validation",
                    )
                expanded_views.add(identity)
                pending_asts.append(_serialized_sql_ast(connection, view_sql))


def _check_policy_expansion(count: int, definition: object) -> None:
    if count >= 100 or not isinstance(definition, str):
        raise ArtifactError(
            "query_policy_complexity",
            "The query expansion is too complex to validate safely",
        )


def _serialized_sql_ast(
    connection: duckdb.DuckDBPyConnection,
    sql: str,
) -> JsonValue:
    row = connection.execute("select json_serialize_sql(?)", [sql]).fetchone()
    if row is None or not isinstance(row[0], str):
        raise ArtifactError(
            "query_policy_unavailable",
            "DuckDB could not serialize the query for policy validation",
        )
    parsed = cast(JsonValue, json.loads(row[0]))
    if not isinstance(parsed, dict) or parsed.get("error") is not False:
        raise ArtifactError(
            "query_policy_unavailable",
            "DuckDB could not serialize the query for policy validation",
        )
    return parsed


def _function_identities(value: JsonValue) -> set[tuple[str, str]]:
    identities: set[tuple[str, str]] = set()
    if isinstance(value, dict):
        mapping = cast(dict[str, JsonValue], value)
        if mapping.get("class") == "FUNCTION":
            name = mapping.get("function_name")
            schema = mapping.get("schema")
            if isinstance(name, str) and isinstance(schema, str):
                identities.add((schema.lower(), name.lower()))
        for child in mapping.values():
            identities.update(_function_identities(child))
    elif isinstance(value, list):
        for child in cast(list[JsonValue], value):
            identities.update(_function_identities(child))
    return identities


def _persistent_relation_identities(
    value: JsonValue,
    inherited_ctes: frozenset[str] = frozenset(),
) -> set[tuple[str, str]]:
    identities: set[tuple[str, str]] = set()
    if isinstance(value, dict):
        mapping = cast(dict[str, JsonValue], value)
        local_ctes = set(inherited_ctes)
        cte_map = mapping.get("cte_map")
        entries: list[JsonValue] = []
        if isinstance(cte_map, dict):
            raw_entries = cast(dict[str, JsonValue], cte_map).get("map")
            if isinstance(raw_entries, list):
                entries = cast(list[JsonValue], raw_entries)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_mapping = cast(dict[str, JsonValue], entry)
            key = entry_mapping.get("key")
            definition = entry_mapping.get("value")
            if not isinstance(key, str) or not isinstance(definition, dict):
                continue
            definition_mapping = cast(dict[str, JsonValue], definition)
            query = definition_mapping.get("query")
            definition_ctes = set(local_ctes)
            if _is_recursive_cte_query(query, key):
                definition_ctes.add(key.lower())
            if query is not None:
                identities.update(
                    _persistent_relation_identities(query, frozenset(definition_ctes))
                )
            local_ctes.add(key.lower())
        if mapping.get("type") == "BASE_TABLE":
            name = mapping.get("table_name")
            schema = mapping.get("schema_name")
            if (
                isinstance(name, str)
                and isinstance(schema, str)
                and (schema or name.lower() not in local_ctes)
            ):
                identities.add((schema.lower(), name.lower()))
        for key, child in mapping.items():
            if key != "cte_map":
                identities.update(
                    _persistent_relation_identities(child, frozenset(local_ctes))
                )
    elif isinstance(value, list):
        for child in cast(list[JsonValue], value):
            identities.update(_persistent_relation_identities(child, inherited_ctes))
    return identities


def _is_recursive_cte_query(value: JsonValue | None, name: str) -> bool:
    if not isinstance(value, dict):
        return False
    node = cast(dict[str, JsonValue], value).get("node")
    if not isinstance(node, dict):
        return False
    node_mapping = cast(dict[str, JsonValue], node)
    cte_name = node_mapping.get("cte_name")
    return (
        node_mapping.get("type") == "RECURSIVE_CTE_NODE"
        and isinstance(cte_name, str)
        and cte_name.lower() == name.lower()
    )


def _view_select_sql(definition: str) -> str | None:
    match = _VIEW_DEFINITION.fullmatch(definition)
    return match.group(1) if match is not None else None


def _bound_table_functions(value: JsonValue) -> set[str]:
    functions: set[str] = set()
    if isinstance(value, dict):
        mapping = cast(dict[str, JsonValue], value)
        extra = mapping.get("extra_info")
        if isinstance(extra, dict):
            function = cast(dict[str, JsonValue], extra).get("Function")
            if isinstance(function, str):
                functions.add(function.lower())
        for child in mapping.values():
            functions.update(_bound_table_functions(child))
    elif isinstance(value, list):
        for child in cast(list[JsonValue], value):
            functions.update(_bound_table_functions(child))
    return functions


def _qualified_relation(schema: str, name: str) -> str:
    return f"{_quote_identifier(schema)}.{_quote_identifier(name)}"


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _parse_qualified_relation(value: str) -> tuple[str, str] | None:
    if not value.startswith('"'):
        if value.count(".") != 1:
            return None
        schema, name = value.split(".", maxsplit=1)
        return (schema, name) if schema and name else None

    def parse_identifier(offset: int) -> tuple[str, int] | None:
        if offset >= len(value) or value[offset] != '"':
            return None
        offset += 1
        characters: list[str] = []
        while offset < len(value):
            character = value[offset]
            if character != '"':
                characters.append(character)
                offset += 1
                continue
            if offset + 1 < len(value) and value[offset + 1] == '"':
                characters.append('"')
                offset += 2
                continue
            return "".join(characters), offset + 1
        return None

    parsed_schema = parse_identifier(0)
    if parsed_schema is None:
        return None
    schema, offset = parsed_schema
    if offset >= len(value) or value[offset] != ".":
        return None
    parsed_name = parse_identifier(offset + 1)
    if parsed_name is None:
        return None
    name, offset = parsed_name
    if offset != len(value) or not schema or not name:
        return None
    return schema, name


def _python_request_issue(
    source: str,
    inputs: Sequence[str],
    expected_outputs: Sequence[str],
) -> tuple[str, str] | None:
    if not source.strip():
        return "python_source_blank", "Python source must not be blank"
    if len(set(inputs)) != len(inputs):
        return "python_inputs_duplicate", "Python input handles must be unique"
    if not expected_outputs:
        return "python_outputs_empty", "Declare at least one expected output file"
    if len(set(expected_outputs)) != len(expected_outputs):
        return "python_outputs_duplicate", "Expected output names must be unique"
    for name in expected_outputs:
        valid_name = _SAFE_OUTPUT_NAME.fullmatch(name) is not None
        if not valid_name or Path(name).suffix not in _OUTPUT_MEDIA_TYPES:
            return (
                "python_output_name_invalid",
                "Expected outputs must be safe .json or .parquet file names",
            )
    return None


def _open_connection(path: Path, memory_bytes: int) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(
        str(path),
        read_only=True,
        config={
            "enable_external_access": "false",
            "memory_limit": f"{memory_bytes}B",
        },
    )


def _interrupt_after(
    connection: duckdb.DuckDBPyConnection,
    seconds: int,
    timed_out: Event,
) -> Timer:
    def interrupt() -> None:
        timed_out.set()
        connection.interrupt()

    timer = Timer(seconds, interrupt)
    timer.daemon = True
    timer.start()
    return timer


def _table_rows(table: pa.Table) -> list[list[JsonValue]]:
    columns = [column.to_pylist() for column in table.columns]
    return [
        [_json_cell(columns[column][row], set()) for column in range(table.num_columns)]
        for row in range(table.num_rows)
    ]


def _json_cell(value: Any, seen: set[int]) -> JsonValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floating-point result")
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite decimal result")
        return {"$type": "decimal", "value": str(value)}
    if isinstance(value, datetime):
        return {"$type": "timestamp", "value": value.isoformat()}
    if isinstance(value, date):
        return {"$type": "date", "value": value.isoformat()}
    if isinstance(value, time):
        return {"$type": "time", "value": value.isoformat()}
    if isinstance(value, (bytes, bytearray)):
        return {"$type": "bytes", "base64": b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, memoryview):
        return {
            "$type": "bytes",
            "base64": b64encode(value.tobytes()).decode("ascii"),
        }
    if isinstance(value, UUID):
        return {"$type": "uuid", "value": str(value)}
    identity = id(value)
    if identity in seen:
        raise ValueError("cyclic query result")
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        if any(not isinstance(key, str) for key in mapping):
            raise TypeError("query result mappings must have string keys")
        seen.add(identity)
        try:
            return {
                cast(str, key): _json_cell(child, seen)
                for key, child in sorted(mapping.items(), key=lambda item: cast(str, item[0]))
            }
        finally:
            seen.remove(identity)
    if isinstance(value, (list, tuple)):
        sequence = cast(Sequence[object], value)
        seen.add(identity)
        try:
            return [_json_cell(child, seen) for child in sequence]
        finally:
            seen.remove(identity)
    raise TypeError(f"unsupported query result type: {type(value).__name__}")


def _stream_catalog(cursor: Any, limit: int) -> dict[str, JsonValue]:
    relations: list[JsonValue] = []
    while True:
        row = cursor.fetchone()
        if row is None:
            return {
                "ok": True,
                "complete": True,
                "relations": relations,
            }
        schema, name, kind = cast(tuple[str, str, str], row)
        item: dict[str, JsonValue] = {
            "name": _qualified_relation(schema, name),
            "type": kind,
        }
        candidate: dict[str, JsonValue] = {
            "ok": True,
            "complete": False,
            "relations": [*relations, item],
        }
        if len(_canonical_json(candidate).encode("utf-8")) > limit:
            return {
                "ok": True,
                "complete": False,
                "relations": relations,
            }
        relations.append(item)


def _stream_columns(cursor: Any, relation: str, limit: int) -> dict[str, JsonValue]:
    columns: list[JsonValue] = []
    while True:
        row = cursor.fetchone()
        if row is None:
            return {
                "ok": True,
                "complete": True,
                "relation": relation,
                "columns": columns,
            }
        name, kind, nullable = cast(tuple[str, str, str], row)
        item: dict[str, JsonValue] = {
            "name": name,
            "type": kind,
            "nullable": nullable == "YES",
        }
        candidate: dict[str, JsonValue] = {
            "ok": True,
            "complete": False,
            "relation": relation,
            "columns": [*columns, item],
        }
        if len(_canonical_json(candidate).encode("utf-8")) > limit:
            return {
                "ok": True,
                "complete": False,
                "relation": relation,
                "columns": columns,
            }
        columns.append(item)


def _bound_inspection_result(
    result: dict[str, JsonValue],
    limit: int,
    error_result: Any,
) -> str:
    encoded = _canonical_json(result)
    if len(encoded.encode("utf-8")) <= limit:
        return encoded
    collection_name = "relations" if "relations" in result else "columns"
    collection = result.get(collection_name)
    if not isinstance(collection, list):
        return error_result(
            "inspection_result_limit",
            "Inspection metadata exceeds its result byte limit",
        )
    kept: list[JsonValue] = []
    bounded = {**result, "complete": False, collection_name: kept}
    for item in collection:
        candidate = {**bounded, collection_name: [*kept, item]}
        if len(_canonical_json(candidate).encode("utf-8")) > limit:
            break
        kept.append(item)
    bounded[collection_name] = kept
    encoded = _canonical_json(bounded)
    if len(encoded.encode("utf-8")) > limit:
        return error_result(
            "inspection_result_limit",
            "Inspection metadata exceeds its result byte limit",
        )
    return encoded


def _error_value(code: str, message: str) -> dict[str, JsonValue]:
    return {"ok": False, "error": {"code": code, "message": message}}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _safe_duckdb_message(error: duckdb.Error, database_path: Path) -> str:
    message = str(error).replace(str(database_path), "<database>")
    return message if len(message) <= 1_000 else message[:1_000] + "…"


def _bounded_utf8(value: str, limit: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value, False
    return encoded[:limit].decode("utf-8", errors="ignore"), True


def _validate_executor_output(
    path: Path,
    media_type: str,
    max_bytes: int,
    max_json_bytes: int,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise ArtifactError(
            "python_output_invalid",
            "Python outputs must be regular files",
        )
    content_limit = min(max_bytes, max_json_bytes) if media_type == _JSON_MEDIA_TYPE else max_bytes
    if path.stat().st_size > content_limit:
        raise ArtifactError(
            "artifact_size_limit",
            "A complete output is too large to retain",
        )
    try:
        if media_type == _JSON_MEDIA_TYPE:
            json.loads(
                path.read_bytes(),
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
        else:
            parquet_file = parquet.ParquetFile(  # pyright: ignore[reportUnknownMemberType]
                path
            )
            for _batch in parquet_file.iter_batches(  # pyright: ignore[reportUnknownMemberType]
                batch_size=8_192
            ):
                pass
    except Exception as error:
        raise ArtifactError(
            "python_output_invalid",
            "Python output content does not match its declared file type",
        ) from error


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

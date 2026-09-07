"""Replayable derivation verification and deterministic notebook projection."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import stat
import sys
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from pydantic import JsonValue

from dsa.contract import Derivation, DerivationCodeCell, RunPolicy
from dsa.environment import AnalysisEnvironment, ArtifactError, PythonExecutor
from dsa.plots import MAX_NOTEBOOK_BYTES, MAX_PLOT_BYTES, VerifiedPlot, validate_png
from dsa.record import DerivationVerification, RetainedDerivationNotebook

_RESULT_NAME = "result.json"
_MAX_NOTEBOOK_BYTES = MAX_NOTEBOOK_BYTES
_INFRASTRUCTURE_CODES = frozenset(
    {
        "python_backend_error",
        "python_backend_unavailable",
        "python_container_cleanup",
        "python_failed",
        "python_workspace_cleanup_failed",
        "python_workspace_invalid",
    }
)


class DerivationError(RuntimeError):
    """Stable replay failure suitable for bounded model retry or terminalization."""

    def __init__(self, code: str, message: str, *, infrastructure: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.infrastructure = infrastructure


@dataclass(frozen=True)
class VerifiedDerivation:
    """Verified model content and the retained notebook projected from it."""

    derivation: Derivation
    verification: DerivationVerification
    notebook: RetainedDerivationNotebook


@dataclass(frozen=True)
class ReplayedDerivation:
    """Model content and JSON result replayed before notebook publication."""

    derivation: Derivation
    answer: JsonValue
    runtime_identity: str | None
    plots: tuple[VerifiedPlot, ...] = ()


async def verify_derivation(
    derivation: Derivation,
    answer: JsonValue,
    *,
    question: str,
    source_database: Path,
    source_database_sha256: str,
    run_directory: Path,
    policy: RunPolicy,
    python_executor: PythonExecutor | None,
    allow_plots: bool = False,
) -> VerifiedDerivation:
    """Replay cells from a pristine database and retain their notebook projection."""
    replayed = await replay_derivation(
        derivation,
        allow_plots=allow_plots,
        source_database=source_database,
        source_database_sha256=source_database_sha256,
        run_directory=run_directory,
        policy=policy,
        python_executor=python_executor,
    )
    answer_bytes = _canonical_json_bytes(answer)
    if _canonical_json_bytes(replayed.answer) != answer_bytes:
        raise DerivationError(
            "derivation_result_mismatch",
            "The derivation result does not exactly match the submitted answer",
        )
    return retain_replayed_derivation(
        replayed,
        question=question,
        source_database_sha256=source_database_sha256,
        run_directory=run_directory,
    )


async def replay_derivation(
    derivation: Derivation,
    *,
    source_database: Path,
    source_database_sha256: str,
    run_directory: Path,
    policy: RunPolicy,
    python_executor: PythonExecutor | None,
    allow_plots: bool = False,
) -> ReplayedDerivation:
    """Replay a derivation without publishing its notebook."""
    if derivation.plots and not allow_plots:
        raise DerivationError("derivation_plots_disabled", "Plots are not enabled for this request")
    replayed_answer, runtime_identity, plots = await _execute_derivation(
        derivation,
        source_database=source_database,
        source_database_sha256=source_database_sha256,
        run_directory=run_directory,
        policy=policy,
        python_executor=python_executor,
    )
    return ReplayedDerivation(
        derivation=derivation,
        answer=replayed_answer,
        runtime_identity=runtime_identity,
        plots=plots,
    )


def retain_replayed_derivation(
    replayed: ReplayedDerivation,
    *,
    question: str,
    source_database_sha256: str,
    run_directory: Path,
) -> VerifiedDerivation:
    """Publish a notebook only after the caller accepts the replayed answer."""
    return _retain_verified_derivation(
        replayed.derivation,
        answer_bytes=_canonical_json_bytes(replayed.answer),
        question=question,
        source_database_sha256=source_database_sha256,
        run_directory=run_directory,
        runtime_identity=replayed.runtime_identity,
        plots=replayed.plots,
    )


async def _execute_derivation(
    derivation: Derivation,
    *,
    source_database: Path,
    source_database_sha256: str,
    run_directory: Path,
    policy: RunPolicy,
    python_executor: PythonExecutor | None,
) -> tuple[JsonValue, str | None, tuple[VerifiedPlot, ...]]:
    """Replay cells from a pristine database without publishing a notebook."""
    if python_executor is None:
        raise DerivationError(
            "derivation_executor_unavailable",
            "A derivation requires an isolated Python executor",
            infrastructure=True,
        )
    replay_directory = run_directory / "work" / f"derivation-{uuid4().hex}"
    replay_database = replay_directory / "database.duckdb"
    runtime_identity: str | None = None
    try:
        replay_directory.mkdir(mode=0o700)
        _link_verified_source(
            source_database,
            replay_database,
            source_database_sha256,
        )
        environment = AnalysisEnvironment(
            database_path=replay_database,
            run_directory=replay_directory,
            policy=policy,
            python_executor=python_executor,
        )
        raw_result = await environment.run_python(
            _replay_source(derivation),
            (),
            (_RESULT_NAME, *(plot.filename for plot in derivation.plots)),
            tool_call_id="derivation-verification",
        )
        result = _parse_execution_result(raw_result)
        runtime = result.get("runtime_identity")
        if isinstance(runtime, str):
            runtime_identity = runtime
        if result.get("ok") is not True:
            error = result.get("error")
            code = error.get("code") if isinstance(error, dict) else None
            stable_code = code if isinstance(code, str) else "derivation_execution_failed"
            raise DerivationError(
                stable_code,
                "The derivation did not execute successfully",
                infrastructure=stable_code in _INFRASTRUCTURE_CODES,
            )
        outputs = result.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 1 + len(derivation.plots):
            raise DerivationError(
                "derivation_result_unavailable",
                "The derivation did not produce its required result",
            )
        output = outputs[0]
        handle = output.get("handle") if isinstance(output, dict) else None
        if not isinstance(handle, str):
            raise DerivationError(
                "derivation_result_unavailable",
                "The derivation did not produce its required result",
            )
        try:
            replayed_answer = environment.load_json_artifact(handle)
            retained_plots: list[VerifiedPlot] = []
            total = 0
            for declaration, item in zip(derivation.plots, outputs[1:], strict=True):
                plot_handle = item.get("handle") if isinstance(item, dict) else None
                if not isinstance(plot_handle, str):
                    raise ValueError("missing plot handle")
                content = environment.load_png_artifact(plot_handle)
                total += len(content)
                if total > MAX_PLOT_BYTES:
                    raise ValueError("combined plots exceed 5 MiB")
                retained_plots.append(VerifiedPlot(declaration, content))
        except ArtifactError as error:
            raise DerivationError(error.code, error.message) from error
        except (ValueError, OSError) as error:
            raise DerivationError(
                "derivation_plot_invalid", "Plots must be valid bounded PNGs"
            ) from error
    finally:
        active_exception = sys.exc_info()[0]
        try:
            if replay_directory.exists():
                shutil.rmtree(replay_directory)
        except OSError as error:
            if active_exception is None or not issubclass(
                active_exception,
                asyncio.CancelledError,
            ):
                raise DerivationError(
                    "derivation_workspace_cleanup_failed",
                    "The private derivation workspace could not be removed",
                    infrastructure=True,
                ) from error

    return replayed_answer, runtime_identity, tuple(retained_plots)


def _retain_verified_derivation(
    derivation: Derivation,
    *,
    answer_bytes: bytes,
    question: str,
    source_database_sha256: str,
    run_directory: Path,
    runtime_identity: str | None,
    plots: tuple[VerifiedPlot, ...] = (),
) -> VerifiedDerivation:
    derivation_bytes = _canonical_json_bytes(derivation.model_dump(mode="json"))
    try:
        if tuple(plot.declaration for plot in plots) != derivation.plots:
            raise ValueError("replayed plots must match their declarations")
        if sum(len(plot.content) for plot in plots) > MAX_PLOT_BYTES:
            raise ValueError("combined plot byte limit exceeded")
        for plot in plots:
            validate_png(plot.content)
        derivation_sha256 = sha256(derivation_bytes).hexdigest()
        notebook_bytes = _notebook_bytes(
            derivation,
            question=question,
            source_database_sha256=source_database_sha256,
            runtime_identity=runtime_identity,
            derivation_sha256=derivation_sha256,
            plots=plots,
        )
        notebook = RetainedDerivationNotebook(
            path=run_directory / "derivation.ipynb",
            sha256=sha256(notebook_bytes).hexdigest(),
            byte_length=len(notebook_bytes),
        )
        verification = DerivationVerification(
            derivation_sha256=derivation_sha256,
            result_sha256=sha256(answer_bytes).hexdigest(),
            source_database_sha256=source_database_sha256,
            runtime_identity=runtime_identity,
            notebook_sha256=notebook.sha256,
            notebook_byte_length=notebook.byte_length,
        )
        _write_notebook(run_directory, notebook_bytes)
    except (OSError, ValueError) as error:
        raise DerivationError(
            "derivation_notebook_failed",
            "The verified derivation notebook could not be retained",
            infrastructure=True,
        ) from error
    return VerifiedDerivation(
        derivation=derivation,
        verification=verification,
        notebook=notebook,
    )


def _replay_source(derivation: Derivation) -> str:
    sources = [cell.source for cell in derivation.cells if isinstance(cell, DerivationCodeCell)]
    encoded_sources = json.dumps(sources, ensure_ascii=False, allow_nan=False)
    return f"""\
import json as __dsa_json
import os as __dsa_os
from pathlib import Path as __dsa_Path

__dsa_namespace = {{
    "database_path": __dsa_Path(__dsa_os.environ["DSAGENT_DATABASE"]),
    "plot_directory": __dsa_Path(__dsa_os.environ["DSAGENT_OUTPUTS"]),
}}
__dsa_sources = {encoded_sources}
for __dsa_index, __dsa_source in enumerate(__dsa_sources, start=1):
    exec(
        compile(__dsa_source, f"<dsa-derivation-cell-{{__dsa_index}}>", "exec"),
        __dsa_namespace,
        __dsa_namespace,
    )
if "result" not in __dsa_namespace:
    raise RuntimeError("derivation must assign result")
with open(
    __dsa_os.path.join(__dsa_os.environ["DSAGENT_OUTPUTS"], "{_RESULT_NAME}"),
    "x",
    encoding="utf-8",
) as __dsa_output:
    __dsa_json.dump(
        __dsa_namespace["result"],
        __dsa_output,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    __dsa_output.write("\\n")
"""


def _parse_execution_result(value: str) -> dict[str, JsonValue]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise DerivationError(
            "derivation_protocol_invalid",
            "The derivation executor returned an invalid response",
            infrastructure=True,
        ) from error
    if not isinstance(parsed, dict):
        raise DerivationError(
            "derivation_protocol_invalid",
            "The derivation executor returned an invalid response",
            infrastructure=True,
        )
    return cast(dict[str, JsonValue], parsed)


def _link_verified_source(source: Path, destination: Path, expected_sha256: str) -> None:
    if source.is_symlink() or not source.is_file():
        raise DerivationError(
            "derivation_source_unavailable",
            "The private source database is unavailable for derivation replay",
            infrastructure=True,
        )
    descriptor: int | None = None
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("private source is not a regular file")
        digest = sha256()
        with os.fdopen(descriptor, "rb") as origin:
            descriptor = None
            while chunk := origin.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise DerivationError(
                "derivation_source_integrity",
                "The private source database failed its integrity check",
                infrastructure=True,
            )
        os.link(source, destination, follow_symlinks=False)
    except DerivationError:
        raise
    except OSError as error:
        destination.unlink(missing_ok=True)
        raise DerivationError(
            "derivation_source_unavailable",
            "The private source database is unavailable for derivation replay",
            infrastructure=True,
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not destination.is_file() or destination.is_symlink():
        destination.unlink(missing_ok=True)
        raise DerivationError(
            "derivation_source_unavailable",
            "The private source database is unavailable for derivation replay",
            infrastructure=True,
        )


def _notebook_bytes(
    derivation: Derivation,
    *,
    question: str,
    source_database_sha256: str,
    runtime_identity: str | None,
    derivation_sha256: str,
    plots: tuple[VerifiedPlot, ...] = (),
) -> bytes:
    cells: list[dict[str, Any]] = [
        _notebook_cell(
            "markdown",
            "# DSA verification derivation\n\n"
            f"**Question:** {question}\n\n"
            f"**Source database SHA-256:** `{source_database_sha256}`",
            0,
        ),
        _notebook_cell(
            "code",
            "from pathlib import Path\n\n"
            'database_path = Path("database.duckdb")'
            + (
                '\nplot_directory = Path("plots")\nplot_directory.mkdir(exist_ok=True)'
                if plots
                else ""
            ),
            1,
        ),
    ]
    for index, cell in enumerate(derivation.cells, start=2):
        cells.append(_notebook_cell(cell.type, cell.source, index))
    cells.append(_notebook_cell("code", "result", len(cells)))
    for plot in plots:
        # Captions are text/plain, never active Markdown or HTML.
        description = plot.declaration.title + (
            "\n" + plot.declaration.caption if plot.declaration.caption else ""
        )
        cell = _notebook_cell(
            "code",
            "from IPython.display import Image, display\n"
            f"display(Image(filename=str(plot_directory / {plot.declaration.filename!r})))\n"
            f"print({description!r})",
            len(cells),
        )
        cell["metadata"]["dsa_plot"] = plot.declaration.model_dump(mode="json")
        cell["outputs"] = [
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {
                    "image/png": base64.b64encode(plot.content).decode("ascii"),
                    "text/plain": description,
                },
            }
        ]
        cell["outputs"].append(
            {"output_type": "stream", "name": "stdout", "text": description + "\n"}
        )
        cells.append(cell)
    dsa_metadata: dict[str, JsonValue] = {
        "derivation_format": derivation.format,
        "derivation_sha256": derivation_sha256,
        "source_database_sha256": source_database_sha256,
    }
    if runtime_identity is not None:
        dsa_metadata["runtime_identity"] = runtime_identity
    notebook: dict[str, Any] = {
        "cells": cells,
        "metadata": {
            "dsa": dsa_metadata,
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    validator = cast(Callable[[object], None], import_module("nbformat").validate)
    validator(notebook)
    content = _canonical_json_bytes(cast(JsonValue, notebook)) + b"\n"
    if len(content) > _MAX_NOTEBOOK_BYTES:
        raise ValueError("derivation notebook exceeds its retained byte limit")
    return content


def _notebook_cell(cell_type: str, source: str, index: int) -> dict[str, Any]:
    identity = sha256(f"{index}:{cell_type}:{source}".encode()).hexdigest()[:16]
    cell: dict[str, Any] = {
        "cell_type": cell_type,
        "id": f"dsa-{index}-{identity}",
        "metadata": {},
        "source": source,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def _write_notebook(directory: Path, content: bytes) -> None:
    destination = directory / "derivation.ipynb"
    if destination.exists():
        raise FileExistsError("derivation notebook already exists")
    temporary = directory / f".derivation.{uuid4().hex}.tmp"
    descriptor: int | None = None
    published = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as target:
            descriptor = None
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, destination)
        published = True
        temporary.unlink()
        _fsync_directory(directory)
        metadata = destination.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("retained derivation notebook is not a regular file")
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        if published:
            destination.unlink(missing_ok=True)
        raise


def _canonical_json_bytes(value: JsonValue) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

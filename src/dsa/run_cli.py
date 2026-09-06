"""Standalone command-line boundary for exactly one DSA task."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from pathlib import Path
from typing import Any, cast

from dsa.cli import HelpRequested, Parser, canonical_json, emit, read_contract
from dsa.contract import RunRequest
from dsa.docker import DockerPythonExecutor, default_docker_configuration
from dsa.environment import PythonExecutor
from dsa.record import RunFailure, RunSuccess, validate_retained_derivation_notebook
from dsa.runner import RunCompletion, run_analysis

Runner = Callable[..., Coroutine[Any, Any, RunCompletion]]
ExecutorFactory = Callable[[str], PythonExecutor]
_MAX_INLINE_ANSWER_BYTES = 1024 * 1024


class _RunInterrupted(Exception):
    def __init__(self, completion: RunCompletion | None) -> None:
        super().__init__("standalone run interrupted")
        self.completion = completion


def main(
    argv: Sequence[str] | None = None,
    *,
    prog: str = "dsa-run",
    runner: Runner = run_analysis,
    executor_factory: ExecutorFactory | None = None,
) -> int:
    """Validate and execute one standalone analysis request."""
    try:
        parsed = _parser(prog).parse_args(argv)
        request_path = Path(parsed.request)
        request = _load_request(request_path)
        runs_directory = Path(parsed.runs_directory).resolve()
        create_executor = executor_factory or _docker_executor
        python_executor = create_executor(parsed.docker_image)
    except HelpRequested:
        return 0
    except Exception:
        emit({"code": "run_usage", "status": "rejected"})
        return 2

    try:
        completion = asyncio.run(
            _preserve_cancellation(
                runner(
                    request,
                    runs_directory=runs_directory,
                    python_executor=python_executor,
                    report_to_mlflow=parsed.report_to_mlflow,
                )
            )
        )
    except _RunInterrupted as interrupted:
        if interrupted.completion is None:
            emit({"code": "run_cancelled", "status": "failed"})
        else:
            emit(_result_projection(interrupted.completion))
        return 130
    except KeyboardInterrupt:
        return 130
    except Exception:
        emit({"code": "run_execution_failed", "status": "failed"})
        return 1

    try:
        result = _result_projection(completion)
    except Exception:
        emit({"code": "run_execution_failed", "status": "failed"})
        return 1
    emit(result)
    return 0 if isinstance(completion.outcome, RunSuccess) else 1


async def _preserve_cancellation(
    operation: Coroutine[Any, Any, RunCompletion],
) -> RunCompletion:
    try:
        return await operation
    except asyncio.CancelledError as error:
        raise _RunInterrupted(_cancelled_completion(error)) from None


def _cancelled_completion(error: asyncio.CancelledError) -> RunCompletion | None:
    retained_error = cast(Any, error)
    try:
        return RunCompletion.model_validate(
            {
                "record": retained_error.terminal_record,
                "retained_record": retained_error.retained_record,
                "reporting": getattr(retained_error, "reporting", {}),
            }
        )
    except Exception:
        return None


def _parser(prog: str = "dsa-run") -> Parser:
    parser = Parser(prog=prog, add_help=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--runs-directory", required=True)
    parser.add_argument("--docker-image", required=True)
    parser.add_argument("--report-to-mlflow", action="store_true")
    return parser


def _load_request(path: Path) -> RunRequest:
    content = read_contract(path)
    request = RunRequest.model_validate_json(content)
    canonical = canonical_json(request.model_dump(mode="json")).encode() + b"\n"
    if content != canonical:
        raise ValueError("request must use canonical JSON")
    database_path = request.database_path
    if not database_path.is_absolute():
        database_path = (path.parent / database_path).resolve()
    return RunRequest.model_validate(
        {
            **request.model_dump(mode="python", round_trip=True),
            "database_path": database_path,
        }
    )


def _docker_executor(image: str) -> DockerPythonExecutor:
    return DockerPythonExecutor(default_docker_configuration(image))


def _result_projection(completion: RunCompletion) -> dict[str, Any]:
    validate_retained_derivation_notebook(
        completion.record,
        completion.retained_record,
        completion.retained_notebook,
    )
    retained = completion.retained_record
    result: dict[str, Any] = {
        "reporting": completion.reporting.model_dump(mode="json", exclude_none=True),
        "run_id": completion.record.run_id,
        "status": completion.outcome.status,
        "terminal_record": {
            "byte_length": retained.byte_length,
            "path": str(retained.path),
            "sha256": retained.sha256,
        },
    }
    notebook = completion.retained_notebook
    if notebook is not None:
        result["derivation_notebook"] = {
            "byte_length": notebook.byte_length,
            "path": str(notebook.path),
            "sha256": notebook.sha256,
        }
    outcome = completion.outcome
    if isinstance(outcome, RunFailure):
        result["failure"] = {
            "code": outcome.failure.code,
            "stage": outcome.failure.stage,
        }
        return result

    answer_bytes = canonical_json(outcome.answer).encode()
    result["answer_inline"] = len(answer_bytes) <= _MAX_INLINE_ANSWER_BYTES
    if result["answer_inline"]:
        result["answer"] = outcome.answer
    return result


if __name__ == "__main__":
    raise SystemExit(main())

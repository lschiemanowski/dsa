"""Digest-pinned Docker execution for model-authored Python."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol, cast
from uuid import uuid4

from pydantic import Field, field_validator

from dsa.contract import ContractModel, PositiveInt
from dsa.environment import (
    PythonExecutionError,
    PythonExecutionRequest,
    PythonExecutionResult,
)

_CONTROL_OUTPUT_BYTES = 8 * 1024
_IMMUTABLE_IMAGE = re.compile(
    r"(?:sha256:|[^@\s]+@sha256:)[0-9a-f]{64}"
)
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_SERVER_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}")
_CONTAINER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}")
_CONTAINER_LABEL_KEY = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_CONTAINER_LABEL_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SLEEP_SOURCE = """\
import os
import shutil
import signal
import time

def reap_children(*_ignored):
    while True:
        try:
            child, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if child == 0:
            return

signal.signal(signal.SIGCHLD, reap_children)
shutil.copyfile('/seed/database.duckdb', '/database/database.duckdb')
open('/database/.ready', 'xb').close()
while True:
    time.sleep(3600)
"""
_WAIT_FOR_SEED_SOURCE = (
    "import pathlib,time; "
    "ready=pathlib.Path('/database/.ready'); "
    "deadline=time.monotonic()+30; "
    "exec(\"while not ready.exists():\\n"
    " if time.monotonic() >= deadline: raise SystemExit(1)\\n"
    " time.sleep(0.01)\"); "
    "ready.unlink()"
)
_CHECKPOINT_DATABASE_SOURCE = """\
import os
import signal
import stat
import time
from pathlib import Path

import duckdb

database_directory = Path('/database')
database = database_directory / 'database.duckdb'
os.chmod(database_directory, 0o700)
metadata = database.lstat()
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit(2)
os.chmod(database, 0o600)
self_pid = os.getpid()
deadline = time.monotonic() + 5
while True:
    foreign_pids = [
        int(entry.name)
        for entry in Path('/proc').iterdir()
        if entry.name.isdigit() and int(entry.name) not in (1, self_pid)
    ]
    if not foreign_pids:
        break
    for pid in foreign_pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if time.monotonic() >= deadline:
        raise SystemExit(3)
    time.sleep(0.01)
connection = duckdb.connect(
    str(database),
    config={'enable_external_access': 'false'},
)
try:
    connection.execute('force checkpoint')
finally:
    connection.close()
entries = list(database_directory.iterdir())
if len(entries) != 1 or entries[0].name != 'database.duckdb':
    raise SystemExit(4)
metadata = database.lstat()
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit(5)
"""
_VALIDATE_OUTPUTS_SOURCE = (
    "import json,os,sys; "
    "raise SystemExit(0 if sorted(os.listdir('/outputs')) == json.loads(sys.argv[1]) else 2)"
)
_STREAM_FILE_SOURCE = (
    "import shutil,sys; "
    "source=open(sys.argv[1],'rb',buffering=0); "
    "shutil.copyfileobj(source,sys.stdout.buffer,length=1048576)"
)


class DockerExecutorConfiguration(ContractModel):
    """Operator-owned immutable image and Docker control configuration."""

    image: str
    python_executable: str = "/usr/local/bin/python"
    user_id: PositiveInt
    group_id: Annotated[int, Field(ge=0)]
    docker_executable: str = "docker"
    control_timeout_seconds: PositiveInt = 10

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: str) -> str:
        if len(value) > 512 or _IMMUTABLE_IMAGE.fullmatch(value) is None:
            raise ValueError("image must be pinned by a lowercase SHA-256 digest")
        return value

    @field_validator("python_executable")
    @classmethod
    def validate_python_executable(cls, value: str) -> str:
        if not value.startswith("/") or "\x00" in value:
            raise ValueError("python executable must be an absolute container path")
        return value

    @field_validator("docker_executable")
    @classmethod
    def validate_docker_executable(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("docker executable must be nonblank")
        return value


@dataclass(frozen=True)
class DockerCommandResult:
    """Bounded result from one shell-free Docker CLI command."""

    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    timed_out: bool = False
    size_limit_exceeded: bool = False
    written_bytes: int = 0


class DockerCommandRunner(Protocol):
    async def run(
        self,
        arguments: Sequence[str],
        *,
        input_bytes: bytes | None,
        timeout_seconds: float,
        output_limit: int,
    ) -> DockerCommandResult: ...

    async def run_to_file(
        self,
        arguments: Sequence[str],
        *,
        destination: Path,
        timeout_seconds: float,
        output_limit: int,
    ) -> DockerCommandResult: ...


class AsyncSubprocessDockerRunner:
    """Run Docker without a shell while draining and bounding both output streams."""

    async def run(
        self,
        arguments: Sequence[str],
        *,
        input_bytes: bytes | None,
        timeout_seconds: float,
        output_limit: int,
    ) -> DockerCommandResult:
        try:
            process = await asyncio.create_subprocess_exec(
                *arguments,
                stdin=(asyncio.subprocess.PIPE if input_bytes is not None else None),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError:
            return DockerCommandResult(returncode=127)

        stdout_task = asyncio.create_task(_drain_stream(process.stdout, output_limit))
        stderr_task = asyncio.create_task(_drain_stream(process.stderr, output_limit))

        async def interact() -> None:
            if input_bytes is not None:
                assert process.stdin is not None
                process.stdin.write(input_bytes)
                await process.stdin.drain()
                process.stdin.close()
            await process.wait()

        timed_out = False
        cancel_streams = False
        try:
            async with asyncio.timeout(timeout_seconds):
                await interact()
        except TimeoutError:
            timed_out = True
            cancel_streams = True
            process.kill()
            await process.wait()
        except asyncio.CancelledError:
            cancel_streams = True
            process.kill()
            await process.wait()
            raise
        finally:
            if cancel_streams:
                stdout_task.cancel()
                stderr_task.cancel()
            stdout, stdout_truncated = await stdout_task
            stderr, stderr_truncated = await stderr_task

        return DockerCommandResult(
            returncode=process.returncode if process.returncode is not None else -1,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            timed_out=timed_out,
        )

    async def run_to_file(
        self,
        arguments: Sequence[str],
        *,
        destination: Path,
        timeout_seconds: float,
        output_limit: int,
    ) -> DockerCommandResult:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *arguments,
                stdin=None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except asyncio.CancelledError:
            os.close(descriptor)
            destination.unlink(missing_ok=True)
            raise
        except OSError:
            os.close(descriptor)
            destination.unlink(missing_ok=True)
            return DockerCommandResult(returncode=127)

        exceeded = asyncio.Event()
        stdout_task = asyncio.create_task(
            _drain_stream_to_descriptor(
                process.stdout,
                descriptor,
                output_limit,
                exceeded,
            )
        )
        stderr_task = asyncio.create_task(
            _drain_stream(process.stderr, _CONTROL_OUTPUT_BYTES)
        )
        wait_task = asyncio.create_task(process.wait())
        exceeded_task = asyncio.create_task(exceeded.wait())
        timed_out = False
        size_limit_exceeded = False
        cancel_streams = False
        try:
            async with asyncio.timeout(timeout_seconds):
                done, _pending = await asyncio.wait(
                    (wait_task, exceeded_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if exceeded_task in done and exceeded.is_set():
                    size_limit_exceeded = True
                    cancel_streams = True
                    if process.returncode is None:
                        process.kill()
                await process.wait()
        except TimeoutError:
            timed_out = True
            cancel_streams = True
            if process.returncode is None:
                process.kill()
            await process.wait()
        except asyncio.CancelledError:
            cancel_streams = True
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        finally:
            wait_task.cancel()
            exceeded_task.cancel()
            if cancel_streams:
                stdout_task.cancel()
                stderr_task.cancel()
            written_bytes = await stdout_task
            stderr, stderr_truncated = await stderr_task
            os.fsync(descriptor)
            os.close(descriptor)

        return DockerCommandResult(
            returncode=process.returncode if process.returncode is not None else -1,
            stderr=stderr,
            stderr_truncated=stderr_truncated,
            timed_out=timed_out,
            size_limit_exceeded=size_limit_exceeded,
            written_bytes=written_bytes,
        )


class DockerPythonExecutor:
    """Execute one Python request in a fresh constrained Docker container."""

    def __init__(
        self,
        configuration: DockerExecutorConfiguration,
        *,
        runner: DockerCommandRunner | None = None,
        container_name_factory: Callable[[], str] | None = None,
        container_labels: Mapping[str, str] | None = None,
    ) -> None:
        self.configuration = configuration
        self.runner = runner or AsyncSubprocessDockerRunner()
        self.container_name_factory = container_name_factory or (
            lambda: f"dsa-python-{uuid4().hex}"
        )
        labels = container_labels or {}
        if any(
            _CONTAINER_LABEL_KEY.fullmatch(key) is None
            or _CONTAINER_LABEL_VALUE.fullmatch(value) is None
            for key, value in labels.items()
        ):
            raise ValueError("Docker container labels must use safe bounded identities")
        self.container_labels = tuple(sorted(labels.items()))

    async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult:
        runtime_identity = await self._runtime_identity()
        if runtime_identity is None:
            raise PythonExecutionError(
                "python_backend_unavailable",
                "The Docker server or configured immutable image is unavailable",
            )
        name = self.container_name_factory()
        if _CONTAINER_NAME.fullmatch(name) is None:
            raise PythonExecutionError(
                "python_backend_error",
                "The host generated an invalid container identity",
                runtime_identity=runtime_identity,
            )
        database = request.database_path.resolve()
        inputs = request.inputs_directory.resolve()
        outputs = request.output_directory.resolve()
        if (
            request.database_path.name != "database.duckdb"
            or any("," in str(path) or ":" in str(path) for path in (database, inputs, outputs))
            or not _separate_mounts((database.parent, inputs, outputs))
        ):
            raise PythonExecutionError(
                "python_backend_error",
                "A managed mount path cannot be represented safely",
                runtime_identity=runtime_identity,
            )

        execution: DockerCommandResult | None = None
        failure: tuple[str, str] | None = None
        removed: DockerCommandResult | None = None
        create_started = False
        created = False
        export_database = database.parent / f".container.{uuid4().hex}.duckdb"
        try:
            database.chmod(0o444)
            create_started = True
            create = await self._control(
                self._create_arguments(name, request, database, inputs)
            )
            if create.timed_out or create.returncode != 0:
                failure = (
                    "python_backend_error",
                    "The isolated Python container could not be created",
                )
            else:
                created = True
            if failure is None:
                start = await self._control(
                    (self.configuration.docker_executable, "start", name)
                )
                if start.timed_out or start.returncode != 0:
                    failure = (
                        "python_backend_error",
                        "The isolated Python container could not be started",
                    )
            if failure is None:
                seeded = await self._control(
                    (
                        self.configuration.docker_executable,
                        "exec",
                        name,
                        self.configuration.python_executable,
                        "-I",
                        "-B",
                        "-c",
                        _WAIT_FOR_SEED_SOURCE,
                    )
                )
                database.chmod(0o600)
                if seeded.timed_out or seeded.returncode != 0:
                    failure = (
                        "python_backend_error",
                        "The private database could not initialize in the isolated container",
                    )
            if failure is None:
                execution = await self.runner.run(
                    (
                        self.configuration.docker_executable,
                        "exec",
                        "--interactive",
                        name,
                        self.configuration.python_executable,
                        "-I",
                        "-B",
                        "-",
                    ),
                    input_bytes=request.source.encode("utf-8"),
                    timeout_seconds=max(0.05, request.timeout_seconds - 0.5),
                    output_limit=request.output_bytes,
                )
            if execution is not None and execution.timed_out:
                failure = (
                    "python_timeout",
                    "Python execution exceeded its elapsed-time limit",
                )
                await self._control(
                    (self.configuration.docker_executable, "kill", name)
                )
            elif execution is not None:
                state = await self._container_state(name)
                if state is None:
                    failure = (
                        "python_backend_error",
                        "The isolated Python container state was unavailable",
                    )
                elif state[1]:
                    failure = (
                        "python_memory_limit",
                        "Python execution exceeded its memory limit",
                    )
                elif execution.returncode != 0 or not state[2]:
                    failure = (
                        "python_exit_nonzero",
                        "Python execution exited unsuccessfully",
                    )
            if failure is None:
                checkpointed = await self._control(
                    (
                        self.configuration.docker_executable,
                        "exec",
                        name,
                        self.configuration.python_executable,
                        "-I",
                        "-B",
                        "-c",
                        _CHECKPOINT_DATABASE_SOURCE,
                    )
                )
                if checkpointed.timed_out or checkpointed.returncode != 0:
                    failure = (
                        "python_backend_error",
                        "The private database could not be checkpointed safely",
                    )
            if failure is None:
                validated_outputs = await self._control(
                    (
                        self.configuration.docker_executable,
                        "exec",
                        name,
                        self.configuration.python_executable,
                        "-I",
                        "-B",
                        "-c",
                        _VALIDATE_OUTPUTS_SOURCE,
                        json.dumps(sorted(request.expected_outputs)),
                    )
                )
                if validated_outputs.timed_out or validated_outputs.returncode != 0:
                    failure = (
                        "python_backend_error",
                        "The isolated Python output set could not be recovered safely",
                    )
            if failure is None:
                streamed_database = await self.runner.run_to_file(
                    (
                        self.configuration.docker_executable,
                        "exec",
                        name,
                        self.configuration.python_executable,
                        "-I",
                        "-B",
                        "-c",
                        _STREAM_FILE_SOURCE,
                        "/database/database.duckdb",
                    ),
                    destination=export_database,
                    timeout_seconds=self.configuration.control_timeout_seconds,
                    output_limit=request.database_storage_bytes,
                )
                if (
                    streamed_database.timed_out
                    or streamed_database.size_limit_exceeded
                    or streamed_database.returncode != 0
                ):
                    failure = (
                        "python_backend_error",
                        "The private database could not be recovered within its limit",
                    )
                else:
                    remaining_output_bytes = request.output_storage_bytes
                    for expected_output in request.expected_outputs:
                        streamed_output = await self.runner.run_to_file(
                            (
                                self.configuration.docker_executable,
                                "exec",
                                name,
                                self.configuration.python_executable,
                                "-I",
                                "-B",
                                "-c",
                                _STREAM_FILE_SOURCE,
                                f"/outputs/{expected_output}",
                            ),
                            destination=outputs / expected_output,
                            timeout_seconds=self.configuration.control_timeout_seconds,
                            output_limit=remaining_output_bytes,
                        )
                        if (
                            streamed_output.timed_out
                            or streamed_output.size_limit_exceeded
                            or streamed_output.returncode != 0
                        ):
                            failure = (
                                "python_backend_error",
                                "An isolated Python output could not be recovered within "
                                "its limit",
                            )
                            break
                        remaining_output_bytes -= streamed_output.written_bytes
                    if failure is None:
                        os.replace(export_database, database)
        except OSError:
            failure = (
                "python_backend_error",
                "The isolated Python workspace could not be prepared safely",
            )
        finally:
            with suppress(OSError):
                database.chmod(0o600)
            with suppress(OSError):
                export_database.unlink(missing_ok=True)
            if create_started:
                removed = await self._remove_container(name)

        stdout = (
            execution.stdout.decode("utf-8", errors="replace")
            if execution is not None
            else ""
        )
        stderr = (
            execution.stderr.decode("utf-8", errors="replace")
            if execution is not None
            else ""
        )
        if created and (
            removed is None or removed.timed_out or removed.returncode != 0
        ):
            failure = (
                "python_container_cleanup",
                "The isolated Python container could not be removed",
            )
        if failure is not None:
            raise PythonExecutionError(
                failure[0],
                failure[1],
                stdout=stdout,
                stderr=stderr,
                stdout_truncated=(
                    execution.stdout_truncated if execution is not None else False
                ),
                stderr_truncated=(
                    execution.stderr_truncated if execution is not None else False
                ),
                runtime_identity=runtime_identity,
            )
        return PythonExecutionResult(
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=execution.stdout_truncated if execution is not None else False,
            stderr_truncated=execution.stderr_truncated if execution is not None else False,
            runtime_identity=runtime_identity,
        )

    async def _runtime_identity(self) -> str | None:
        executable = self.configuration.docker_executable
        version = await self._control(
            (executable, "version", "--format", "{{.Server.Version}}")
        )
        if version.timed_out or version.returncode != 0:
            return None
        server = version.stdout.decode("utf-8", errors="replace").strip()
        if _SERVER_VERSION.fullmatch(server) is None:
            return None
        inspected = await self._control(
            (
                executable,
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                self.configuration.image,
            )
        )
        if inspected.timed_out or inspected.returncode != 0:
            return None
        image_id = inspected.stdout.decode("utf-8", errors="replace").strip()
        if _IMAGE_ID.fullmatch(image_id) is None:
            return None
        if (
            self.configuration.image.startswith("sha256:")
            and image_id != self.configuration.image
        ):
            return None
        return f"docker/{server}|{self.configuration.image}|{image_id}"

    async def _container_state(self, name: str) -> tuple[int, bool, bool] | None:
        inspected = await self._control(
            (
                self.configuration.docker_executable,
                "inspect",
                "--format",
                "{{json .State}}",
                name,
            )
        )
        if inspected.timed_out or inspected.returncode != 0:
            return None
        try:
            state = json.loads(inspected.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(state, dict):
            return None
        mapping = cast(dict[str, object], state)
        exit_code = mapping.get("ExitCode")
        oom_killed = mapping.get("OOMKilled")
        running = mapping.get("Running")
        if (
            type(exit_code) is not int
            or type(oom_killed) is not bool
            or type(running) is not bool
        ):
            return None
        return exit_code, oom_killed, running

    async def _control(self, arguments: Sequence[str]) -> DockerCommandResult:
        return await self.runner.run(
            arguments,
            input_bytes=None,
            timeout_seconds=self.configuration.control_timeout_seconds,
            output_limit=_CONTROL_OUTPUT_BYTES,
        )

    async def _remove_container(self, name: str) -> DockerCommandResult:
        """Finish one bounded removal attempt even if this task is cancelled."""
        cleanup = asyncio.create_task(
            self._control(
                (self.configuration.docker_executable, "rm", "--force", name)
            )
        )
        try:
            return await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            with suppress(asyncio.CancelledError):
                await cleanup
            raise

    def _create_arguments(
        self,
        name: str,
        request: PythonExecutionRequest,
        database: Path,
        inputs: Path,
    ) -> tuple[str, ...]:
        labels = tuple(
            argument
            for key, value in self.container_labels
            for argument in ("--label", f"{key}={value}")
        )
        return (
            self.configuration.docker_executable,
            "create",
            "--name",
            name,
            "--pull",
            "never",
            "--log-driver",
            "none",
            "--interactive",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            f"{self.configuration.user_id}:{self.configuration.group_id}",
            "--ipc",
            "none",
            "--memory",
            str(request.memory_bytes),
            "--memory-swap",
            str(request.memory_bytes),
            "--cpus",
            str(request.cpu_count),
            "--pids-limit",
            str(request.process_limit),
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,noexec,mode=1777,size={request.scratch_bytes}",
            "--tmpfs",
            "/database:rw,nosuid,nodev,noexec,"
            f"uid={self.configuration.user_id},gid={self.configuration.group_id},"
            f"mode=0700,size={request.database_storage_bytes}",
            "--tmpfs",
            "/outputs:rw,nosuid,nodev,noexec,"
            f"uid={self.configuration.user_id},gid={self.configuration.group_id},"
            f"mode=0700,size={request.output_storage_bytes}",
            "--workdir",
            "/outputs",
            "--env",
            "PYTHONIOENCODING=utf-8",
            "--env",
            "PYTHONUNBUFFERED=1",
            "--env",
            "OPENBLAS_NUM_THREADS=1",
            "--env",
            "OMP_NUM_THREADS=1",
            "--env",
            "MKL_NUM_THREADS=1",
            "--env",
            "NUMEXPR_NUM_THREADS=1",
            "--env",
            "VECLIB_MAXIMUM_THREADS=1",
            "--env",
            "JOBLIB_MULTIPROCESSING=0",
            "--env",
            "LOKY_MAX_CPU_COUNT=1",
            "--env",
            "MALLOC_CONF=background_thread:false",
            "--env",
            "ARROW_DEFAULT_MEMORY_POOL=system",
            "--env",
            "DSAGENT_DATABASE=/database/database.duckdb",
            "--env",
            "DSAGENT_INPUTS=/inputs",
            "--env",
            "DSAGENT_OUTPUTS=/outputs",
            "--mount",
            f"type=bind,source={database},destination=/seed/database.duckdb,readonly",
            "--mount",
            f"type=bind,source={inputs},destination=/inputs,readonly",
            *labels,
            self.configuration.image,
            self.configuration.python_executable,
            "-I",
            "-B",
            "-c",
            _SLEEP_SOURCE,
        )


async def _drain_stream(
    stream: asyncio.StreamReader | None,
    limit: int,
) -> tuple[bytes, bool]:
    if stream is None:
        return b"", False
    retained = bytearray()
    truncated = False
    try:
        while chunk := await stream.read(64 * 1024):
            remaining = max(0, limit - len(retained))
            retained.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated = True
    except asyncio.CancelledError:
        # A killed attach client can otherwise wait forever for a still-running
        # container to close its inherited stream. Return the bounded prefix so
        # the executor can kill and remove that container in its own finally.
        pass
    return bytes(retained), truncated


async def _drain_stream_to_descriptor(
    stream: asyncio.StreamReader | None,
    descriptor: int,
    limit: int,
    exceeded: asyncio.Event,
) -> int:
    if stream is None:
        return 0
    written = 0
    try:
        while chunk := await stream.read(64 * 1024):
            remaining = max(0, limit - written)
            retained = chunk[:remaining]
            offset = 0
            while offset < len(retained):
                offset += os.write(descriptor, retained[offset:])
            written += len(retained)
            if len(chunk) > remaining:
                exceeded.set()
                break
    except asyncio.CancelledError:
        pass
    return written


def _separate_mounts(paths: tuple[Path, Path, Path]) -> bool:
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                return False
    return True


def default_docker_configuration(image: str) -> DockerExecutorConfiguration:
    """Use the invoking non-root operator identity for private bind mounts."""
    return DockerExecutorConfiguration(
        image=image,
        user_id=os.getuid(),
        group_id=os.getgid(),
    )

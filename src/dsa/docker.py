"""Digest-pinned Docker execution for model-authored Python."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Callable, Sequence
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


class DockerCommandRunner(Protocol):
    async def run(
        self,
        arguments: Sequence[str],
        *,
        input_bytes: bytes | None,
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


class DockerPythonExecutor:
    """Execute one Python request in a fresh constrained Docker container."""

    def __init__(
        self,
        configuration: DockerExecutorConfiguration,
        *,
        runner: DockerCommandRunner | None = None,
        container_name_factory: Callable[[], str] | None = None,
    ) -> None:
        self.configuration = configuration
        self.runner = runner or AsyncSubprocessDockerRunner()
        self.container_name_factory = container_name_factory or (
            lambda: f"dsa-python-{uuid4().hex}"
        )

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
        paths = (
            request.database_path.resolve().parent,
            request.inputs_directory.resolve(),
            request.output_directory.resolve(),
        )
        if (
            request.database_path.name != "database.duckdb"
            or any("," in str(path) for path in paths)
            or not _separate_mounts(paths)
        ):
            raise PythonExecutionError(
                "python_backend_error",
                "A managed mount path cannot be represented safely",
                runtime_identity=runtime_identity,
            )

        create = await self._control(self._create_arguments(name, request, paths))
        if create.timed_out or create.returncode != 0:
            raise PythonExecutionError(
                "python_backend_error",
                "The isolated Python container could not be created",
                runtime_identity=runtime_identity,
            )

        start: DockerCommandResult | None = None
        failure: tuple[str, str] | None = None
        removed: DockerCommandResult | None = None
        try:
            start = await self.runner.run(
                (
                    self.configuration.docker_executable,
                    "start",
                    "--attach",
                    "--interactive",
                    name,
                ),
                input_bytes=request.source.encode("utf-8"),
                timeout_seconds=max(0.05, request.timeout_seconds - 0.25),
                output_limit=request.output_bytes,
            )
            if start.timed_out:
                failure = (
                    "python_timeout",
                    "Python execution exceeded its elapsed-time limit",
                )
                await self._control(
                    (self.configuration.docker_executable, "kill", name)
                )
            else:
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
                elif state[0] != 0:
                    failure = (
                        "python_exit_nonzero",
                        "Python execution exited unsuccessfully",
                    )
        finally:
            removed = await self._control(
                (self.configuration.docker_executable, "rm", "--force", name)
            )

        assert start is not None
        assert removed is not None
        stdout = start.stdout.decode("utf-8", errors="replace")
        stderr = start.stderr.decode("utf-8", errors="replace")
        if removed.timed_out or removed.returncode != 0:
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
                stdout_truncated=start.stdout_truncated,
                stderr_truncated=start.stderr_truncated,
                runtime_identity=runtime_identity,
            )
        return PythonExecutionResult(
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=start.stdout_truncated,
            stderr_truncated=start.stderr_truncated,
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

    async def _container_state(self, name: str) -> tuple[int, bool] | None:
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
        if type(exit_code) is not int or type(oom_killed) is not bool:
            return None
        return exit_code, oom_killed

    async def _control(self, arguments: Sequence[str]) -> DockerCommandResult:
        return await self.runner.run(
            arguments,
            input_bytes=None,
            timeout_seconds=self.configuration.control_timeout_seconds,
            output_limit=_CONTROL_OUTPUT_BYTES,
        )

    def _create_arguments(
        self,
        name: str,
        request: PythonExecutionRequest,
        paths: tuple[Path, Path, Path],
    ) -> tuple[str, ...]:
        database, inputs, outputs = paths
        return (
            self.configuration.docker_executable,
            "create",
            "--name",
            name,
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
            f"type=bind,source={database},destination=/database",
            "--mount",
            f"type=bind,source={inputs},destination=/inputs,readonly",
            "--mount",
            f"type=bind,source={outputs},destination=/outputs",
            self.configuration.image,
            self.configuration.python_executable,
            "-I",
            "-B",
            "-",
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

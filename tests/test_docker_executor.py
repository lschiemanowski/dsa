from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import duckdb
import pyarrow.parquet as parquet
import pytest
from pydantic import ValidationError

from dsa import PythonExecutionError, PythonExecutionRequest, RunPolicy
from dsa.docker import (
    AsyncSubprocessDockerRunner,
    DockerCommandResult,
    DockerExecutorConfiguration,
    DockerPythonExecutor,
)
from dsa.environment import AnalysisEnvironment

IMAGE = "sha256:" + "1" * 64


class RecordingDockerRunner:
    def __init__(self, results: Sequence[DockerCommandResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[tuple[str, ...], bytes | None, float, int]] = []

    async def run(
        self,
        arguments: Sequence[str],
        *,
        input_bytes: bytes | None,
        timeout_seconds: float,
        output_limit: int,
    ) -> DockerCommandResult:
        self.calls.append(
            (tuple(arguments), input_bytes, timeout_seconds, output_limit)
        )
        if not self.results:
            raise AssertionError(f"unexpected Docker command: {arguments!r}")
        return self.results.pop(0)


def result(
    returncode: int = 0,
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
    timed_out: bool = False,
) -> DockerCommandResult:
    return DockerCommandResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        timed_out=timed_out,
    )


def successful_results(
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    state: dict[str, object] | None = None,
    remove: DockerCommandResult | None = None,
) -> list[DockerCommandResult]:
    terminal = state or {"ExitCode": 0, "OOMKilled": False}
    return [
        result(stdout=b"29.7.2\n"),
        result(stdout=(IMAGE + "\n").encode()),
        result(stdout=b"container-id\n"),
        result(stdout=stdout, stderr=stderr),
        result(stdout=json.dumps(terminal).encode()),
        remove or result(),
    ]


def configuration(**overrides: object) -> DockerExecutorConfiguration:
    values: dict[str, object] = {
        "image": IMAGE,
        "python_executable": "/usr/local/bin/python",
        "user_id": 1000,
        "group_id": 1000,
        "docker_executable": "docker",
        "control_timeout_seconds": 5,
    }
    values.update(overrides)
    return DockerExecutorConfiguration.model_validate(values)


def execution_request(tmp_path: Path) -> PythonExecutionRequest:
    tmp_path.mkdir(parents=True, exist_ok=True)
    database_directory = tmp_path / "database"
    database_directory.mkdir()
    database = database_directory / "database.duckdb"
    connection = duckdb.connect(str(database))
    connection.close()
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    return PythonExecutionRequest(
        source="print('ok')",
        database_path=database,
        inputs_directory=inputs,
        output_directory=outputs,
        expected_outputs=(),
        environment={
            "DSAGENT_DATABASE": "/database/database.duckdb",
            "DSAGENT_INPUTS": "/inputs",
            "DSAGENT_OUTPUTS": "/outputs",
        },
        timeout_seconds=3,
        memory_bytes=256 * 1024 * 1024,
        cpu_count=2,
        process_limit=32,
        scratch_bytes=16 * 1024 * 1024,
        output_bytes=1024,
    )


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"image": "python:latest"}, "image"),
        ({"python_executable": "python"}, "python_executable"),
        ({"user_id": 0}, "user_id"),
        ({"group_id": -1}, "group_id"),
        ({"docker_executable": ""}, "docker_executable"),
        ({"control_timeout_seconds": 0}, "control_timeout_seconds"),
    ],
)
def test_configuration_requires_immutable_nonroot_bounded_controls(
    overrides: dict[str, object],
    field: str,
) -> None:
    """Mutable images, root execution, and missing controls cannot enter the backend."""
    with pytest.raises(ValidationError) as caught:
        configuration(**overrides)

    assert (field,) in {tuple(error["loc"]) for error in caught.value.errors()}


async def test_executor_applies_complete_narrow_isolation_boundary(tmp_path: Path) -> None:
    """One shell-free create command contains every required control and only three mounts."""
    runner = RecordingDockerRunner(successful_results(stdout=b"ok\n"))
    executor = DockerPythonExecutor(
        configuration(),
        runner=runner,
        container_name_factory=lambda: "dsa-python-test",
    )
    request = execution_request(tmp_path)

    observation = await executor.execute(request)

    assert observation.stdout == "ok\n"
    assert observation.runtime_identity == f"docker/29.7.2|{IMAGE}|{IMAGE}"
    create = runner.calls[2][0]
    joined = " ".join(create)
    assert create[:2] == ("docker", "create")
    assert "--network none" in joined
    assert "--read-only" in create
    assert "--cap-drop ALL" in joined
    assert "--security-opt no-new-privileges" in joined
    assert "--user 1000:1000" in joined
    assert "--ipc none" in joined
    assert "--memory 268435456" in joined
    assert "--memory-swap 268435456" in joined
    assert "--cpus 2" in joined
    assert "--pids-limit 32" in joined
    assert "/tmp:rw,nosuid,nodev,noexec,mode=1777,size=16777216" in joined
    assert joined.count("type=bind,source=") == 3
    assert (
        f"source={request.database_path.resolve().parent},destination=/database" in joined
    )
    assert f"source={request.inputs_directory.resolve()},destination=/inputs,readonly" in joined
    assert f"source={request.output_directory.resolve()},destination=/outputs" in joined
    assert "/var/run/docker.sock" not in joined
    assert "--privileged" not in create
    assert "--pid" not in create
    assert create[-4:] == (IMAGE, "/usr/local/bin/python", "-I", "-B", "-")[-4:]
    assert runner.calls[3][0] == (
        "docker",
        "start",
        "--attach",
        "--interactive",
        "dsa-python-test",
    )
    assert runner.calls[3][1] == b"print('ok')"
    assert runner.calls[-1][0] == ("docker", "rm", "--force", "dsa-python-test")


async def test_executor_rejects_overlapping_managed_mounts(tmp_path: Path) -> None:
    """A database directory cannot expose inputs, outputs, or their parent tree."""
    runner = RecordingDockerRunner(
        [result(stdout=b"29.7.2\n"), result(stdout=(IMAGE + "\n").encode())]
    )
    executor = DockerPythonExecutor(configuration(), runner=runner)
    request = execution_request(tmp_path)
    overlapping = replace(
        request,
        output_directory=request.database_path.parent,
    )

    with pytest.raises(PythonExecutionError) as caught:
        await executor.execute(overlapping)

    assert caught.value.code == "python_backend_error"
    assert len(runner.calls) == 2
    assert runner.results == []


async def test_timeout_kills_and_removes_container_with_bounded_diagnostics(
    tmp_path: Path,
) -> None:
    """An elapsed-time breach explicitly kills and removes before returning failure."""
    runner = RecordingDockerRunner(
        [
            result(stdout=b"29.7.2\n"),
            result(stdout=(IMAGE + "\n").encode()),
            result(),
            result(stdout=b"partial", stdout_truncated=True, timed_out=True),
            result(),
            result(),
        ]
    )
    executor = DockerPythonExecutor(
        configuration(),
        runner=runner,
        container_name_factory=lambda: "dsa-python-timeout",
    )

    with pytest.raises(PythonExecutionError) as caught:
        await executor.execute(execution_request(tmp_path))

    assert caught.value.code == "python_timeout"
    assert caught.value.stdout == "partial"
    assert caught.value.stdout_truncated is True
    assert runner.calls[-2][0] == ("docker", "kill", "dsa-python-timeout")
    assert runner.calls[-1][0] == (
        "docker",
        "rm",
        "--force",
        "dsa-python-timeout",
    )


async def test_oom_and_cleanup_failures_are_distinct(tmp_path: Path) -> None:
    """OOM is actionable, while failed removal invalidates an otherwise successful call."""
    oom_runner = RecordingDockerRunner(
        successful_results(state={"ExitCode": 137, "OOMKilled": True})
    )
    cleanup_runner = RecordingDockerRunner(
        successful_results(remove=result(returncode=1, stderr=b"private daemon detail"))
    )

    with pytest.raises(PythonExecutionError) as oom:
        await DockerPythonExecutor(configuration(), runner=oom_runner).execute(
            execution_request(tmp_path / "oom")
        )
    with pytest.raises(PythonExecutionError) as cleanup:
        await DockerPythonExecutor(configuration(), runner=cleanup_runner).execute(
            execution_request(tmp_path / "cleanup")
        )

    assert oom.value.code == "python_memory_limit"
    assert cleanup.value.code == "python_container_cleanup"
    assert "private daemon detail" not in str(cleanup.value)


async def test_subprocess_runner_drains_but_bounds_both_streams() -> None:
    """Untrusted process output cannot accumulate unbounded host memory or block a pipe."""
    runner = AsyncSubprocessDockerRunner()
    command = (
        sys.executable,
        "-c",
        "import sys; print('x' * 100000); print('y' * 100000, file=sys.stderr)",
    )

    observation = await runner.run(
        command,
        input_bytes=None,
        timeout_seconds=5,
        output_limit=32,
    )

    assert observation.returncode == 0
    assert len(observation.stdout) == 32
    assert len(observation.stderr) == 32
    assert observation.stdout_truncated is True
    assert observation.stderr_truncated is True


@pytest.mark.integration
async def test_real_docker_is_nonroot_offline_narrow_and_transactional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real sandbox preserves only declared database and artifact effects."""
    image = os.environ.get("DSA_DOCKER_TEST_IMAGE")
    if image is None:
        pytest.skip("DSA_DOCKER_TEST_IMAGE is required for the real Docker tier")
    monkeypatch.setenv("DSA_HOST_SECRET_SENTINEL", "must-not-enter-container")
    source = tmp_path / "source.duckdb"
    connection = duckdb.connect(str(source))
    try:
        connection.execute("create table events(value integer)")
        connection.execute("insert into events values (1), (2)")
    finally:
        connection.close()
    source_before = source.read_bytes()
    run_directory = tmp_path / "run"
    work_directory = run_directory / "work"
    work_directory.mkdir(parents=True)
    working = work_directory / "database.duckdb"
    shutil.copyfile(source, working)
    container_name = "dsa-python-real-contract-smoke"
    executor = DockerPythonExecutor(
        DockerExecutorConfiguration(
            image=image,
            user_id=os.getuid(),
            group_id=os.getgid(),
        ),
        container_name_factory=lambda: container_name,
    )
    runtime = AnalysisEnvironment(
        database_path=working,
        run_directory=run_directory,
        policy=RunPolicy(
            max_preview_rows=1,
            max_python_seconds=20,
            max_python_memory_bytes=512 * 1024 * 1024,
        ),
        python_executor=executor,
    )
    query = json.loads(
        await runtime.query_database(
            "select * from events order by value",
            tool_call_id="query-real-docker-input",
        )
    )
    assert query["artifact_handle"] == "a1"
    source_code = """
import json
import os
import socket
from pathlib import Path
import duckdb
import pyarrow.parquet as parquet

print(f"uid={os.geteuid()}")
try:
    socket.create_connection(("1.1.1.1", 53), timeout=0.2)
except OSError:
    print("network=blocked")
else:
    print("network=open")
try:
    Path("/rootfs-probe").write_text("bad")
except OSError:
    print("rootfs=readonly")
else:
    print("rootfs=writable")
input_path = Path(os.environ["DSAGENT_INPUTS"]) / "a1.parquet"
print(f"input_rows={parquet.read_table(input_path).num_rows}")
try:
    input_path.write_bytes(b"bad")
except OSError:
    print("inputs=readonly")
else:
    print("inputs=writable")
print(f"host_home_visible={Path('/home/lothar').exists()}")
print(f"docker_socket_visible={Path('/var/run/docker.sock').exists()}")
print(f"host_secret_visible={'DSA_HOST_SECRET_SENTINEL' in os.environ}")
connection = duckdb.connect(os.environ["DSAGENT_DATABASE"])
connection.execute("insert into events values (3)")
connection.close()
output = Path(os.environ["DSAGENT_OUTPUTS"]) / "result.json"
output.write_text(json.dumps({"isolated": True}))
"""

    tool_result = json.loads(
        await runtime.run_python(
            source_code,
            inputs=["a1"],
            expected_outputs=["result.json"],
            tool_call_id="python-real-docker",
        )
    )
    assert tool_result["ok"] is True, tool_result
    assert tool_result["stdout"].splitlines() == [
        f"uid={os.getuid()}",
        "network=blocked",
        "rootfs=readonly",
        "input_rows=2",
        "inputs=readonly",
        "host_home_visible=False",
        "docker_socket_visible=False",
        "host_secret_visible=False",
    ]
    assert tool_result["runtime_identity"].startswith("docker/")
    assert runtime.load_json_artifact("a2") == {"isolated": True}
    connection = duckdb.connect(str(working), read_only=True)
    try:
        count = connection.execute("select count(*) from events").fetchone()
    finally:
        connection.close()
    assert count == (3,)
    assert source.read_bytes() == source_before
    assert parquet.read_table(  # pyright: ignore[reportUnknownMemberType]
        run_directory / runtime.artifact_records[0].relative_path
    ).num_rows == 2
    inspected = subprocess.run(
        ["docker", "inspect", container_name],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    assert inspected.returncode != 0


@pytest.mark.integration
async def test_real_docker_enforces_timeout_and_memory_with_rollback(
    tmp_path: Path,
) -> None:
    """Real cgroup and elapsed-time failures discard attempts and remove containers."""
    image = os.environ.get("DSA_DOCKER_TEST_IMAGE")
    if image is None:
        pytest.skip("DSA_DOCKER_TEST_IMAGE is required for the real Docker tier")
    source = tmp_path / "source.duckdb"
    connection = duckdb.connect(str(source))
    try:
        connection.execute("create table events(value integer)")
        connection.execute("insert into events values (1)")
    finally:
        connection.close()

    async def exercise(
        *,
        name: str,
        source_code: str,
        seconds: int,
        memory_bytes: int,
        expected_code: str,
    ) -> None:
        run_directory = tmp_path / name
        work_directory = run_directory / "work"
        work_directory.mkdir(parents=True)
        working = work_directory / "database.duckdb"
        shutil.copyfile(source, working)
        before = working.read_bytes()
        executor = DockerPythonExecutor(
            DockerExecutorConfiguration(
                image=image,
                user_id=os.getuid(),
                group_id=os.getgid(),
            ),
            container_name_factory=lambda: name,
        )
        runtime = AnalysisEnvironment(
            database_path=working,
            run_directory=run_directory,
            policy=RunPolicy(
                max_python_seconds=seconds,
                max_python_memory_bytes=memory_bytes,
            ),
            python_executor=executor,
        )

        result_value = json.loads(
            await runtime.run_python(
                source_code,
                inputs=[],
                expected_outputs=[],
                tool_call_id=f"python-{name}",
            )
        )

        assert result_value["ok"] is False
        assert result_value["error"]["code"] == expected_code
        assert working.read_bytes() == before
        assert runtime.artifact_records == ()
        inspected = subprocess.run(
            ["docker", "inspect", name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        assert inspected.returncode != 0

    await exercise(
        name="dsa-python-real-timeout",
        source_code=(
            "import duckdb, os\n"
            "connection = duckdb.connect(os.environ['DSAGENT_DATABASE'])\n"
            "connection.execute('insert into events values (2)')\n"
            "while True: pass\n"
        ),
        seconds=1,
        memory_bytes=256 * 1024 * 1024,
        expected_code="python_timeout",
    )
    await exercise(
        name="dsa-python-real-oom",
        source_code=(
            "chunks = []\n"
            "while True:\n"
            "    chunks.append(bytearray(8 * 1024 * 1024))\n"
        ),
        seconds=10,
        memory_bytes=64 * 1024 * 1024,
        expected_code="python_memory_limit",
    )

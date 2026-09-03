"""Host-owned adapter from the chat boundary to DSA's public run API."""

from __future__ import annotations

import os
import stat
from collections.abc import Awaitable, Callable
from hashlib import sha256
from pathlib import Path

from pydantic import Field, JsonValue, field_validator, model_validator

from apps.private_data_chat.contracts import (
    AnalysisRequest,
    AnalysisResult,
    AppContract,
    ArtifactIdentity,
)
from dsa import (
    DerivationRequest,
    DockerPythonExecutor,
    ModelConfiguration,
    RunCompletion,
    RunFailure,
    RunPolicy,
    RunRequest,
    RunSuccess,
    default_docker_configuration,
    run_analysis,
)
from dsa.environment import PythonExecutor
from dsa.record import RetainedDerivationNotebook, RetainedTerminalRecord

_MAX_NOTEBOOK_BYTES = 2 * 1024 * 1024


class DsaRuntimeConfiguration(AppContract):
    """Trusted configuration which is never derived from clarifier output."""

    data_source_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,127}$")
    database_path: Path
    runs_directory: Path
    trusted_model_name: str = Field(min_length=1, max_length=512)
    trusted_model_settings: dict[str, JsonValue] = Field(default_factory=dict)
    docker_image: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._/:@-]*@sha256:[0-9a-f]{64}$",
        max_length=512,
    )
    report_to_mlflow: bool = False
    policy: RunPolicy = RunPolicy()

    @field_validator("database_path", "runs_directory")
    @classmethod
    def require_absolute_paths(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("DSA runtime paths must be absolute")
        return value

    @field_validator("trusted_model_name")
    @classmethod
    def reject_blank_model_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("trusted model name must not be blank")
        return value

    @model_validator(mode="after")
    def validate_model_configuration(self) -> DsaRuntimeConfiguration:
        ModelConfiguration(name=self.trusted_model_name, settings=self.trusted_model_settings)
        return self


Runner = Callable[..., Awaitable[RunCompletion]]
ExecutorFactory = Callable[[str], PythonExecutor]


class DsaAnalysisExecutor:
    """Resolve all privileged fields locally and execute one derivation-enabled run."""

    def __init__(
        self,
        configuration: DsaRuntimeConfiguration,
        *,
        runner: Runner = run_analysis,
        executor_factory: ExecutorFactory | None = None,
    ) -> None:
        self.configuration = configuration.model_copy(deep=True)
        self._runner = runner
        self._executor_factory = executor_factory or _docker_executor
        self._notebooks: dict[str, RetainedDerivationNotebook] = {}

    async def execute(self, request: AnalysisRequest) -> AnalysisResult:
        if request.data_source_id != self.configuration.data_source_id:
            return _failed(request, "data_source_not_configured")
        try:
            dsa_request = RunRequest(
                database_path=self.configuration.database_path,
                question=request.question,
                answer_schema=request.answer_schema,
                derivation=DerivationRequest(),
                model=ModelConfiguration(
                    name=self.configuration.trusted_model_name,
                    settings=self.configuration.trusted_model_settings,
                ),
                policy=self.configuration.policy,
            )
            executor = self._executor_factory(self.configuration.docker_image)
            completion = await self._runner(
                dsa_request,
                runs_directory=self.configuration.runs_directory,
                python_executor=executor,
                identity_factory=lambda: request.run_id,
                report_to_mlflow=self.configuration.report_to_mlflow,
            )
            terminal = _artifact(completion.retained_record, "terminal", "application/json")
            outcome = completion.outcome
            if isinstance(outcome, RunFailure):
                return _failed(request, "dsa_run_failed", terminal=terminal)
            assert isinstance(outcome, RunSuccess)
            notebook = None
            if completion.retained_notebook is not None:
                notebook = _artifact(
                    completion.retained_notebook,
                    "notebook",
                    "application/x-ipynb+json",
                )
                self._notebooks[notebook.artifact_id] = completion.retained_notebook
            return AnalysisResult(
                run_id=request.run_id,
                proposal_id=request.proposal_id,
                status="succeeded",
                answer=outcome.answer,
                terminal=terminal,
                notebook=notebook,
            )
        except Exception:
            return _failed(request, "dsa_execution_unavailable")

    def read_notebook(self, identity: ArtifactIdentity) -> bytes:
        """Read the exact notebook retained by this executor invocation."""
        retained = self._notebooks.get(identity.artifact_id)
        if retained is None or (
            retained.sha256 != identity.sha256
            or retained.byte_length != identity.byte_length
            or identity.media_type != "application/x-ipynb+json"
        ):
            raise ValueError("notebook artifact is unavailable")
        if retained.byte_length > _MAX_NOTEBOOK_BYTES:
            raise ValueError("notebook artifact exceeds the application limit")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(retained.path, flags)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != retained.byte_length:
                raise ValueError("notebook artifact identity changed")
            content = b""
            while len(content) <= retained.byte_length:
                chunk = os.read(descriptor, min(64 * 1024, retained.byte_length + 1 - len(content)))
                if not chunk:
                    break
                content += chunk
        finally:
            os.close(descriptor)
        if len(content) != retained.byte_length or sha256(content).hexdigest() != retained.sha256:
            raise ValueError("notebook artifact identity changed")
        return content


def _docker_executor(image: str) -> DockerPythonExecutor:
    return DockerPythonExecutor(default_docker_configuration(image))


def _artifact(
    retained: RetainedTerminalRecord | RetainedDerivationNotebook,
    kind: str,
    media_type: str,
) -> ArtifactIdentity:
    return ArtifactIdentity(
        artifact_id=f"artifact-{kind}-{retained.sha256[:32]}",
        sha256=retained.sha256,
        byte_length=retained.byte_length,
        media_type=media_type,
    )


def _failed(
    request: AnalysisRequest,
    code: str,
    *,
    terminal: ArtifactIdentity | None = None,
) -> AnalysisResult:
    return AnalysisResult(
        run_id=request.run_id,
        proposal_id=request.proposal_id,
        status="failed",
        terminal=terminal,
        failure_code=code,
    )

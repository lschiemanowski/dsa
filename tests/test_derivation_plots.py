from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from dsa import RunRequest, RunSuccess, run_analysis
from dsa.contract import Derivation, DerivationPlot, DerivationRequest, RunPolicy
from dsa.derivation import DerivationError, verify_derivation
from dsa.docker import DockerExecutorConfiguration, DockerPythonExecutor
from dsa.environment import PythonExecutionRequest, PythonExecutionResult
from dsa.plots import notebook_plots, validate_png
from tests.test_derivation import ResultExecutor, database, sample_derivation
from tests.test_episode import expected_derivation_receipt, valid_request


def png(size: tuple[int, int] = (20, 20)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, "white").save(output, format="PNG")
    return output.getvalue()


class PlotExecutor(ResultExecutor):
    def __init__(self, content: bytes) -> None:
        super().__init__({"count": 3})
        self.content = content

    async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult:
        result = await super().execute(request)
        (request.output_directory / "counts.png").write_bytes(self.content)
        return result


def plotted_derivation() -> Derivation:
    raw = sample_derivation().model_dump(mode="json")
    raw["plots"] = [{"filename": "counts.png", "title": "Counts", "caption": "Three rows."}]
    return Derivation.model_validate(raw)


async def test_replay_embeds_exact_verified_png(tmp_path: Path) -> None:
    source, digest = database(tmp_path)
    run = tmp_path / "run"
    (run / "work").mkdir(parents=True)
    content = png()
    verified = await verify_derivation(
        plotted_derivation(),
        {"count": 3},
        question="Count rows",
        source_database=source,
        source_database_sha256=digest,
        run_directory=run,
        policy=RunPolicy(),
        python_executor=PlotExecutor(content),
        allow_plots=True,
    )
    notebook = verified.notebook.path.read_bytes()
    plots = notebook_plots(notebook)
    assert len(plots) == 1
    assert plots[0].content == content
    assert plots[0].declaration.filename == "counts.png"
    assert "plot_directory" in json.dumps(json.loads(notebook)["cells"])
    assert not list((run / "work").iterdir())


async def test_plots_require_explicit_opt_in(tmp_path: Path) -> None:
    executor = PlotExecutor(png())
    with pytest.raises(DerivationError, match="not enabled"):
        await verify_derivation(
            plotted_derivation(),
            {"count": 3},
            question="Count rows",
            source_database=tmp_path / "unused",
            source_database_sha256="0" * 64,
            run_directory=tmp_path,
            policy=RunPolicy(),
            python_executor=executor,
        )
    assert not executor.requests
    assert DerivationRequest().model_dump() == {"format": "dsa-derivation/v1"}


@pytest.mark.parametrize("content", [b"not png", png((2049, 1)), png()[:-20]])
def test_invalid_png_rejected(content: bytes) -> None:
    with pytest.raises((ValueError, OSError, SyntaxError)):
        validate_png(content)


def test_plot_declaration_rejects_unsafe_names_and_duplicates() -> None:
    with pytest.raises(ValidationError):
        DerivationPlot(filename="../x.png", title="Unsafe")
    raw = plotted_derivation().model_dump(mode="json")
    raw["plots"] *= 2
    with pytest.raises(ValidationError, match="distinct"):
        Derivation.model_validate(raw)


@pytest.mark.parametrize("content", [b"not png", png((2049, 1)), b"x" * (5 * 1024 * 1024 + 1)])
async def test_invalid_replay_plot_leaves_no_notebook(tmp_path: Path, content: bytes) -> None:
    source, digest = database(tmp_path)
    run = tmp_path / "run"
    (run / "work").mkdir(parents=True)
    with pytest.raises(DerivationError) as error:
        await verify_derivation(
            plotted_derivation(),
            {"count": 3},
            question="Count rows",
            source_database=source,
            source_database_sha256=digest,
            run_directory=run,
            policy=RunPolicy(),
            python_executor=PlotExecutor(content),
            allow_plots=True,
        )
    assert not error.value.infrastructure
    assert not (run / "derivation.ipynb").exists()
    assert not list((run / "work").iterdir())


async def test_combined_plot_limit_is_not_a_per_image_limit(tmp_path: Path) -> None:
    source, digest = database(tmp_path)
    run = tmp_path / "run"
    (run / "work").mkdir(parents=True)
    output = io.BytesIO()
    Image.frombytes("RGB", (1100, 900), os.urandom(1100 * 900 * 3)).save(output, format="PNG")
    content = output.getvalue()
    assert 2.5 * 1024 * 1024 < len(content) < 5 * 1024 * 1024
    raw = plotted_derivation().model_dump(mode="json")
    raw["plots"].append({"filename": "second.png", "title": "Second"})

    class TwoPlots(PlotExecutor):
        async def execute(self, request: PythonExecutionRequest) -> PythonExecutionResult:
            result = await super().execute(request)
            (request.output_directory / "second.png").write_bytes(content)
            return result

    with pytest.raises(DerivationError) as error:
        await verify_derivation(
            Derivation.model_validate(raw),
            {"count": 3},
            question="Count rows",
            source_database=source,
            source_database_sha256=digest,
            run_directory=run,
            policy=RunPolicy(),
            python_executor=TwoPlots(content),
            allow_plots=True,
        )
    assert error.value.code == "derivation_plot_invalid"
    assert not (run / "derivation.ipynb").exists()
    assert not list((run / "work").iterdir())


async def test_plot_answer_mismatch_does_not_publish_notebook(tmp_path: Path) -> None:
    source, digest = database(tmp_path)
    run = tmp_path / "run"
    (run / "work").mkdir(parents=True)
    with pytest.raises(DerivationError) as error:
        await verify_derivation(
            plotted_derivation(),
            {"count": 4},
            question="Count rows",
            source_database=source,
            source_database_sha256=digest,
            run_directory=run,
            policy=RunPolicy(),
            python_executor=PlotExecutor(png()),
            allow_plots=True,
        )
    assert error.value.code == "derivation_result_mismatch"
    assert not (run / "derivation.ipynb").exists()


async def test_missing_declared_plot_is_retryable(tmp_path: Path) -> None:
    source, digest = database(tmp_path)
    run = tmp_path / "run"
    (run / "work").mkdir(parents=True)
    with pytest.raises(DerivationError) as error:
        await verify_derivation(
            plotted_derivation(),
            {"count": 3},
            question="Count rows",
            source_database=source,
            source_database_sha256=digest,
            run_directory=run,
            policy=RunPolicy(),
            python_executor=ResultExecutor({"count": 3}),
            allow_plots=True,
        )
    assert not error.value.infrastructure
    assert not (run / "derivation.ipynb").exists()


async def test_runner_retains_plot_notebook_bound_to_success(tmp_path: Path) -> None:
    request = RunRequest.model_validate(
        {
            **valid_request(tmp_path).model_dump(mode="python"),
            "derivation": {"allow_plots": True},
        }
    )
    derivation = plotted_derivation().model_dump(mode="json")
    receipt = expected_derivation_receipt(request, derivation, {"count": 3})
    calls = 0

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart("validate_derivation", derivation)])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "final_answer",
                    {"answer": {"count": 3}, "derivation_receipt": receipt},
                )
            ]
        )

    completion = await run_analysis(
        request,
        runs_directory=tmp_path / "runs",
        model=FunctionModel(respond),
        python_executor=PlotExecutor(png()),
    )
    assert isinstance(completion.outcome, RunSuccess)
    assert completion.outcome.derivation_verification is not None
    assert completion.retained_notebook is not None
    assert len(notebook_plots(completion.retained_notebook.path.read_bytes())) == 1
    assert calls == 2


@pytest.mark.integration
async def test_real_docker_replays_matplotlib_plot(tmp_path: Path) -> None:
    image = os.environ.get("DSA_DOCKER_TEST_IMAGE")
    if image is None:
        pytest.skip("DSA_DOCKER_TEST_IMAGE is required")
    source, digest = database(tmp_path)
    run = tmp_path / "run"
    (run / "work").mkdir(parents=True)
    derivation = Derivation.model_validate(
        {
            "cells": [
                {"type": "markdown", "source": "Count events and show the total."},
                {
                    "type": "code",
                    "source": (
                        "import duckdb\nimport matplotlib\nmatplotlib.use('Agg')\n"
                        "import matplotlib.pyplot as plt\n"
                        "with duckdb.connect(str(database_path), read_only=True) as db:\n"
                        "    count = db.execute('select count(*) from events').fetchone()[0]\n"
                        "fig, ax = plt.subplots(figsize=(4, 3))\n"
                        "ax.bar(['Events'], [count])\nax.set_ylabel('Number of rows')\n"
                        "fig.savefig(plot_directory / 'counts.png', dpi=100)\nplt.close(fig)\n"
                        "result = {'count': count}"
                    ),
                },
            ],
            "plots": [{"filename": "counts.png", "title": "Event count"}],
        }
    )
    verified = await verify_derivation(
        derivation,
        {"count": 3},
        question="Count rows",
        source_database=source,
        source_database_sha256=digest,
        run_directory=run,
        policy=RunPolicy(max_python_seconds=30),
        allow_plots=True,
        python_executor=DockerPythonExecutor(
            DockerExecutorConfiguration(
                image=image,
                user_id=os.getuid(),
                group_id=os.getgid(),
            )
        ),
    )
    plots = notebook_plots(verified.notebook.path.read_bytes())
    assert len(plots) == 1
    with Image.open(io.BytesIO(plots[0].content)) as rendered:
        assert rendered.size == (400, 300)

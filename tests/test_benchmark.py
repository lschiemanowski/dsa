"""Behavioral contracts for isolated benchmark studies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dsa.benchmark import (
    BenchmarkRuntime,
    BenchmarkStudy,
    expand_benchmark_study,
    prepare_benchmark,
)
from dsa.pack import LoadedEvaluationPack

REVISION = "a" * 40
IMAGE = "sha256:" + "b" * 64


def reference(pack: str) -> dict[str, str]:
    return {
        "repo_id": "lschiemanowski/dsa-datasets",
        "revision": "c" * 40,
        "path": f"{pack}/1.0.0",
        "manifest_sha256": "d" * 64,
    }


def study_value(**updates: object) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": "dsa-benchmark-study/v1",
        "study_id": "online-retail-model-comparison",
        "version": "1.0.0",
        "agent_revision": REVISION,
        "packs": (
            {"pack_id": "pack-b", "reference": reference("pack-b")},
            {"pack_id": "pack-a", "reference": reference("pack-a")},
        ),
        "models": (
            {
                "model_id": "model-b",
                "configuration": {"name": "test", "settings": {"temperature": 0}},
            },
            {
                "model_id": "model-a",
                "configuration": {"name": "test", "settings": {}},
            },
        ),
        "policy": {},
        "execution": {
            "docker_image": IMAGE,
            "case_workers": 1,
            "cell_workers": 1,
        },
        "repetitions": 2,
    }
    value.update(updates)
    return value


def runtime_value(tmp_path: Path, **updates: object) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": "dsa-benchmark-runtime/v1",
        "workspace_root": tmp_path / "study",
        "datasets": (
            {"pack_id": "pack-b", "dataset_name": "catalog.schema.pack_b"},
            {"pack_id": "pack-a", "dataset_name": "catalog.schema.pack_a"},
        ),
    }
    value.update(updates)
    return value


def test_study_is_canonical_ordered_and_content_addressed() -> None:
    study = BenchmarkStudy.model_validate(study_value())
    reordered = BenchmarkStudy.model_validate(
        study_value(
            packs=tuple(reversed(study_value()["packs"])),
            models=tuple(reversed(study_value()["models"])),
        )
    )

    assert tuple(item.pack_id for item in study.packs) == ("pack-a", "pack-b")
    assert tuple(item.model_id for item in study.models) == ("model-a", "model-b")
    assert study.canonical_json.endswith("\n")
    assert study.sha256 == reordered.sha256
    assert len(study.sha256) == 64


@pytest.mark.parametrize(
    "updates",
    [
        {"agent_revision": "main"},
        {"repetitions": 0},
        {"packs": []},
        {"models": []},
        {
            "execution": {
                "docker_image": "dsa:latest",
                "case_workers": 1,
                "cell_workers": 1,
            }
        },
        {
            "execution": {
                "docker_image": IMAGE,
                "case_workers": 0,
                "cell_workers": 1,
            }
        },
    ],
)
def test_study_rejects_mutable_or_unbounded_execution(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        BenchmarkStudy.model_validate(study_value(**updates))


def test_study_rejects_duplicate_pack_and_model_identities() -> None:
    value = study_value()
    value["packs"] = (value["packs"][0], value["packs"][0])
    with pytest.raises(ValidationError, match="pack identities must be unique"):
        BenchmarkStudy.model_validate(value)

    value = study_value()
    value["models"] = (value["models"][0], value["models"][0])
    with pytest.raises(ValidationError, match="model identities must be unique"):
        BenchmarkStudy.model_validate(value)


def test_study_bounds_the_expanded_cell_matrix() -> None:
    packs = tuple(
        {"pack_id": f"pack-{index:02d}", "reference": reference(f"pack-{index:02d}")}
        for index in range(64)
    )

    with pytest.raises(ValidationError, match="too many cells"):
        BenchmarkStudy.model_validate(
            study_value(packs=packs, repetitions=100)
        )


def test_study_bounds_cell_wide_concurrent_container_cleanup() -> None:
    with pytest.raises(ValidationError, match="container concurrency is too large"):
        BenchmarkStudy.model_validate(
            study_value(
                policy={"max_tool_calls": 157},
                execution={
                    "docker_image": IMAGE,
                    "case_workers": 64,
                    "cell_workers": 1,
                },
            )
        )


def test_runtime_rejects_dataset_values_that_cannot_be_safely_retained(
    tmp_path: Path,
) -> None:
    for dataset_name in (
        "https://private.example/?token=secret",
        "catalog..table",
        "catalog.schema.table.extra",
        "Catalog.schema.table",
    ):
        with pytest.raises(ValidationError):
            BenchmarkRuntime.model_validate(
                runtime_value(
                    tmp_path,
                    datasets=(
                        {
                            "pack_id": "pack-a",
                            "dataset_name": dataset_name,
                        },
                    ),
                )
            )


def test_runtime_accepts_safe_local_dataset_names(tmp_path: Path) -> None:
    runtime = BenchmarkRuntime.model_validate(
        runtime_value(
            tmp_path,
            datasets=({"pack_id": "pack-a", "dataset_name": "local_dataset"},),
        )
    )

    assert runtime.datasets[0].dataset_name == "local_dataset"


def test_runtime_requires_a_distinct_mlflow_dataset_for_each_pack(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="dataset names must be unique"):
        BenchmarkRuntime.model_validate(
            runtime_value(
                tmp_path,
                datasets=(
                    {"pack_id": "pack-a", "dataset_name": "catalog.schema.shared"},
                    {"pack_id": "pack-b", "dataset_name": "catalog.schema.shared"},
                ),
            )
        )


def test_cell_expansion_is_the_exact_ordered_cartesian_product() -> None:
    study = BenchmarkStudy.model_validate(study_value())

    cells = expand_benchmark_study(study)

    assert [
        (cell.pack_id, cell.model_id, cell.repetition) for cell in cells
    ] == [
        (pack, model, repetition)
        for pack in ("pack-a", "pack-b")
        for model in ("model-a", "model-b")
        for repetition in range(2)
    ]
    assert len({cell.cell_id for cell in cells}) == 8
    assert all(cell.study_sha256 == study.sha256 for cell in cells)


def test_cell_expansion_revalidates_mutated_typed_studies() -> None:
    study = BenchmarkStudy.model_validate(study_value())
    study.models[0].configuration.settings["api_key"] = "SECRET"

    with pytest.raises(ValidationError, match="credentials or endpoints"):
        expand_benchmark_study(study)


async def test_preflight_rejects_before_pack_or_docker_work(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def load_pack(_reference: object) -> LoadedEvaluationPack:
        calls.append("pack")
        raise AssertionError("pack loading must not start")

    async def verify_image(_image: str) -> None:
        calls.append("docker")
        raise AssertionError("Docker verification must not start")

    runtime = BenchmarkRuntime.model_validate(runtime_value(tmp_path))
    environment = {
        "MLFLOW_TRACKING_URI": "databricks",
        "MLFLOW_EXPERIMENT_ID": "experiment",
        "DATABRICKS_HOST": "https://example.cloud.databricks.com",
        "DATABRICKS_TOKEN": "secret",
    }

    with pytest.raises(ValueError, match="agent revision"):
        await prepare_benchmark(
            BenchmarkStudy.model_validate(study_value()),
            runtime,
            current_revision="e" * 40,
            environment=environment,
            pack_loader=load_pack,
            image_verifier=verify_image,
        )

    assert calls == []
    assert not runtime.workspace_root.exists()


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"MLFLOW_TRACKING_URI": "file:/tmp/mlruns"},
        {"MLFLOW_TRACKING_URI": "databricks", "MLFLOW_EXPERIMENT_ID": "x"},
        {
            "MLFLOW_TRACKING_URI": "http://tracking.example",
            "MLFLOW_EXPERIMENT_ID": "x",
        },
    ],
)
async def test_preflight_requires_supported_mlflow_before_external_work(
    tmp_path: Path,
    environment: dict[str, str],
) -> None:
    async def verify_image(_image: str) -> None:
        raise AssertionError("Docker verification must not start")

    with pytest.raises(ValueError, match="MLflow"):
        await prepare_benchmark(
            BenchmarkStudy.model_validate(study_value()),
            BenchmarkRuntime.model_validate(runtime_value(tmp_path)),
            current_revision=REVISION,
            environment=environment,
            pack_loader=lambda _reference: (_ for _ in ()).throw(
                AssertionError("pack loading must not start")
            ),
            image_verifier=verify_image,
        )


async def test_preflight_accepts_local_mlflow_and_plain_dataset_name(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def load_pack(_reference: object) -> LoadedEvaluationPack:
        calls.append("pack")
        raise RuntimeError("pack loading reached")

    runtime = BenchmarkRuntime.model_validate(
        runtime_value(
            tmp_path,
            datasets=(
                {"pack_id": "pack-a", "dataset_name": "pack_a"},
                {"pack_id": "pack-b", "dataset_name": "pack_b"},
            ),
        )
    )

    with pytest.raises(RuntimeError, match="pack loading reached"):
        await prepare_benchmark(
            BenchmarkStudy.model_validate(study_value()),
            runtime,
            current_revision=REVISION,
            environment={
                "MLFLOW_TRACKING_URI": "http://127.0.0.1:5000",
                "MLFLOW_EXPERIMENT_ID": "1",
            },
            pack_loader=load_pack,
            image_verifier=_unexpected_image_verifier,
        )

    assert calls == ["pack"]


async def test_preflight_requires_unity_catalog_name_for_databricks(
    tmp_path: Path,
) -> None:
    runtime = BenchmarkRuntime.model_validate(
        runtime_value(
            tmp_path,
            datasets=(
                {"pack_id": "pack-a", "dataset_name": "pack_a"},
                {"pack_id": "pack-b", "dataset_name": "pack_b"},
            ),
        )
    )

    with pytest.raises(ValueError, match="dataset name"):
        await prepare_benchmark(
            BenchmarkStudy.model_validate(study_value()),
            runtime,
            current_revision=REVISION,
            environment={
                "MLFLOW_TRACKING_URI": "databricks",
                "MLFLOW_EXPERIMENT_ID": "1",
                "DATABRICKS_HOST": "https://example.cloud.databricks.com",
                "DATABRICKS_TOKEN": "secret",
            },
            pack_loader=lambda _reference: (_ for _ in ()).throw(
                AssertionError("pack loading must not start")
            ),
            image_verifier=_unexpected_image_verifier,
        )


async def test_preflight_requires_exact_runtime_coverage_before_external_work(
    tmp_path: Path,
) -> None:
    environment = {
        "MLFLOW_TRACKING_URI": "databricks",
        "MLFLOW_EXPERIMENT_ID": "experiment",
        "DATABRICKS_HOST": "host",
        "DATABRICKS_TOKEN": "token",
    }
    runtime = BenchmarkRuntime.model_validate(
        runtime_value(
            tmp_path,
            datasets=({"pack_id": "pack-a", "dataset_name": "catalog.schema.pack_a"},),
        )
    )

    with pytest.raises(ValueError, match="exactly cover"):
        await prepare_benchmark(
            BenchmarkStudy.model_validate(study_value()),
            runtime,
            current_revision=REVISION,
            environment=environment,
            pack_loader=lambda _reference: (_ for _ in ()).throw(
                AssertionError("pack loading must not start")
            ),
            image_verifier=_unexpected_image_verifier,
        )


async def _unexpected_image_verifier(_image: str) -> None:
    raise AssertionError("Docker verification must not start")

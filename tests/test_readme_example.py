"""Preparation tests do not invoke Docker or model providers."""

import hashlib
import runpy
import shutil
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from apps.private_data_chat.settings import load_configuration
from dsa import RunRequest


@pytest.mark.parametrize("image", ["sha256:" + "a" * 64, "repo/image@sha256:" + "b" * 64])
def test_preparation_preserves_immutable_image(
    image: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = Path(__file__).resolve().parents[1] / "examples/smard"
    for filename in ("question.txt", "answer-schema.json", "chat.toml.example"):
        shutil.copyfile(example / filename, tmp_path / filename)
    # Only file identity is checked by prepare.py; no analysis is executed here.
    database = tmp_path / "fixture.duckdb"
    database.write_bytes(b"test database identity")
    namespace = runpy.run_path(str(example / "prepare.py"))
    main = namespace["main"]
    monkeypatch.setitem(main.__globals__, "HERE", tmp_path)
    monkeypatch.setitem(main.__globals__, "DATABASE", database)
    monkeypatch.setitem(
        main.__globals__, "DATABASE_SHA256", hashlib.sha256(database.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(sys, "argv", ["prepare.py", "--docker-image", image])
    main()
    config_path = tmp_path / "local/chat.toml"
    config = load_configuration(config_path=config_path)
    assert config.dsa.docker_image == image
    request = RunRequest.model_validate_json((tmp_path / "local/task.json").read_text())
    assert request.database_path == database
    assert request.derivation is not None and request.derivation.allow_plots
    with pytest.raises(ValidationError):
        type(config.dsa).model_validate({**config.dsa.model_dump(), "docker_image": "image:latest"})

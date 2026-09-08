"""Prepare the README's standalone request and chat configuration; no model calls."""

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

import tomli_w

from apps.private_data_chat.settings import load_configuration
from dsa import DerivationRequest, ModelConfiguration, RunPolicy, RunRequest
from dsa.cli import canonical_json
from dsa.docker import default_docker_configuration

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATABASE = ROOT / "data/smard-de-lu-2024/1.2.0/database/smard_de_lu_2024.duckdb"
DATABASE_SHA256 = "249143dd8b399a6be84d666190f16b89ebc71545783362a50efaa40dc02bc45b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-image", required=True)
    args = parser.parse_args()
    default_docker_configuration(args.docker_image)
    if DATABASE.is_symlink() or not DATABASE.is_file():
        parser.error("download the SMARD database using the command in README.md first")
    with DATABASE.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != DATABASE_SHA256:
            parser.error("the SMARD database does not match the README's pinned snapshot")

    output = HERE / "local"
    output.mkdir(exist_ok=True)
    config = tomllib.loads((HERE / "chat.toml.example").read_text())
    config.update(
        database_path=str(DATABASE),
        runs_directory=str(output / "chat-runs"),
        docker_image=(
            f"dsa-python@{args.docker_image}"
            if args.docker_image.startswith("sha256:")
            else args.docker_image
        ),
    )
    chat_path = output / "chat.toml"
    chat_path.write_text(tomli_w.dumps(config))
    load_configuration(config_path=chat_path)

    request = RunRequest(
        database_path=DATABASE,
        question=(HERE / "question.txt").read_text().strip(),
        answer_schema=json.loads((HERE / "answer-schema.json").read_text()),
        derivation=DerivationRequest(allow_plots=True),
        model=ModelConfiguration(
            name=config["trusted"]["model"], settings=config["trusted"]["settings"]
        ),
        policy=RunPolicy(),
    )
    request_path = output / "task.json"
    request_path.write_text(canonical_json(request.model_dump(mode="json")) + "\n")
    print(f"Standalone request: {request_path}")
    print(f"Chat configuration: {chat_path}")


if __name__ == "__main__":
    main()

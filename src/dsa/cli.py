"""Shared safe primitives for DSA command-line boundaries."""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path
from typing import Any, Never

_MAX_CONTRACT_BYTES = 1024 * 1024


class HelpRequested(Exception):
    """Internal control flow for argparse help without process exit."""


class Parser(argparse.ArgumentParser):
    """Argument parser that leaves exit-code and safe output policy to its caller."""

    def error(self, message: str) -> Never:
        raise ValueError(message)

    def exit(self, status: int = 0, message: str | None = None) -> Never:
        if status == 0:
            raise HelpRequested
        raise ValueError(message or "parser exit")


def read_contract(path: Path) -> bytes:
    """Read one bounded regular file without following its final symlink."""
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > _MAX_CONTRACT_BYTES
        ):
            raise ValueError("contract size invalid")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = None
            content = source.read(_MAX_CONTRACT_BYTES + 1)
    except ValueError:
        raise
    except OSError:
        raise ValueError("contract unavailable") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not content or len(content) > _MAX_CONTRACT_BYTES:
        raise ValueError("contract size invalid")
    return content


def emit(value: dict[str, Any]) -> None:
    """Write one canonical JSON object to standard output."""
    print(canonical_json(value))


def canonical_json(value: object) -> str:
    """Serialize finite JSON deterministically."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )

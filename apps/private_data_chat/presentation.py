"""Safe presentation helpers for text crossing an untrusted boundary."""

from __future__ import annotations

import string

_MARKDOWN_ESCAPES = str.maketrans(
    {character: f"\\{character}" for character in string.punctuation}
)


def escape_markdown_text(value: str) -> str:
    """Render arbitrary text literally inside a Markdown document."""
    return value.translate(_MARKDOWN_ESCAPES)


def escape_markdown_inline(value: str) -> str:
    """Render arbitrary text literally without letting it break its container line."""
    return escape_markdown_text(" ".join(value.split()))

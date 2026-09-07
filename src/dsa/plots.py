"""Bounded static plot validation and notebook extraction; no model execution."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass

from PIL import Image

from dsa.contract import DerivationPlot

MAX_PLOT_BYTES = 5 * 1024 * 1024
MAX_NOTEBOOK_BYTES = 10 * 1024 * 1024
MAX_PLOT_DIMENSION = 2048


@dataclass(frozen=True)
class VerifiedPlot:
    declaration: DerivationPlot
    content: bytes


def validate_png(content: bytes) -> None:
    """Check the format and dimensions before decoding bounded pixels fully."""
    if not content or len(content) > MAX_PLOT_BYTES:
        raise ValueError("plot exceeds byte limit")
    with Image.open(io.BytesIO(content)) as image:
        if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
            raise ValueError("plots must be static PNG")
        if not all(0 < size <= MAX_PLOT_DIMENSION for size in image.size):
            raise ValueError("plot exceeds dimension limit")
        image.verify()
    with Image.open(io.BytesIO(content)) as image:
        image.load()


def notebook_plots(content: bytes) -> tuple[VerifiedPlot, ...]:
    """Extract bounded PNG downloads from an already identity-verified notebook."""
    if len(content) > MAX_NOTEBOOK_BYTES:
        raise ValueError("notebook exceeds byte limit")
    notebook = json.loads(content)
    plots: list[VerifiedPlot] = []
    total = 0
    for cell in notebook["cells"]:
        declaration = cell.get("metadata", {}).get("dsa_plot")
        if declaration is None:
            continue
        if len(plots) >= 3:
            raise ValueError("too many plots")
        plot = DerivationPlot.model_validate(declaration)
        data = base64.b64decode(cell["outputs"][0]["data"]["image/png"], validate=True)
        total += len(data)
        if total > MAX_PLOT_BYTES or any(p.declaration.filename == plot.filename for p in plots):
            raise ValueError("plot batch exceeds limits")
        validate_png(data)
        plots.append(VerifiedPlot(plot, data))
    return tuple(plots)

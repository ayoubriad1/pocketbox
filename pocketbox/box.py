"""Docking-box geometry: turn a set of pocket points into an AutoDock Vina box.

The center and size of the box are fully determined by the pocket coordinates;
this module is deterministic and depends only on numpy.

NOTE ON CONVENTIONS (not hard rules): a focused docking box is commonly sized
to the pocket extent plus a ~4-5 A buffer, ending up roughly 18-25 A per side.
Boxes that are too large degrade both accuracy and speed. Tune per target and
ALWAYS validate placement by re-docking a known ligand when one exists.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np


@dataclass
class Box:
    center: tuple[float, float, float]
    size: tuple[float, float, float]

    def as_dict(self) -> dict:
        cx, cy, cz = self.center
        sx, sy, sz = self.size
        return dict(center_x=cx, center_y=cy, center_z=cz,
                    size_x=sx, size_y=sy, size_z=sz)


def compute_box(points: Iterable[Sequence[float]],
                buffer: float = 5.0,
                cubic: bool = False,
                min_size: float = 12.0,
                max_size: float = 30.0,
                round_to: int = 3) -> Box:
    """Compute a docking box enclosing `points`.

    Parameters
    ----------
    points   : iterable of (x, y, z) -- pocket alpha-sphere vertices are ideal;
               pocket-lining atom coordinates work as a fallback.
    buffer   : padding (A) added on every side beyond the point extent.
    cubic    : if True, force a cube using the largest dimension.
    min_size : floor for each side (A); avoids absurdly tiny boxes.
    max_size : ceiling for each side (A); guards against runaway blind boxes.
    """
    pts = np.asarray(list(points), dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise ValueError("`points` must be a non-empty Nx3 array of coordinates")

    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = (lo + hi) / 2.0
    size = (hi - lo) + 2.0 * buffer
    size = np.clip(size, min_size, max_size)

    if cubic:
        side = float(np.max(size))
        size = np.array([side, side, side])

    return Box(
        center=tuple(round(float(v), round_to) for v in center),
        size=tuple(round(float(v), round_to) for v in size),
    )


def write_vina_config(box: Box,
                      receptor: str = "receptor.pdbqt",
                      ligand: str = "ligand.pdbqt",
                      out: str = "docked.pdbqt",
                      exhaustiveness: int = 16,
                      num_modes: int = 9,
                      seed: int | None = None) -> str:
    """Return an AutoDock Vina config file as a string."""
    cx, cy, cz = box.center
    sx, sy, sz = box.size
    lines = [
        f"receptor = {receptor}",
        f"ligand = {ligand}",
        f"out = {out}",
        "",
        f"center_x = {cx}",
        f"center_y = {cy}",
        f"center_z = {cz}",
        "",
        f"size_x = {sx}",
        f"size_y = {sy}",
        f"size_z = {sz}",
        "",
        f"exhaustiveness = {exhaustiveness}",
        f"num_modes = {num_modes}",
    ]
    if seed is not None:
        lines.append(f"seed = {seed}")
    return "\n".join(lines) + "\n"

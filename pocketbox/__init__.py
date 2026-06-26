"""
pocketbox - ligand-aware binding-pocket triage + docking-box export.

Pipeline:
    PDB id / file -> detect pockets (fpocket) -> describe each pocket
    -> rank pockets against a ligand profile -> compute AutoDock Vina box
    -> visualize (py3Dmol) -> export Vina config.

Heavy/optional dependencies (rdkit, py3Dmol, stmol) are imported lazily inside
the functions that need them, so the geometry/ranking core works with only
numpy installed.
"""

__version__ = "0.1.0"

from .box import compute_box, write_vina_config           # noqa: E402,F401
from .match import score_pocket, rank_pockets             # noqa: E402,F401
from .descriptors import describe_pocket                   # noqa: E402,F401

__all__ = [
    "compute_box",
    "write_vina_config",
    "score_pocket",
    "rank_pockets",
    "describe_pocket",
]

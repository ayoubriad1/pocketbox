"""Physicochemical description of a pocket from its lining residues and atoms.

These descriptors feed the ligand-compatibility ranking. They are intentionally
simple and transparent (no black box) so a method section can describe exactly
what was computed.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np

# Standard residue property tables (amino acids).
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "CYS"}
AROMATIC = {"PHE", "TRP", "TYR", "HIS"}
POLAR = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS"}
POS_CHARGED = {"LYS", "ARG"}           # His treated as weakly/contextually +
NEG_CHARGED = {"ASP", "GLU"}
# Residues that can form covalent bonds with reactive warheads.
REACTIVE = {"CYS", "SER", "LYS", "THR", "TYR", "HIS"}
# Common catalytic / structural metal ions seen as HETATM.
METAL_IONS = {"ZN", "MG", "MN", "FE", "CA", "CU", "NI", "CO", "NA", "K"}


@dataclass
class PocketDescriptors:
    n_residues: int
    hydrophobic_fraction: float
    aromatic_count: int
    polar_count: int
    net_charge: int                 # (#pos) - (#neg) among lining residues
    has_metal: bool
    metals: list[str]
    reactive_residues: list[str]    # residues usable by covalent warheads
    volume: float | None            # A^3 if known (e.g. from fpocket)
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def describe_pocket(residue_names: Sequence[str],
                    het_names: Sequence[str] = (),
                    volume: float | None = None) -> PocketDescriptors:
    """Build descriptors from a list of pocket-lining residue names.

    `residue_names` should be 3-letter codes (repeats allowed, one per residue
    instance lining the pocket). `het_names` are HETATM residue names found in
    or adjacent to the pocket (used to flag metals / cofactors).
    """
    res = [r.strip().upper() for r in residue_names if r and r.strip()]
    n = len(res) if res else 0

    hyd = sum(1 for r in res if r in HYDROPHOBIC)
    aro = sum(1 for r in res if r in AROMATIC)
    pol = sum(1 for r in res if r in POLAR)
    pos = sum(1 for r in res if r in POS_CHARGED)
    neg = sum(1 for r in res if r in NEG_CHARGED)
    reactive = sorted({r for r in res if r in REACTIVE})

    metals = sorted({h.strip().upper() for h in het_names
                     if h.strip().upper() in METAL_IONS})

    return PocketDescriptors(
        n_residues=n,
        hydrophobic_fraction=(hyd / n) if n else 0.0,
        aromatic_count=aro,
        polar_count=pol,
        net_charge=pos - neg,
        has_metal=bool(metals),
        metals=metals,
        reactive_residues=reactive,
        volume=volume,
    )


def centroid(points) -> tuple[float, float, float]:
    pts = np.asarray(list(points), dtype=float)
    c = pts.mean(axis=0)
    return (float(c[0]), float(c[1]), float(c[2]))

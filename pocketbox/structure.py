"""Tiny dependency-free PDB atom reader.

Used to resolve P2Rank references (residue ids -> residue names, surface atom
serials -> coordinates) against the original structure. Deliberately minimal;
for heavy structural work use Biopython instead.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Atom:
    serial: int
    name: str
    resname: str
    chain: str
    resseq: int
    x: float
    y: float
    z: float
    is_het: bool


def parse_pdb_atoms(path: str) -> list[Atom]:
    atoms: list[Atom] = []
    with open(path) as fh:
        for line in fh:
            tag = line[:6].strip()
            if tag not in ("ATOM", "HETATM"):
                continue
            try:
                serial = int(line[6:11])
            except ValueError:
                serial = -1
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            try:
                resseq = int(line[22:26])
            except ValueError:
                resseq = -1
            atoms.append(Atom(
                serial=serial,
                name=line[12:16].strip(),
                resname=line[17:20].strip().upper(),
                chain=line[21].strip(),
                resseq=resseq,
                x=x, y=y, z=z,
                is_het=(tag == "HETATM"),
            ))
    return atoms


def coord_by_serial(atoms: list[Atom]) -> dict[int, tuple[float, float, float]]:
    return {a.serial: (a.x, a.y, a.z) for a in atoms}


def resname_by_key(atoms: list[Atom]) -> dict[tuple[str, int], str]:
    """Map (chain, resseq) -> residue name."""
    out: dict[tuple[str, int], str] = {}
    for a in atoms:
        out[(a.chain, a.resseq)] = a.resname
    return out

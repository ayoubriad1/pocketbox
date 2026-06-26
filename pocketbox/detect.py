"""Run fpocket and parse its output into structured Pocket objects.

fpocket must be installed and on PATH (bioconda: `conda install -c bioconda
fpocket`, or build from source). Running `fpocket -f prot.pdb` produces a
`prot_out/` directory containing:

    prot_info.txt                      per-pocket properties (incl. druggability)
    prot_out/pockets/pocketN_atm.pdb   atoms lining pocket N (+ REMARK props)
    prot_out/pockets/pocketN_vert.pqr  alpha-sphere vertices defining the cavity

We center the docking box on the alpha-sphere vertices (best estimate of the
cavity), falling back to lining-atom coordinates if vertices are unavailable.

The exact output file naming has varied slightly across fpocket versions, so
the parser is defensive. If parsing yields nothing, check your fpocket version's
output layout.
"""

from __future__ import annotations
import glob
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field

import numpy as np

from .descriptors import describe_pocket, PocketDescriptors


@dataclass
class Pocket:
    id: int
    druggability: float | None
    score: float | None
    volume: float | None
    n_alpha_spheres: int | None
    vertices: np.ndarray                  # Nx3 alpha-sphere coordinates
    lining_atoms: np.ndarray              # Mx3 atom coordinates
    residue_names: list[str]              # 3-letter codes of lining residues
    het_names: list[str] = field(default_factory=list)
    descriptors: PocketDescriptors | None = None

    def points_for_box(self) -> np.ndarray:
        if self.vertices is not None and len(self.vertices):
            return self.vertices
        return self.lining_atoms


def fpocket_available() -> bool:
    return shutil.which("fpocket") is not None


def run_fpocket(pdb_path: str, extra_args: list[str] | None = None) -> str:
    """Run fpocket on a PDB file; return the path to its *_out directory."""
    if not fpocket_available():
        raise RuntimeError(
            "fpocket not found on PATH. Install via "
            "`conda install -c bioconda fpocket` or build from source.")
    pdb_path = os.path.abspath(pdb_path)
    cmd = ["fpocket", "-f", pdb_path] + (extra_args or [])
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    base = os.path.splitext(pdb_path)[0]
    out_dir = base + "_out"
    if not os.path.isdir(out_dir):
        # some versions place it next to cwd; search.
        cand = glob.glob(os.path.join(os.path.dirname(pdb_path), "*_out"))
        if cand:
            out_dir = cand[0]
        else:
            raise RuntimeError("fpocket finished but no *_out directory found")
    return out_dir


_FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def _parse_info(info_path: str) -> dict[int, dict]:
    """Parse prot_info.txt -> {pocket_id: {prop: value}}."""
    props: dict[int, dict] = {}
    if not os.path.isfile(info_path):
        return props
    current = None
    with open(info_path) as fh:
        for line in fh:
            m = re.match(r"\s*Pocket\s+(\d+)\s*:", line)
            if m:
                current = int(m.group(1))
                props[current] = {}
                continue
            if current is None:
                continue
            m = re.match(r"\s*([A-Za-z][\w .%/()-]*?)\s*:\s*(" + _FLOAT + r")", line)
            if m:
                key = m.group(1).strip().lower()
                props[current][key] = float(m.group(2))
    return props


def _read_coords_pdb_like(path: str, records=("ATOM", "HETATM")):
    """Read x,y,z (cols 31-54) and resnames from a PDB/PQR-ish file."""
    coords = []
    resnames = []
    hetnames = []
    if not os.path.isfile(path):
        return np.empty((0, 3)), resnames, hetnames
    with open(path) as fh:
        for line in fh:
            tag = line[:6].strip()
            if tag not in records:
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                parts = line.split()
                try:
                    x, y, z = float(parts[-5]), float(parts[-4]), float(parts[-3])
                except Exception:
                    continue
            coords.append((x, y, z))
            res = line[17:20].strip()
            if tag == "HETATM":
                hetnames.append(res)
            else:
                resnames.append(res)
    return (np.array(coords) if coords else np.empty((0, 3)),
            resnames, hetnames)


def parse_fpocket(out_dir: str) -> list[Pocket]:
    """Parse an fpocket *_out directory into a list of Pocket objects."""
    # locate info file
    info_files = glob.glob(os.path.join(out_dir, "*_info.txt"))
    info = _parse_info(info_files[0]) if info_files else {}

    pockets_dir = os.path.join(out_dir, "pockets")
    atm_files = sorted(glob.glob(os.path.join(pockets_dir, "pocket*_atm.pdb")),
                       key=_pocket_num)
    pockets: list[Pocket] = []
    for atm in atm_files:
        pid = _pocket_num(atm)
        vert = atm.replace("_atm.pdb", "_vert.pqr")

        lining, resnames, het_from_atm = _read_coords_pdb_like(atm)
        verts, _, _ = _read_coords_pdb_like(vert)

        info_p = info.get(pid, {})
        drug = info_p.get("druggability score")
        score = info_p.get("score")
        volume = info_p.get("volume")
        n_alpha = info_p.get("number of alpha spheres")

        # unique residue list (one entry per residue instance is fine for ratios)
        desc = describe_pocket(resnames, het_from_atm, volume=volume)

        pockets.append(Pocket(
            id=pid,
            druggability=drug,
            score=score,
            volume=volume,
            n_alpha_spheres=int(n_alpha) if n_alpha is not None else None,
            vertices=verts,
            lining_atoms=lining,
            residue_names=resnames,
            het_names=het_from_atm,
            descriptors=desc,
        ))
    return pockets


def _pocket_num(path: str) -> int:
    m = re.search(r"pocket(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else 0


def detect_pockets(pdb_path: str) -> list[Pocket]:
    """Convenience: run fpocket then parse. Returns pockets sorted by id."""
    out_dir = run_fpocket(pdb_path)
    return parse_fpocket(out_dir)


# --- backend-agnostic entry point ----------------------------------------- #
def backend_available(backend: str = "fpocket") -> bool:
    backend = backend.lower()
    if backend == "fpocket":
        return fpocket_available()
    if backend == "p2rank":
        from .detect_p2rank import p2rank_available
        return p2rank_available()
    raise ValueError(f"Unknown backend {backend!r} (use 'fpocket' or 'p2rank')")


def detect(pdb_path: str, backend: str = "fpocket") -> list[Pocket]:
    """Detect pockets with the chosen backend; both return `Pocket` objects."""
    backend = backend.lower()
    if backend == "fpocket":
        return detect_pockets(pdb_path)
    if backend == "p2rank":
        from .detect_p2rank import detect_pockets_p2rank
        return detect_pockets_p2rank(pdb_path)
    raise ValueError(f"Unknown backend {backend!r} (use 'fpocket' or 'p2rank')")

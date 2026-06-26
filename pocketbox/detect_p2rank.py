"""P2Rank backend.

P2Rank is a machine-learning pocket predictor (Java; CLI command `prank`).
Running `prank predict -f protein.pdb -o OUTDIR` writes
`OUTDIR/protein.pdb_predictions.csv` whose columns include rank, score, a
calibrated `probability`, `center_x/y/z`, a space-separated `residue_ids`
list (chain_resnum), and a space-separated `surf_atom_ids` list (atom serials).

We map `probability` into the druggability slot (it is a 0-1 ligandability
estimate), resolve residue names and surface-atom coordinates from the original
PDB, and build the same `Pocket` objects the fpocket backend produces, so the
rest of the pipeline is backend-agnostic.

The predictions CSV format has been stable but can vary slightly across P2Rank
versions; the parser is header-driven and defensive. If a pocket list comes
back empty, check your version's CSV header.
"""

from __future__ import annotations
import csv
import glob
import os
import shutil
import subprocess

import numpy as np

from .descriptors import describe_pocket
from .detect import Pocket            # reuse the shared dataclass
from .structure import parse_pdb_atoms, coord_by_serial, resname_by_key


def p2rank_available() -> bool:
    return shutil.which("prank") is not None


def run_p2rank(pdb_path: str, out_dir: str | None = None) -> str:
    if not p2rank_available():
        raise RuntimeError(
            "P2Rank not found on PATH (command `prank`). Install via "
            "`conda install -c bioconda p2rank` (needs Java), or download the "
            "standalone distribution and add it to PATH.")
    pdb_path = os.path.abspath(pdb_path)
    if out_dir is None:
        out_dir = os.path.splitext(pdb_path)[0] + "_p2rank"
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(["prank", "predict", "-f", pdb_path, "-o", out_dir],
                   check=True, capture_output=True, text=True)
    return out_dir


def _find_predictions_csv(out_dir: str) -> str | None:
    hits = glob.glob(os.path.join(out_dir, "*_predictions.csv"))
    if not hits:
        hits = glob.glob(os.path.join(out_dir, "**", "*_predictions.csv"),
                         recursive=True)
    return hits[0] if hits else None


def _norm(key: str) -> str:
    return key.strip().lower().replace(" ", "")


def parse_p2rank(out_dir: str, pdb_path: str) -> list[Pocket]:
    csv_path = _find_predictions_csv(out_dir)
    if not csv_path:
        return []

    atoms = parse_pdb_atoms(pdb_path)
    coord_lut = coord_by_serial(atoms)
    resname_lut = resname_by_key(atoms)
    het_resnames = {a.resname for a in atoms if a.is_het}

    pockets: list[Pocket] = []
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        rows = [r for r in reader if r and any(c.strip() for c in r)]
    if not rows:
        return []

    header = [_norm(c) for c in rows[0]]
    idx = {name: header.index(name) for name in header}

    def col(row, name, default=""):
        i = idx.get(name)
        return row[i].strip() if (i is not None and i < len(row)) else default

    for row in rows[1:]:
        try:
            rank = int(float(col(row, "rank", "0")))
        except ValueError:
            rank = len(pockets) + 1
        prob = _to_float(col(row, "probability"))
        score = _to_float(col(row, "score"))

        # surface atom serials -> coordinates
        surf_ids = [int(s) for s in col(row, "surf_atom_ids").split() if s.isdigit()]
        pts = [coord_lut[s] for s in surf_ids if s in coord_lut]
        verts = np.array(pts) if pts else np.empty((0, 3))

        # residue ids "A_123" -> residue names via the structure
        res_names = []
        for rid in col(row, "residue_ids").split():
            if "_" in rid:
                ch, _, num = rid.partition("_")
                try:
                    key = (ch.strip(), int(num))
                except ValueError:
                    continue
                if key in resname_lut:
                    res_names.append(resname_lut[key])

        # if a center is given but no surf atoms resolved, seed a point cloud
        if verts.shape[0] == 0:
            cx = _to_float(col(row, "center_x"))
            cy = _to_float(col(row, "center_y"))
            cz = _to_float(col(row, "center_z"))
            if None not in (cx, cy, cz):
                verts = np.array([[cx, cy, cz]])

        desc = describe_pocket(res_names, het_names=list(het_resnames),
                               volume=None)
        pockets.append(Pocket(
            id=rank,
            druggability=prob,         # calibrated 0-1 ligandability
            score=score,
            volume=None,               # P2Rank does not report volume by default
            n_alpha_spheres=None,
            vertices=verts,
            lining_atoms=verts,
            residue_names=res_names,
            het_names=list(het_resnames),
            descriptors=desc,
        ))
    return pockets


def _to_float(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def detect_pockets_p2rank(pdb_path: str) -> list[Pocket]:
    out_dir = run_p2rank(pdb_path)
    return parse_p2rank(out_dir, pdb_path)

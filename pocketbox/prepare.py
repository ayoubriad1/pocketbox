"""Best-effort receptor/ligand preparation to PDBQT.

Preparation tooling varies a lot between setups, so each helper tries a couple
of common routes and raises a clear, actionable error if none are present.
Order tried: AutoDockTools / ADFR (`prepare_receptor`, `prepare_ligand`) first,
then Open Babel (`obabel`) as a portable fallback.

This path depends on the external prep tools above and is not covered by the
unit tests, so treat it as a starting point and check the output PDBQT on your
own machine.
"""

from __future__ import annotations
import os
import shutil
import subprocess


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def receptor_to_pdbqt(pdb_path: str, out_pdbqt: str) -> str:
    """Convert a receptor PDB to a rigid-receptor PDBQT."""
    pdb_path = os.path.abspath(pdb_path)
    out_pdbqt = os.path.abspath(out_pdbqt)
    if _have("prepare_receptor"):
        _run(["prepare_receptor", "-r", pdb_path, "-o", out_pdbqt])
        return out_pdbqt
    if _have("obabel"):
        # -xr writes a rigid receptor PDBQT; -p 7.4 sets protonation at pH 7.4.
        _run(["obabel", pdb_path, "-O", out_pdbqt, "-xr", "-p", "7.4"])
        return out_pdbqt
    raise RuntimeError(
        "No receptor-prep tool found. Install ADFR/AutoDockTools "
        "(`prepare_receptor`) or Open Babel (`obabel`).")


def ligand_to_pdbqt(in_path: str, out_pdbqt: str) -> str:
    """Convert a ligand (PDB/SDF/MOL2) to PDBQT with Gasteiger charges."""
    in_path = os.path.abspath(in_path)
    out_pdbqt = os.path.abspath(out_pdbqt)
    if _have("prepare_ligand"):
        # prepare_ligand often insists on a local path; run in the file's dir.
        cwd = os.path.dirname(in_path) or "."
        _run(["prepare_ligand", "-l", os.path.basename(in_path),
              "-o", out_pdbqt])
        return out_pdbqt
    if _have("obabel"):
        _run(["obabel", in_path, "-O", out_pdbqt, "-h",
              "--partialcharge", "gasteiger"])
        return out_pdbqt
    raise RuntimeError(
        "No ligand-prep tool found. Install ADFR/AutoDockTools "
        "(`prepare_ligand`) or Open Babel (`obabel`).")

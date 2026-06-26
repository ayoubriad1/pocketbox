"""Run Vina-family docking and validate a pose by RMSD.

Supports three interchangeable engines that share Vina's flag interface:
  * AutoDock Vina   (binary: vina)
  * smina           (binary: smina)
  * QuickVina2      (binary: qvina2, sometimes qvina02)

The redocking workflow re-docks a target's *native* (co-crystallised) ligand
inside a computed box and measures how far the top pose lands from the crystal
pose. A symmetry-corrected, NON-superimposed RMSD (RDKit `CalcRMS`) of ≲ 2 A is
the usual sanity threshold; superimposing RMSD would hide a misplaced pose, so
it is deliberately not used.

IMPORTANT: the subprocess docking calls and the PDBQT preparation are NOT run in
the environment where this file was generated. The pieces that ARE unit-tested
are: native-ligand extraction, score parsing, and the RMSD computation. Validate
the full loop on your own machine before trusting it.
"""

from __future__ import annotations
import os
import re
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

from .box import Box, write_vina_config   # noqa: F401  (config export reused)


# --- engine registry ------------------------------------------------------- #
@dataclass
class Engine:
    key: str
    default_binary: str
    note: str


ENGINES: dict[str, Engine] = {
    "vina": Engine("vina", "vina", "AutoDock Vina (1.2.x flag interface)."),
    "smina": Engine("smina", "smina", "smina; reads pdbqt, Vina-style flags."),
    "qvina2": Engine("qvina2", "qvina2",
                     "QuickVina2; binary may be 'qvina2' or 'qvina02'."),
}


def engine_binary(engine: str, override: str | None = None) -> str:
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine {engine!r}. Use one of {list(ENGINES)}.")
    return override or ENGINES[engine].default_binary


def engine_available(engine: str, override: str | None = None) -> bool:
    return shutil.which(engine_binary(engine, override)) is not None


# --- native ligand extraction (unit-tested) -------------------------------- #
def extract_ligand_pdb(pdb_path: str, resname: str,
                       out_path: str, chain: str | None = None) -> str:
    """Write a PDB containing only HETATM records for `resname` (optionally a
    specific chain). Returns the output path. Used to pull the crystal ligand
    out for redocking + RMSD reference.
    """
    resname = resname.strip().upper()
    kept = []
    with open(pdb_path) as fh:
        for line in fh:
            if line[:6].strip() != "HETATM":
                continue
            if line[17:20].strip().upper() != resname:
                continue
            if chain is not None and line[21].strip() != chain.strip():
                continue
            kept.append(line)
    if not kept:
        raise ValueError(
            f"No HETATM records for residue {resname!r}"
            + (f" in chain {chain}" if chain else "")
            + f" found in {pdb_path}.")
    with open(out_path, "w") as out:
        out.writelines(kept)
        out.write("END\n")
    return out_path


# --- score parsing (unit-tested) ------------------------------------------- #
_SCORE_PATTERNS = [
    re.compile(r"REMARK\s+VINA\s+RESULT:\s*(-?\d+\.\d+)"),  # vina / qvina
    re.compile(r"minimizedAffinity[>:\s]+(-?\d+\.\d+)"),    # smina pdbqt / sdf
    re.compile(r"REMARK\s+SMINA\s+RESULT:\s*(-?\d+\.\d+)"),
]


def parse_docked_scores(out_path: str) -> list[float]:
    """Return docking affinities (kcal/mol) in pose order from a docked file
    (pdbqt or sdf). Robust to Vina / smina / QuickVina output conventions.
    """
    scores: list[float] = []
    with open(out_path) as fh:
        for line in fh:
            for pat in _SCORE_PATTERNS:
                m = pat.search(line)
                if m:
                    scores.append(float(m.group(1)))
                    break
    return scores


# --- pose RMSD (computation unit-tested via CalcRMS) ----------------------- #
def pose_rmsd(ref_mol, probe_mol) -> float:
    """Symmetry-corrected, in-place heavy-atom RMSD between two RDKit mols.

    Uses rdMolAlign.CalcRMS, which does NOT superimpose (so a pose that landed
    in the wrong place is correctly penalised) and accounts for molecular
    symmetry. Both mols must share the same heavy-atom topology.
    """
    try:
        from rdkit.Chem import rdMolAlign
    except ImportError as e:                                    # pragma: no cover
        raise ImportError("RDKit required for pose_rmsd.") from e
    return float(rdMolAlign.CalcRMS(probe_mol, ref_mol))


def naive_rmsd(coords_a, coords_b) -> float:
    """Fallback heavy-atom RMSD assuming consistent atom ordering (NO symmetry
    handling). Use only when RDKit atom matching is unavailable; flag it as
    approximate."""
    a = np.asarray(coords_a, float)
    b = np.asarray(coords_b, float)
    if a.shape != b.shape:
        raise ValueError("coordinate arrays must have identical shape")
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def load_mol_with_template(pose_path: str, template_smiles: str | None):
    """Load a docked/reference pose for RMSD. If a SMILES template is given,
    bond orders are assigned from it so atom matching is reliable (recommended).
    Requires RDKit; PDBQT must first be converted to PDB/SDF (e.g. with obabel).
    """
    from rdkit import Chem
    mol = Chem.MolFromPDBFile(pose_path, sanitize=False, removeHs=True)
    if mol is None:
        raise ValueError(f"RDKit could not read {pose_path} (convert pdbqt->pdb first).")
    if template_smiles:
        tmpl = Chem.MolFromSmiles(template_smiles)
        mol = Chem.AssignBondOrdersFromTemplate(tmpl, mol)
    return mol


# --- run one engine -------------------------------------------------------- #
def run_engine(engine: str,
               receptor_pdbqt: str,
               ligand_pdbqt: str,
               box: Box,
               out_path: str,
               exhaustiveness: int = 16,
               num_modes: int = 9,
               seed: int | None = None,
               binary: str | None = None) -> tuple[str, list[float]]:
    """Run a single docking and return (out_path, affinities)."""
    bin_ = engine_binary(engine, binary)
    if shutil.which(bin_) is None:
        raise RuntimeError(f"{bin_} not found on PATH for engine '{engine}'.")
    cx, cy, cz = box.center
    sx, sy, sz = box.size
    cmd = [
        bin_,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--out", out_path,
        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
        "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_path, parse_docked_scores(out_path)


# --- high-level redocking validation (composes tested pieces) -------------- #
@dataclass
class RedockResult:
    engine: str
    top_affinity: float | None
    rmsd: float | None
    passed: bool | None
    all_affinities: list[float]
    out_path: str
    note: str = ""


def redock_validate(engine: str,
                    receptor_pdbqt: str,
                    native_ligand_pdbqt: str,
                    reference_pose_pdb: str,
                    box: Box,
                    workdir: str,
                    template_smiles: str | None = None,
                    rmsd_threshold: float = 2.0,
                    exhaustiveness: int = 16,
                    seed: int | None = 42,
                    binary: str | None = None) -> RedockResult:
    """Re-dock a native ligand and compare the top pose to the crystal pose.

    `native_ligand_pdbqt` : the crystal ligand prepared as PDBQT (the thing to
                            dock back).
    `reference_pose_pdb`  : the crystal ligand as PDB, used as the RMSD reference
                            (convert the top docked pose to PDB to compare).
    `template_smiles`     : optional, strongly recommended for reliable matching.
    """
    os.makedirs(workdir, exist_ok=True)
    out_path = os.path.join(workdir, f"redock_{engine}.pdbqt")
    _, scores = run_engine(engine, receptor_pdbqt, native_ligand_pdbqt, box,
                           out_path, exhaustiveness=exhaustiveness,
                           seed=seed, binary=binary)
    top = scores[0] if scores else None

    rmsd = None
    note = ""
    try:
        # convert the top pose to PDB for RDKit; requires obabel at runtime.
        top_pdb = os.path.join(workdir, f"redock_{engine}_top.pdb")
        if shutil.which("obabel"):
            subprocess.run(["obabel", out_path, "-O", top_pdb, "-f", "1", "-l", "1"],
                           check=True, capture_output=True, text=True)
            ref = load_mol_with_template(reference_pose_pdb, template_smiles)
            probe = load_mol_with_template(top_pdb, template_smiles)
            rmsd = pose_rmsd(ref, probe)
        else:
            note = "obabel not found; RMSD skipped (scores still reported)."
    except Exception as e:                                      # pragma: no cover
        note = f"RMSD step failed: {e}. Scores still reported."

    passed = None if rmsd is None else (rmsd <= rmsd_threshold)
    return RedockResult(engine=engine, top_affinity=top, rmsd=rmsd,
                        passed=passed, all_affinities=scores,
                        out_path=out_path, note=note)

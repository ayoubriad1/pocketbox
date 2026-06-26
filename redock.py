#!/usr/bin/env python3
"""Redocking validation CLI.

Re-docks a target's native (co-crystallised) ligand inside a box centred on that
ligand and reports, per engine, the top affinity and the symmetry-corrected
RMSD to the crystal pose (pass if ≲ threshold). This is the validation step that
turns box placement into something you've actually checked.

Example
-------
  python redock.py --pdb-id 2V5Z --ligand-resname A3T \
      --engine all --template-smiles "O=C(...)..."

Requires (at runtime, on your machine): a prep tool (ADFR `prepare_receptor`/
`prepare_ligand` or Open Babel), the chosen docking engine(s), and Open Babel
for the RMSD conversion. None of these run in the scaffold-generation sandbox.
"""

from __future__ import annotations
import argparse
import os
import sys
import tempfile

from pocketbox.box import compute_box
from pocketbox.structure import parse_pdb_atoms
from pocketbox.dock import (ENGINES, engine_available, extract_ligand_pdb,
                            redock_validate)
from pocketbox.prepare import receptor_to_pdbqt, ligand_to_pdbqt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Redocking validation (RMSD to crystal pose).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdb-id", help="4-char RCSB PDB id (downloaded).")
    g.add_argument("--pdb-file", help="Local receptor+ligand .pdb file.")
    ap.add_argument("--ligand-resname", required=True,
                    help="HETATM residue name of the native ligand (e.g. AHT).")
    ap.add_argument("--ligand-chain", default=None,
                    help="Restrict native ligand to this chain (optional).")
    ap.add_argument("--engine", choices=list(ENGINES) + ["all"], default="all",
                    help="Docking engine(s) to validate with.")
    ap.add_argument("--template-smiles", default=None,
                    help="Ligand SMILES; strongly recommended for reliable RMSD.")
    ap.add_argument("--buffer", type=float, default=5.0,
                    help="Box buffer around the native ligand (Å).")
    ap.add_argument("--exhaustiveness", type=int, default=16)
    ap.add_argument("--rmsd-threshold", type=float, default=2.0)
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args(argv)

    workdir = args.workdir or tempfile.mkdtemp(prefix="pocketbox_redock_")

    # 1) receptor structure
    if args.pdb_id:
        from pocketbox.fetch import fetch_pdb
        receptor_pdb = fetch_pdb(args.pdb_id, workdir)
    else:
        receptor_pdb = args.pdb_file
        if not os.path.isfile(receptor_pdb):
            print(f"ERROR: no such file {receptor_pdb}", file=sys.stderr)
            return 2

    # 2) extract native ligand (reference pose + docking input)
    native_pdb = os.path.join(workdir, "native_ligand.pdb")
    try:
        extract_ligand_pdb(receptor_pdb, args.ligand_resname, native_pdb,
                           chain=args.ligand_chain)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # 3) box centred on the native ligand
    lig_atoms = parse_pdb_atoms(native_pdb)
    coords = [(a.x, a.y, a.z) for a in lig_atoms]
    box = compute_box(coords, buffer=args.buffer)
    print(f"Box around native ligand: center={box.center} size={box.size}")

    # 4) prepare receptor + ligand to PDBQT (best-effort)
    try:
        receptor_pdbqt = receptor_to_pdbqt(receptor_pdb,
                                           os.path.join(workdir, "receptor.pdbqt"))
        native_pdbqt = ligand_to_pdbqt(native_pdb,
                                       os.path.join(workdir, "native_ligand.pdbqt"))
    except RuntimeError as e:
        print(f"ERROR during prep: {e}", file=sys.stderr)
        return 3

    engines = list(ENGINES) if args.engine == "all" else [args.engine]
    print(f"\nValidating with: {', '.join(engines)}\n")
    any_run = False
    for eng in engines:
        if not engine_available(eng):
            print(f"[skip] {eng}: binary not on PATH")
            continue
        any_run = True
        try:
            res = redock_validate(
                eng, receptor_pdbqt, native_pdbqt, native_pdb, box, workdir,
                template_smiles=args.template_smiles,
                rmsd_threshold=args.rmsd_threshold,
                exhaustiveness=args.exhaustiveness)
        except Exception as e:
            print(f"[error] {eng}: {e}")
            continue
        verdict = ("PASS" if res.passed else "FAIL") if res.passed is not None else "RMSD n/a"
        rmsd_s = f"{res.rmsd:.2f} A" if res.rmsd is not None else "—"
        aff_s = f"{res.top_affinity:.2f}" if res.top_affinity is not None else "—"
        print(f"  {eng:7s}  top affinity {aff_s} kcal/mol   "
              f"RMSD {rmsd_s}   [{verdict}]")
        if res.note:
            print(f"           note: {res.note}")

    if not any_run:
        print("No engines were available on PATH. Install at least one.",
              file=sys.stderr)
        return 4
    print(f"\nArtifacts in: {workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

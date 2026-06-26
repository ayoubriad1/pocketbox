#!/usr/bin/env python3
"""PocketBox CLI — scriptable pocket triage + Vina box export.

Examples
--------
  python cli.py --pdb-id 2V5Z --ligand-class maob_inhibitor
  python cli.py --pdb-file receptor.pdb --smiles "CC(=O)Oc1ccccc1C(=O)O" --top 5
"""

from __future__ import annotations
import argparse
import os
import sys
import tempfile

from pocketbox.detect import backend_available, detect
from pocketbox.match import rank_pockets
from pocketbox.box import compute_box, write_vina_config
from pocketbox.ligand import (get_profile, list_classes,
                              compute_ligand_descriptors,
                              profile_from_descriptors)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ligand-aware pocket triage + Vina box.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdb-id", help="4-char RCSB PDB id (downloaded).")
    g.add_argument("--pdb-file", help="Path to a local .pdb file.")
    lg = ap.add_mutually_exclusive_group(required=True)
    lg.add_argument("--ligand-class", choices=list_classes(),
                    help="Ligand class from the taxonomy.")
    lg.add_argument("--smiles", help="Ligand SMILES (descriptors via RDKit).")
    ap.add_argument("--backend", choices=["fpocket", "p2rank"],
                    default="fpocket", help="Pocket detector to use.")
    ap.add_argument("--buffer", type=float, default=5.0, help="Box buffer (Å).")
    ap.add_argument("--cubic", action="store_true", help="Force cubic box.")
    ap.add_argument("--top", type=int, default=3, help="How many pockets to show.")
    ap.add_argument("--outdir", default=".", help="Where to write vina configs.")
    args = ap.parse_args(argv)

    if not backend_available(args.backend):
        tool = "fpocket" if args.backend == "fpocket" else "P2Rank (`prank`)"
        print(f"ERROR: {tool} not on PATH. Install via bioconda.",
              file=sys.stderr)
        return 2

    # structure
    tmp = tempfile.mkdtemp(prefix="pocketbox_")
    if args.pdb_id:
        from pocketbox.fetch import fetch_pdb
        pdb_path = fetch_pdb(args.pdb_id, tmp)
    else:
        pdb_path = args.pdb_file
        if not os.path.isfile(pdb_path):
            print(f"ERROR: no such file {pdb_path}", file=sys.stderr)
            return 2

    # ligand profile
    if args.ligand_class:
        profile = get_profile(args.ligand_class)
    else:
        desc = compute_ligand_descriptors(args.smiles)
        profile = profile_from_descriptors(desc)
        print("Ligand descriptors:", desc)

    pockets = detect(pdb_path, backend=args.backend)
    if not pockets:
        print("No pockets found.", file=sys.stderr)
        return 1

    ranked = rank_pockets(
        [{"id": p.id, "descriptors": p.descriptors,
          "druggability": p.druggability, "_pocket": p} for p in pockets],
        profile,
    )

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\nTop {min(args.top, len(ranked))} pockets for "
          f"'{profile.name}':\n")
    for rank, r in enumerate(ranked[:args.top], 1):
        p = r["_pocket"]
        sb = r["score"]
        box = compute_box(p.points_for_box(), buffer=args.buffer, cubic=args.cubic)
        cfg = write_vina_config(box)
        cfg_path = os.path.join(args.outdir, f"vina_pocket{p.id}.txt")
        with open(cfg_path, "w") as f:
            f.write(cfg)
        print(f"#{rank}  pocket {p.id}  match={sb.total}  "
              f"druggability={p.druggability}  vol={p.volume}")
        print(f"      center={box.center}  size={box.size}")
        print(f"      -> {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

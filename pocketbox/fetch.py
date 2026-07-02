"""Fetch a structure by PDB id from RCSB.

Uses the standard RCSB download endpoint. Verify the URL/terms if RCSB changes
its service. Requires network access at runtime.
"""

from __future__ import annotations
import os
import urllib.request

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"


def fetch_pdb(pdb_id: str, out_dir: str = ".", prefer_cif: bool = False) -> str:
    """Download a structure and return the local file path."""
    pdb_id = pdb_id.strip().upper()
    if len(pdb_id) != 4:
        raise ValueError(f"PDB id should be 4 characters, got {pdb_id!r}")
    os.makedirs(out_dir, exist_ok=True)

    url = (RCSB_CIF_URL if prefer_cif else RCSB_PDB_URL).format(pdb_id=pdb_id)
    ext = "cif" if prefer_cif else "pdb"
    path = os.path.join(out_dir, f"{pdb_id}.{ext}")

    req = urllib.request.Request(url, headers={"User-Agent": "pocketbox/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    with open(path, "wb") as f:
        f.write(data)
    return path

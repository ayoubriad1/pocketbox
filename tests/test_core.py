"""Lightweight self-tests for the geometry + ranking core (numpy only).

Run: python -m tests.test_core   (or: python tests/test_core.py)
These do NOT require fpocket, RDKit, or network access.
"""
import numpy as np

from pocketbox.box import compute_box, write_vina_config
from pocketbox.descriptors import describe_pocket
from pocketbox.match import rank_pockets
from pocketbox.ligand import get_profile


def test_box_center_and_size():
    pts = [(0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 10), (10, 10, 10)]
    box = compute_box(pts, buffer=5.0, min_size=0, max_size=100)
    assert box.center == (5.0, 5.0, 5.0), box.center
    # extent 10 + 2*5 buffer = 20 per side
    assert box.size == (20.0, 20.0, 20.0), box.size


def test_box_cubic_and_clip():
    pts = [(0, 0, 0), (2, 40, 4)]   # very anisotropic
    box = compute_box(pts, buffer=0, cubic=True, min_size=0, max_size=100)
    assert box.size[0] == box.size[1] == box.size[2]


def test_vina_config_contains_fields():
    box = compute_box([(0, 0, 0), (10, 10, 10)], buffer=2)
    cfg = write_vina_config(box, receptor="r.pdbqt", ligand="l.pdbqt")
    for token in ("center_x", "size_z", "exhaustiveness", "r.pdbqt"):
        assert token in cfg


def test_ranking_prefers_metal_pocket_for_metal_ligand():
    # ligand needs a metal; pocket A has Zn, pocket B does not.
    a = describe_pocket(["HIS", "HIS", "CYS", "ASP"], het_names=["ZN"], volume=600)
    b = describe_pocket(["ALA", "LEU", "VAL", "PHE"], het_names=[], volume=600)
    profile = get_profile("metal_coordinating")
    ranked = rank_pockets(
        [{"id": "A", "descriptors": a, "druggability": 0.5},
         {"id": "B", "descriptors": b, "druggability": 0.7}],
        profile)
    assert ranked[0]["id"] == "A", "metal pocket should win despite lower druggability"


def test_ranking_charge_complementarity():
    # anionic ligand wants a cationic (Lys/Arg-rich) pocket.
    cationic = describe_pocket(["LYS", "ARG", "ARG", "SER"], volume=600)
    anionic = describe_pocket(["ASP", "GLU", "GLU", "SER"], volume=600)
    profile = get_profile("anionic")
    ranked = rank_pockets(
        [{"id": "cationic", "descriptors": cationic, "druggability": 0.5},
         {"id": "anionic_pocket", "descriptors": anionic, "druggability": 0.5}],
        profile)
    assert ranked[0]["id"] == "cationic"


def test_p2rank_parser(tmp_path=None):
    """Parse a synthetic P2Rank predictions CSV against a tiny PDB."""
    import os, tempfile, numpy as np
    from pocketbox.detect_p2rank import parse_p2rank

    d = tempfile.mkdtemp()
    # minimal PDB: 3 atoms, chain A, residues 10(HIS),11(ASP); + a ZN hetatm
    pdb = os.path.join(d, "prot.pdb")
    with open(pdb, "w") as f:
        f.write(
            "ATOM      1  CA  HIS A  10      11.000  12.000  13.000  1.00  0.00           C\n"
            "ATOM      2  CA  ASP A  11      14.000  12.000  13.000  1.00  0.00           C\n"
            "HETATM    3 ZN    ZN A 200      12.500  12.000  13.000  1.00  0.00          ZN\n")
    csv_path = os.path.join(d, "prot.pdb_predictions.csv")
    with open(csv_path, "w") as f:
        f.write("name,rank,score,probability,sas_points,surf_atoms,"
                "center_x,center_y,center_z,residue_ids,surf_atom_ids\n")
        f.write("pocket1,1,12.5,0.83,40,2,12.5,12.0,13.0,A_10 A_11,1 2\n")

    pockets = parse_p2rank(d, pdb)
    assert len(pockets) == 1
    p = pockets[0]
    assert p.id == 1
    assert abs(p.druggability - 0.83) < 1e-6
    assert p.vertices.shape == (2, 3)             # two surf atoms resolved
    assert set(p.residue_names) == {"HIS", "ASP"}
    assert "ZN" in p.descriptors.metals


def test_extract_ligand_pdb():
    import os, tempfile
    from pocketbox.dock import extract_ligand_pdb
    d = tempfile.mkdtemp()
    src = os.path.join(d, "complex.pdb")
    with open(src, "w") as f:
        f.write(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "HETATM    2  C1  LIG A 300       5.000   5.000   5.000  1.00  0.00           C\n"
            "HETATM    3  O1  LIG A 300       6.000   5.000   5.000  1.00  0.00           O\n"
            "HETATM    4  O   HOH A 400      10.000  10.000  10.000  1.00  0.00           O\n")
    out = os.path.join(d, "lig.pdb")
    extract_ligand_pdb(src, "LIG", out)
    lines = [l for l in open(out) if l.startswith("HETATM")]
    assert len(lines) == 2, lines
    assert all(l[17:20].strip() == "LIG" for l in lines)


def test_parse_docked_scores():
    import os, tempfile
    from pocketbox.dock import parse_docked_scores
    d = tempfile.mkdtemp()
    out = os.path.join(d, "docked.pdbqt")
    with open(out, "w") as f:
        f.write("MODEL 1\nREMARK VINA RESULT:    -7.50      0.000      0.000\nENDMDL\n")
        f.write("MODEL 2\nREMARK VINA RESULT:    -6.10      1.200      2.300\nENDMDL\n")
        f.write("MODEL 3\nREMARK minimizedAffinity -5.05\nENDMDL\n")
    scores = parse_docked_scores(out)
    assert scores == [-7.50, -6.10, -5.05], scores


def test_pose_rmsd_inplace_translation():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from pocketbox.dock import pose_rmsd
    m = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    AllChem.EmbedMolecule(m, randomSeed=7)
    ref = Chem.Mol(m)
    probe = Chem.Mol(m)
    conf = probe.GetConformer()
    for i in range(probe.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (pos.x + 3.0, pos.y + 4.0, pos.z))
    assert abs(pose_rmsd(ref, probe) - 5.0) < 1e-3



if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")

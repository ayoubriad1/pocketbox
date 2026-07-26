"""Ligand side: RDKit descriptors + a curated ligand-class taxonomy.

A LigandProfile expresses, for a class of ligand, what kind of pocket it is
*compatible* with. This is a chemistry-grounded filter (hydrophobic pockets
favour lipophilic ligands; cationic pockets favour anionic ligands; metal sites
favour coordinating groups; covalent warheads need a reactive residue). It is a
shortlisting heuristic, NOT a predictor of the true binding site -- confirm by
docking / re-docking.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class LigandProfile:
    name: str
    # Preferred pocket hydrophobic fraction in [0,1]; the ranker rewards pockets
    # whose hydrophobic_fraction is near this value.
    target_hydrophobicity: float
    # Preferred pocket net charge SIGN: -1 wants a positively charged pocket
    # (because the ligand is anionic), +1 wants a negatively charged pocket,
    # 0 = no strong preference.
    charge_pref: int
    # Preferred pocket size direction: "small", "medium", "large", or "any".
    size_pref: str
    needs_metal: bool = False
    needs_reactive_residue: bool = False
    aromatic_bonus: bool = False
    notes: str = ""


# --- Curated taxonomy ------------------------------------------------------
# Generic medicinal-chemistry classes plus a few PD-target-relevant ones.
# Edit / extend freely for your own targets.
LIGAND_CLASSES: dict[str, LigandProfile] = {
    "small_molecule_inhibitor": LigandProfile(
        "Small-molecule inhibitor", 0.45, 0, "medium",
        aromatic_bonus=True,
        notes="Generic drug-like inhibitor; balanced pocket."),
    "fragment": LigandProfile(
        "Fragment (MW<300)", 0.45, 0, "small",
        notes="Small, low-complexity; binds sub-pockets."),
    "lipophilic": LigandProfile(
        "Lipophilic / lipid-like", 0.75, 0, "medium",
        notes="High logP; prefers hydrophobic, buried pockets."),
    "anionic": LigandProfile(
        "Anionic / acidic ligand", 0.35, -1, "medium",
        notes="Net negative; favours cationic (Lys/Arg-rich) pockets."),
    "cationic": LigandProfile(
        "Cationic / basic ligand", 0.35, +1, "medium",
        notes="Net positive; favours anionic (Asp/Glu-rich) pockets."),
    "metal_coordinating": LigandProfile(
        "Metal-coordinating", 0.40, 0, "medium", needs_metal=True,
        notes="Chelators / metalloenzyme inhibitors; needs a metal in pocket."),
    "covalent": LigandProfile(
        "Covalent warhead", 0.45, 0, "medium", needs_reactive_residue=True,
        notes="Needs a reactive residue (Cys/Ser/Lys/Thr/Tyr/His)."),
    "nucleotide": LigandProfile(
        "Nucleotide / phosphate", 0.30, -1, "large", needs_metal=False,
        notes="Phosphate groups favour cationic pockets, often near Mg."),
    "peptide": LigandProfile(
        "Peptide / peptidomimetic", 0.40, 0, "large",
        notes="Larger, mixed surface; extended pockets / grooves."),
    "macrocycle_protac": LigandProfile(
        "Macrocycle / PROTAC", 0.50, 0, "large",
        notes="Large footprint; shallow extended surfaces."),
    # PD-relevant examples (tune to your 11 targets):
    "maob_inhibitor": LigandProfile(
        "MAO-B inhibitor", 0.60, 0, "medium", aromatic_bonus=True,
        notes="Often aromatic, lipophilic; near FAD cofactor."),
    "kinase_inhibitor": LigandProfile(
        "Kinase inhibitor (e.g. LRRK2)", 0.50, 0, "medium",
        aromatic_bonus=True,
        notes="ATP-competitive; hinge-binding aromatic core."),
    "a2a_antagonist": LigandProfile(
        "Adenosine A2A antagonist", 0.55, 0, "medium", aromatic_bonus=True,
        notes="GPCR orthosteric pocket; aromatic heterocycles."),
}


def list_classes() -> list[str]:
    return list(LIGAND_CLASSES.keys())


def get_profile(class_key: str) -> LigandProfile:
    if class_key not in LIGAND_CLASSES:
        raise KeyError(
            f"Unknown ligand class '{class_key}'. "
            f"Known: {', '.join(LIGAND_CLASSES)}")
    return LIGAND_CLASSES[class_key]


# --- RDKit-derived descriptors + auto profile ------------------------------
def compute_ligand_descriptors(smiles: str) -> dict:
    """Compute descriptors for a ligand from SMILES (requires rdkit)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    except ImportError as e:                                   # pragma: no cover
        raise ImportError(
            "RDKit is required for ligand descriptors. "
            "Install with `pip install rdkit` or via conda-forge.") from e

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles!r}")

    metals = {"Zn", "Mg", "Mn", "Fe", "Ca", "Cu", "Ni", "Co",
              "Pt", "Ru", "Au", "Na", "K"}
    has_metal = any(a.GetSymbol() in metals for a in mol.GetAtoms())
    formal_charge = Chem.GetFormalCharge(mol)

    return {
        "smiles": smiles,
        "mol_weight": round(Descriptors.MolWt(mol), 2),
        "logp": round(Crippen.MolLogP(mol), 2),
        "h_bond_donors": rdMolDescriptors.CalcNumHBD(mol),
        "h_bond_acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "formal_charge": formal_charge,
        "has_metal": has_metal,
    }


def _mol_descriptors(mol, smiles: str | None = None) -> dict:
    """Shared descriptor core for an RDKit mol (from SMILES or a structure file)."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

    metals = {"Zn", "Mg", "Mn", "Fe", "Ca", "Cu", "Ni", "Co",
              "Pt", "Ru", "Au", "Na", "K"}
    has_metal = any(a.GetSymbol() in metals for a in mol.GetAtoms())
    return {
        "smiles": smiles or Chem.MolToSmiles(Chem.RemoveHs(mol)),
        "mol_weight": round(Descriptors.MolWt(mol), 2),
        "logp": round(Crippen.MolLogP(mol), 2),
        "h_bond_donors": rdMolDescriptors.CalcNumHBD(mol),
        "h_bond_acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "formal_charge": Chem.GetFormalCharge(mol),
        "has_metal": has_metal,
    }


def smiles_to_molblock(smiles: str) -> str:
    """Embed a SMILES into a 3D conformer and return an MDL mol block (for 3D view)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as e:                                    # pragma: no cover
        raise ImportError("RDKit is required to build a 3D ligand.") from e

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xf00d
    if AllChem.EmbedMolecule(mol, params) != 0:                 # embedding failed
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:                                          # pragma: no cover
        pass
    return Chem.MolToMolBlock(mol)


def structure_to_molblock_and_desc(text: str, fmt: str) -> tuple[str, dict]:
    """Parse an uploaded ligand structure (mol/sdf/pdb) -> (3D mol block, descriptors).

    If the file has no 3D coordinates, a conformer is generated so it can still
    be shown in 3D and described.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as e:                                    # pragma: no cover
        raise ImportError("RDKit is required to read a ligand structure.") from e

    fmt = fmt.lower().lstrip(".")
    if fmt in ("mol", "sdf"):
        mol = Chem.MolFromMolBlock(text, removeHs=False)
    elif fmt == "pdb":
        mol = Chem.MolFromPDBBlock(text, removeHs=False)
    else:
        raise ValueError(f"Unsupported ligand format '.{fmt}' (use mol/sdf/pdb).")
    if mol is None:
        raise ValueError("Could not parse the ligand structure file.")

    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:                                     # pragma: no cover
            pass
    return Chem.MolToMolBlock(mol), _mol_descriptors(mol)


def profile_from_descriptors(desc: dict) -> LigandProfile:
    """Heuristically derive a LigandProfile from RDKit descriptors, so a user
    can paste a SMILES instead of picking a class."""
    logp = desc.get("logp", 1.0)
    mw = desc.get("mol_weight", 350.0)
    charge = desc.get("formal_charge", 0)
    aromatic = desc.get("aromatic_rings", 0)

    # Map logP -> preferred pocket hydrophobicity (clamped).
    target_hyd = max(0.2, min(0.85, 0.40 + 0.07 * logp))

    if charge < 0:
        charge_pref = -1
    elif charge > 0:
        charge_pref = +1
    else:
        charge_pref = 0

    if mw < 300:
        size_pref = "small"
    elif mw > 550:
        size_pref = "large"
    else:
        size_pref = "medium"

    return LigandProfile(
        name=f"Custom (MW {mw}, logP {logp}, charge {charge})",
        target_hydrophobicity=round(target_hyd, 2),
        charge_pref=charge_pref,
        size_pref=size_pref,
        needs_metal=bool(desc.get("has_metal")),
        aromatic_bonus=aromatic >= 1,
        notes="Profile derived automatically from SMILES descriptors.",
    )

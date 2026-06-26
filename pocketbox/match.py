"""Rank pockets for a given ligand profile.

The score is a transparent weighted blend of:
  * fpocket druggability (how 'bindable' the pocket looks, ligand-agnostic)
  * compatibility between the pocket's physicochemistry and the ligand profile

Every term's contribution is returned so the ranking is fully explainable.
This is a SHORTLISTING heuristic. The honest workflow is: use it to pick a few
candidate pockets, then confirm by docking (and re-dock a native ligand to
validate box placement when a co-crystallised ligand exists).
"""

from __future__ import annotations
from dataclasses import dataclass

from .descriptors import PocketDescriptors
from .ligand import LigandProfile


# Weights (sum need not be 1; final score is normalised to 0-1).
W_DRUGGABILITY = 0.40
W_HYDROPHOBIC = 0.20
W_CHARGE = 0.15
W_SIZE = 0.10
W_METAL = 0.08
W_REACTIVE = 0.04
W_AROMATIC = 0.03

_SIZE_BANDS = {  # rough volume (A^3) midpoints per qualitative band
    "small": 250.0, "medium": 600.0, "large": 1100.0, "any": None,
}


@dataclass
class ScoreBreakdown:
    total: float
    terms: dict[str, float]
    reasons: list[str]


def _size_term(volume: float | None, size_pref: str) -> tuple[float, str]:
    if size_pref == "any" or volume is None:
        return 0.5, "size not constrained / unknown"
    target = _SIZE_BANDS.get(size_pref)
    if target is None:
        return 0.5, "size not constrained"
    # Gaussian-ish closeness on a log scale.
    import math
    ratio = math.log((volume + 1e-6) / target)
    term = math.exp(-(ratio ** 2) / (2 * 0.5 ** 2))   # ~1 when volume~target
    verdict = "matches" if term > 0.6 else "off-target"
    return term, f"pocket volume {volume:.0f} A^3 {verdict} '{size_pref}'"


def score_pocket(pocket: PocketDescriptors,
                 druggability: float | None,
                 profile: LigandProfile) -> ScoreBreakdown:
    terms: dict[str, float] = {}
    reasons: list[str] = []

    # 1) druggability (0-1 already, from fpocket). Neutral 0.5 if unknown.
    drug = 0.5 if druggability is None else max(0.0, min(1.0, druggability))
    terms["druggability"] = drug
    if druggability is not None:
        reasons.append(f"druggability score {drug:.2f}")

    # 2) hydrophobic match (closeness of fraction to target).
    hyd_close = 1.0 - abs(pocket.hydrophobic_fraction - profile.target_hydrophobicity)
    hyd_close = max(0.0, hyd_close)
    terms["hydrophobic"] = hyd_close
    reasons.append(
        f"hydrophobic fraction {pocket.hydrophobic_fraction:.2f} vs target "
        f"{profile.target_hydrophobicity:.2f}")

    # 3) charge complementarity.
    if profile.charge_pref == 0:
        charge_term = 0.5
        reasons.append("no charge preference")
    else:
        # charge_pref encodes the LIGAND's charge sign (anionic -> -1).
        # Electrostatic complementarity rewards a pocket of the OPPOSITE sign.
        want = profile.charge_pref
        net = pocket.net_charge
        if want * net < 0:                         # opposite signs => complement
            charge_term = min(1.0, 0.5 + 0.1 * abs(net))
            reasons.append(f"pocket net charge {net:+d} complements ligand")
        elif net == 0:
            charge_term = 0.45
            reasons.append("pocket roughly neutral")
        else:
            charge_term = max(0.0, 0.5 - 0.1 * abs(net))
            reasons.append(f"pocket net charge {net:+d} does not complement ligand")
    terms["charge"] = charge_term

    # 4) size.
    size_term, size_reason = _size_term(pocket.volume, profile.size_pref)
    terms["size"] = size_term
    reasons.append(size_reason)

    # 5) metal requirement (hard-ish): big penalty if needed and absent.
    if profile.needs_metal:
        metal_term = 1.0 if pocket.has_metal else 0.0
        reasons.append("metal present" if pocket.has_metal
                       else "NO metal in pocket (ligand needs one)")
    else:
        metal_term = 0.5
    terms["metal"] = metal_term

    # 6) reactive residue requirement (covalent).
    if profile.needs_reactive_residue:
        reactive_term = 1.0 if pocket.reactive_residues else 0.0
        reasons.append(
            f"reactive residues {pocket.reactive_residues}" if pocket.reactive_residues
            else "NO reactive residue (covalent warhead needs one)")
    else:
        reactive_term = 0.5
    terms["reactive"] = reactive_term

    # 7) aromatic bonus.
    if profile.aromatic_bonus:
        aro_term = min(1.0, pocket.aromatic_count / 3.0)
        reasons.append(f"{pocket.aromatic_count} aromatic residues")
    else:
        aro_term = 0.5
    terms["aromatic"] = aro_term

    weighted = (
        W_DRUGGABILITY * terms["druggability"] +
        W_HYDROPHOBIC * terms["hydrophobic"] +
        W_CHARGE * terms["charge"] +
        W_SIZE * terms["size"] +
        W_METAL * terms["metal"] +
        W_REACTIVE * terms["reactive"] +
        W_AROMATIC * terms["aromatic"]
    )
    total_w = (W_DRUGGABILITY + W_HYDROPHOBIC + W_CHARGE + W_SIZE +
               W_METAL + W_REACTIVE + W_AROMATIC)
    total = weighted / total_w

    return ScoreBreakdown(total=round(total, 4),
                          terms={k: round(v, 3) for k, v in terms.items()},
                          reasons=reasons)


def rank_pockets(pockets: list[dict], profile: LigandProfile) -> list[dict]:
    """Rank a list of pocket dicts. Each pocket dict must contain at least:
        'descriptors' : PocketDescriptors
        'druggability' : float | None
        'id' : pocket id/label
    Returns the same dicts augmented with 'score' (ScoreBreakdown), sorted
    high to low.
    """
    out = []
    for p in pockets:
        sb = score_pocket(p["descriptors"], p.get("druggability"), profile)
        q = dict(p)
        q["score"] = sb
        out.append(q)
    out.sort(key=lambda d: d["score"].total, reverse=True)
    return out

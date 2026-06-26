"""PocketBox - Streamlit app.

Run locally:   streamlit run app.py
"""

from __future__ import annotations
import os
import tempfile

import streamlit as st

from pocketbox.box import compute_box, write_vina_config
from pocketbox.match import rank_pockets
from pocketbox.ligand import (LIGAND_CLASSES, get_profile,
                              compute_ligand_descriptors,
                              profile_from_descriptors)

st.set_page_config(page_title="PocketBox", page_icon="🧬", layout="wide")

# --------------------------------------------------------------------------- #
# Sidebar: honest framing
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("About")
    st.markdown(
        "**PocketBox** detects pockets on a protein, ranks them for "
        "*compatibility* with your ligand class, and exports a ready-to-run "
        "AutoDock Vina docking box.")
    st.warning(
        "The ranking is a **shortlisting heuristic**, not a predictor of the "
        "true binding site. Confirm candidates by docking, and re-dock a known "
        "ligand to validate box placement when a co-crystallised one exists.")
    st.markdown("---")
    st.caption(
        "Detection uses **fpocket** (must be installed & on PATH). "
        "Ligand descriptors use **RDKit**. Viewer uses **py3Dmol/stmol**.")

st.title("🧬 PocketBox")
st.caption("Ligand-aware pocket triage + docking-box export")

# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
c1, c2 = st.columns(2)

with c1:
    st.subheader("1 · Structure")
    src = st.radio("Source", ["PDB ID", "Upload .pdb"], horizontal=True)
    pdb_id = ""
    uploaded = None
    if src == "PDB ID":
        pdb_id = st.text_input("PDB ID (4 chars)", value="", max_chars=4,
                               placeholder="e.g. 2V5Z").strip().upper()
    else:
        uploaded = st.file_uploader("PDB file", type=["pdb"])

with c2:
    st.subheader("2 · Ligand")
    mode = st.radio("Define ligand by", ["Class", "SMILES"], horizontal=True)
    profile = None
    if mode == "Class":
        keys = list(LIGAND_CLASSES.keys())
        labels = [LIGAND_CLASSES[k].name for k in keys]
        idx = st.selectbox("Ligand class", range(len(keys)),
                           format_func=lambda i: labels[i])
        profile = get_profile(keys[idx])
        st.caption(profile.notes)
    else:
        smiles = st.text_input("SMILES", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")
        if smiles:
            try:
                desc = compute_ligand_descriptors(smiles)
                profile = profile_from_descriptors(desc)
                st.json(desc, expanded=False)
            except Exception as e:
                st.error(f"Could not parse ligand: {e}")

st.subheader("3 · Detector & box settings")
b0, b1, b2, b3 = st.columns(4)
backend = b0.radio("Pocket detector", ["fpocket", "p2rank"],
                   format_func=lambda s: "fpocket" if s == "fpocket" else "P2Rank")
buffer = b1.slider("Buffer (Å)", 0.0, 12.0, 5.0, 0.5)
cubic = b2.checkbox("Force cubic box", value=False)
exhaust = b3.slider("Vina exhaustiveness", 1, 32, 16)

run = st.button("Detect pockets & rank", type="primary",
                use_container_width=True)

# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def _load_structure() -> tuple[str, str] | None:
    """Return (pdb_path, pdb_text) or None."""
    tmp = tempfile.mkdtemp(prefix="pocketbox_")
    if src == "PDB ID":
        if len(pdb_id) != 4:
            st.error("Enter a valid 4-character PDB ID.")
            return None
        from pocketbox.fetch import fetch_pdb
        try:
            path = fetch_pdb(pdb_id, tmp)
        except Exception as e:
            st.error(f"Download failed: {e}")
            return None
    else:
        if uploaded is None:
            st.error("Upload a .pdb file.")
            return None
        path = os.path.join(tmp, uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())
    with open(path) as f:
        return path, f.read()


if run:
    if profile is None:
        st.error("Define a ligand first.")
        st.stop()

    from pocketbox.detect import backend_available, detect
    if not backend_available(backend):
        tool = "fpocket" if backend == "fpocket" else "P2Rank (`prank`)"
        st.error(
            f"{tool} is not installed / not on PATH. Install it "
            "(see the README) and restart the app.")
        st.stop()

    loaded = _load_structure()
    if not loaded:
        st.stop()
    pdb_path, pdb_text = loaded

    with st.spinner(f"Running {backend}…"):
        try:
            pockets = detect(pdb_path, backend=backend)
        except Exception as e:
            st.error(f"{backend} failed: {e}")
            st.stop()

    if not pockets:
        st.warning(f"No pockets parsed. Check your {backend} version's output.")
        st.stop()

    ranked = rank_pockets(
        [{"id": p.id, "descriptors": p.descriptors,
          "druggability": p.druggability, "_pocket": p} for p in pockets],
        profile,
    )
    st.session_state["ranked"] = ranked
    st.session_state["pdb_text"] = pdb_text

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
if "ranked" in st.session_state:
    ranked = st.session_state["ranked"]
    pdb_text = st.session_state["pdb_text"]

    st.subheader("Ranked pockets")
    table = []
    for rank, r in enumerate(ranked, 1):
        p = r["_pocket"]
        sb = r["score"]
        table.append({
            "rank": rank,
            "pocket": p.id,
            "match score": sb.total,
            "druggability": p.druggability,
            "volume (Å³)": p.volume,
            "hydrophobic frac": round(p.descriptors.hydrophobic_fraction, 2),
            "net charge": p.descriptors.net_charge,
            "metal": ",".join(p.descriptors.metals) or "-",
        })
    st.dataframe(table, use_container_width=True, hide_index=True)

    labels = [f"#{i+1} · pocket {r['_pocket'].id} "
              f"(score {r['score'].total})" for i, r in enumerate(ranked)]
    sel = st.selectbox("Inspect pocket", range(len(ranked)),
                       format_func=lambda i: labels[i])
    chosen = ranked[sel]
    pocket = chosen["_pocket"]

    with st.expander("Why this pocket scored as it did", expanded=True):
        st.write({k: v for k, v in chosen["score"].terms.items()})
        for reason in chosen["score"].reasons:
            st.markdown(f"- {reason}")

    # box
    box = compute_box(pocket.points_for_box(), buffer=buffer, cubic=cubic)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**3D view** — protein, pocket cavity (orange), box (magenta)")
        try:
            from stmol import showmol
            from pocketbox.visualize import build_view
            view = build_view(pdb_text, box=box,
                              pocket_atom_coords=pocket.points_for_box()[:400])
            showmol(view, height=520, width=720)
        except Exception as e:
            st.info(f"Viewer unavailable ({e}). Box params are still below.")

    with right:
        st.markdown("**Docking box (AutoDock Vina)**")
        st.json(box.as_dict())
        cfg = write_vina_config(box, exhaustiveness=exhaust)
        st.download_button("⬇ Download vina config", cfg,
                           file_name=f"vina_pocket{pocket.id}.txt",
                           use_container_width=True)
        st.code(cfg, language="ini")

    st.markdown("---")
    st.markdown(
        "**Next:** prepare a receptor `.pdbqt` (e.g. with Meeko or "
        "AutoDockTools `prepare_receptor`), then run "
        "`vina --config vina_pocket{}.txt`. If this target has a "
        "co-crystallised ligand, re-dock it first and check RMSD (≲2 Å) to "
        "validate the box.".format(pocket.id))

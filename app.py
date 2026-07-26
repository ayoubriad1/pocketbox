"""PocketBox - Streamlit app.

Run locally:   streamlit run app.py
"""

from __future__ import annotations
import os
import tempfile
import textwrap

import streamlit as st
import streamlit.components.v1 as components

from pocketbox.box import compute_box, write_vina_config
from pocketbox.match import rank_pockets
from pocketbox.detect import backend_available
from pocketbox.ligand import (LIGAND_CLASSES, get_profile,
                              compute_ligand_descriptors,
                              profile_from_descriptors,
                              smiles_to_molblock,
                              structure_to_molblock_and_desc)

st.set_page_config(page_title="PocketBox", layout="wide")

# Which detector backends are actually installed on this machine.
BACKENDS = ("fpocket", "p2rank")
AVAIL = {b: backend_available(b) for b in BACKENDS}


@st.cache_data(show_spinner=False)
def _smiles_3d(smiles: str) -> str:
    return smiles_to_molblock(smiles)


@st.cache_data(show_spinner=False)
def _structure_3d(text: str, fmt: str):
    return structure_to_molblock_and_desc(text, fmt)


def _show3d(view, height: int) -> None:
    """Embed a py3Dmol view directly via Streamlit components (no stmol dep)."""
    components.html(view._make_html(), height=height + 20, scrolling=False)

# --------------------------------------------------------------------------- #
# Look & feel
# --------------------------------------------------------------------------- #
def _html(markup: str) -> None:
    """Render dedented, left-stripped HTML (avoids Markdown's 4-space code rule)."""
    st.markdown(textwrap.dedent(markup).strip(), unsafe_allow_html=True)


_html(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1150px; }
      /* hero */
      .pb-hero {
        display: flex; align-items: center; gap: 18px;
        background: linear-gradient(135deg, #6d5efc 0%, #9a4bff 55%, #b44bff 100%);
        color: #fff; border-radius: 18px; padding: 22px 26px;
        box-shadow: 0 10px 30px rgba(109,94,252,0.28);
      }
      .pb-hero h1 { margin: 0; font-size: 2.0rem; letter-spacing: -0.5px; color:#fff; }
      .pb-hero p  { margin: 4px 0 0; opacity: 0.92; font-size: 0.98rem; }
      .pb-logo { flex: 0 0 auto; }
      /* pipeline strip */
      .pb-flow { display:flex; flex-wrap:wrap; gap:8px; margin: 14px 0 4px; }
      .pb-step {
        background:#f4f3fb; border:1px solid #e7e5f6; color:#4a4766;
        padding:5px 12px; border-radius:999px; font-size:0.82rem; font-weight:600;
      }
      .pb-arrow { color:#b9b6d6; align-self:center; font-weight:700; }
      /* section headers */
      .pb-sec { font-weight:700; color:#2a2740; font-size:1.05rem;
                margin: 2px 0 2px; }
      /* metric cards */
      div[data-testid="stMetric"] {
        background:#faf9ff; border:1px solid #ececf5; border-radius:14px;
        padding:14px 16px;
      }
      div[data-testid="stMetricValue"] { color:#6d5efc; }
      /* primary button as gradient */
      .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6d5efc, #9a4bff);
        border: 0; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem;
      }
      .stButton > button[kind="primary"]:hover { filter: brightness(1.05); }
      /* rank badges */
      .pb-badge { display:inline-block; padding:3px 10px; border-radius:999px;
                  font-weight:700; font-size:0.8rem; color:#fff; }
      .pb-b1 { background:#f0a500; } .pb-b2 { background:#9aa0ad; }
      .pb-b3 { background:#c07b3a; } .pb-bn { background:#c7c4dd; color:#37344a; }
      .pb-foot { color:#8b88a6; font-size:0.85rem; margin-top:8px; }
    </style>
    """
)

_LOGO = (
    '<svg class="pb-logo" width="52" height="52" viewBox="0 0 100 100" '
    'fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M50 6 L88 28 V72 L50 94 L12 72 V28 Z" '
    'fill="rgba(255,255,255,0.14)" stroke="#fff" stroke-width="3"/>'
    '<circle cx="50" cy="50" r="12" fill="#fff"/>'
    '<circle cx="50" cy="50" r="20" fill="none" stroke="#fff" '
    'stroke-width="2" stroke-dasharray="4 5" opacity="0.8"/></svg>'
)

_html(
    f"""
    <div class="pb-hero">
      {_LOGO}
      <div>
        <h1>PocketBox</h1>
        <p>Ligand-aware binding-pocket triage &amp; AutoDock&nbsp;Vina docking-box export</p>
      </div>
    </div>
    <div class="pb-flow">
      <span class="pb-step">Structure</span><span class="pb-arrow">&rsaquo;</span>
      <span class="pb-step">Detect pockets</span><span class="pb-arrow">&rsaquo;</span>
      <span class="pb-step">Describe</span><span class="pb-arrow">&rsaquo;</span>
      <span class="pb-step">Rank vs ligand</span><span class="pb-arrow">&rsaquo;</span>
      <span class="pb-step">Vina box</span><span class="pb-arrow">&rsaquo;</span>
      <span class="pb-step">Export</span>
    </div>
    """
)

# --------------------------------------------------------------------------- #
# Sidebar: honest framing
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "**PocketBox** detects pockets on a protein, ranks them for "
        "*compatibility* with your ligand class, and exports a ready-to-run "
        "AutoDock Vina docking box.")
    st.warning(
        "The ranking is a **shortlisting heuristic**, not a predictor of the "
        "true binding site. Confirm candidates by docking, and re-dock a known "
        "ligand to validate box placement when a co-crystallised one exists.")
    st.divider()
    st.caption(
        "Detection uses **fpocket** or **P2Rank** (must be installed & on "
        "PATH). Ligand descriptors use **RDKit**. 3D views use **py3Dmol / "
        "3Dmol.js**.")

# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
c1, c2 = st.columns(2)

with c1:
    with st.container(border=True):
        st.markdown('<div class="pb-sec">1 · Structure</div>',
                    unsafe_allow_html=True)
        src = st.radio("Source", ["PDB ID", "Upload .pdb"], horizontal=True)
        pdb_id = ""
        uploaded = None
        if src == "PDB ID":
            pdb_id = st.text_input("PDB ID (4 chars)", value="", max_chars=4,
                                   placeholder="e.g. 2V5Z").strip().upper()
        else:
            uploaded = st.file_uploader("PDB file", type=["pdb"])

with c2:
    with st.container(border=True):
        st.markdown('<div class="pb-sec">2 · Ligand</div>',
                    unsafe_allow_html=True)
        mode = st.radio("Define ligand by", ["Class", "SMILES", "Structure file"],
                        horizontal=True)
        profile = None
        lig_text = None          # 3D structure text for the viewer
        lig_fmt = None           # "mol" or "pdb"
        if mode == "Class":
            keys = list(LIGAND_CLASSES.keys())
            labels = [LIGAND_CLASSES[k].name for k in keys]
            idx = st.selectbox("Ligand class", range(len(keys)),
                               format_func=lambda i: labels[i])
            profile = get_profile(keys[idx])
            st.caption(profile.notes)
        elif mode == "SMILES":
            smiles = st.text_input("SMILES",
                                   placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")
            if smiles:
                try:
                    desc = compute_ligand_descriptors(smiles)
                    profile = profile_from_descriptors(desc)
                    lig_text, lig_fmt = _smiles_3d(smiles), "mol"
                    st.json(desc, expanded=False)
                except Exception as e:
                    st.error(f"Could not parse ligand: {e}")
        else:
            up_lig = st.file_uploader("Ligand structure (mol / sdf / pdb)",
                                      type=["mol", "sdf", "pdb"])
            if up_lig is not None:
                try:
                    fmt = up_lig.name.rsplit(".", 1)[-1].lower()
                    text = up_lig.getvalue().decode("utf-8", "replace")
                    molblock, desc = _structure_3d(text, fmt)
                    profile = profile_from_descriptors(desc)
                    lig_text, lig_fmt = molblock, "mol"
                    st.json(desc, expanded=False)
                except Exception as e:
                    st.error(f"Could not read ligand structure: {e}")

# Ligand 3D preview (SMILES-derived or uploaded) — rotatable / zoomable.
if lig_text:
    with st.container(border=True):
        st.markdown('<div class="pb-sec">Ligand in 3D</div>',
                    unsafe_allow_html=True)
        st.caption("Ball-and-stick · drag to rotate · scroll to zoom")
        try:
            from pocketbox.visualize import build_ligand_view
            _show3d(build_ligand_view(lig_text, fmt=lig_fmt, width=1080,
                                      height=340), height=340)
        except Exception as e:
            st.info(f"Ligand viewer unavailable ({e}).")

with st.container(border=True):
    st.markdown('<div class="pb-sec">3 · Detector &amp; box settings</div>',
                unsafe_allow_html=True)

    def _backend_label(s: str) -> str:
        name = "fpocket" if s == "fpocket" else "P2Rank"
        return name if AVAIL[s] else f"{name} — not installed"

    # Default to a backend that is actually available on this machine.
    _default = next((i for i, b in enumerate(BACKENDS) if AVAIL[b]), 0)

    b0, b1, b2, b3 = st.columns(4)
    backend = b0.radio("Pocket detector", list(BACKENDS), index=_default,
                       format_func=_backend_label)
    buffer = b1.slider("Buffer (Å)", 0.0, 12.0, 5.0, 0.5)
    cubic = b2.checkbox("Force cubic box", value=False)
    exhaust = b3.slider("Vina exhaustiveness", 1, 32, 16)

backend_ok = AVAIL.get(backend, False)
if not backend_ok:
    _name = "fpocket" if backend == "fpocket" else "P2Rank"
    if any(AVAIL.values()):
        st.warning(
            f"**{_name}** is not installed / not on PATH on this machine, so "
            "detection can't run with it. Pick an available backend above.")
    else:
        st.error(
            "Neither fpocket nor P2Rank is installed on this machine. Install "
            "one (see the README) and restart the app to detect pockets.")

run = st.button("Detect pockets & rank", type="primary",
                use_container_width=True, disabled=not backend_ok)

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
    st.session_state["ligand_name"] = profile.name

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def _badge(rank: int) -> str:
    cls = {1: "pb-b1", 2: "pb-b2", 3: "pb-b3"}.get(rank, "pb-bn")
    return f'<span class="pb-badge {cls}">#{rank}</span>'


if "ranked" in st.session_state:
    ranked = st.session_state["ranked"]
    pdb_text = st.session_state["pdb_text"]
    ligand_name = st.session_state.get("ligand_name", "ligand")

    st.divider()
    st.markdown(f"### Results · target vs **{ligand_name}**")

    top = ranked[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pockets found", len(ranked))
    k2.metric("Top match score", f"{top['score'].total:.2f}")
    k3.metric("Top druggability",
              "—" if top["_pocket"].druggability is None
              else f"{top['_pocket'].druggability:.2f}")
    _vol = top["_pocket"].volume
    k4.metric("Top pocket volume", "—" if _vol is None else f"{_vol:.0f} Å³")

    st.markdown("#### Ranked pockets")
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
    st.dataframe(
        table, use_container_width=True, hide_index=True,
        column_config={
            "match score": st.column_config.ProgressColumn(
                "match score", min_value=0.0, max_value=1.0, format="%.3f"),
            "druggability": st.column_config.ProgressColumn(
                "druggability", min_value=0.0, max_value=1.0, format="%.3f"),
        },
    )

    labels = [f"#{i+1} · pocket {r['_pocket'].id} "
              f"(score {r['score'].total})" for i, r in enumerate(ranked)]
    sel = st.selectbox("Inspect pocket", range(len(ranked)),
                       format_func=lambda i: labels[i])
    chosen = ranked[sel]
    pocket = chosen["_pocket"]

    st.markdown(f"{_badge(sel + 1)} &nbsp; **Pocket {pocket.id}**",
                unsafe_allow_html=True)

    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Match score", f"{chosen['score'].total:.3f}")
        m2.metric("Druggability",
                  "—" if pocket.druggability is None
                  else f"{pocket.druggability:.3f}")
        m3.metric("Hydrophobic frac",
                  f"{pocket.descriptors.hydrophobic_fraction:.2f}")
        with st.expander("Why this pocket scored as it did", expanded=True):
            st.write({k: v for k, v in chosen["score"].terms.items()})
            for reason in chosen["score"].reasons:
                st.markdown(f"- {reason}")

    # box
    box = compute_box(pocket.points_for_box(), buffer=buffer, cubic=cubic)
    pocket_pts = pocket.points_for_box()[:400]
    cfg = write_vina_config(box, exhaustiveness=exhaust)

    st.markdown(
        f"The docking box is locked to pocket **{pocket.id}** — rotate a view "
        "and it stays on the active site. Its walls are coloured by dimension: "
        "**X red · Y green · Z blue** (opposite walls share a colour), and kept "
        "translucent so the protein shows through.")
    t_cart, t_surf, t_cfg = st.tabs(
        ["Cartoon view", "Surface view", "Vina config"])
    try:
        from pocketbox.visualize import build_view, build_surface_view
        with t_cart:
            st.caption("Cartoon · pocket cavity (orange) · box walls X/Y/Z = "
                       "red/green/blue")
            _show3d(build_view(pdb_text, box=box, pocket_atom_coords=pocket_pts,
                               width=820, height=480), height=480)
        with t_surf:
            st.caption("Translucent surface · reveals the cavity · box at pocket")
            _show3d(build_surface_view(pdb_text, box=box,
                                       pocket_atom_coords=pocket_pts,
                                       width=820, height=480), height=480)
    except Exception as e:
        with t_cart:
            st.info(f"3D viewer unavailable ({e}). Box params are in the "
                    "'Vina config' tab.")

    with t_cfg:
        st.markdown(f"**Docking box (AutoDock Vina)** — pocket {pocket.id}")
        bcol1, bcol2 = st.columns([2, 3])
        with bcol1:
            st.json(box.as_dict())
            st.download_button("Download vina config", cfg,
                               file_name=f"vina_pocket{pocket.id}.txt",
                               use_container_width=True)
        with bcol2:
            st.code(cfg, language="ini")

    st.divider()
    st.markdown(
        "**Next:** prepare a receptor `.pdbqt` (e.g. with Meeko or "
        "AutoDockTools `prepare_receptor`), then run "
        f"`vina --config vina_pocket{pocket.id}.txt`. If this target has a "
        "co-crystallised ligand, re-dock it first and check RMSD (≲2 Å) to "
        "validate the box.")
else:
    st.markdown(
        '<div class="pb-foot">Set a structure and a ligand above, then '
        '<b>Detect pockets &amp; rank</b> to see ranked pockets, 3D views, and '
        'the exportable Vina box.</div>', unsafe_allow_html=True)

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
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
      html, body, .stApp, [data-testid="stAppViewContainer"], [class*="css"] {
        font-family: 'IBM Plex Sans', system-ui, sans-serif;
      }
      h1, h2, h3, .pb-sec, .pb-hero h1 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.02em;
      }
      .pb-step { font-family: 'IBM Plex Mono', monospace !important;
        text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.74rem; }
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
      /* clear theme: make surfaces transparent so the molecular canvas shows */
      .stApp { background: transparent !important; }
      [data-testid="stHeader"] { background: transparent !important; }
      body { background:
        radial-gradient(1200px 800px at 15% -10%, #eef0ff 0%,
                        #f6f7fb 45%, #ffffff 100%) fixed; }
      /* solid white cards so the moving molecules never sit behind the text */
      [class*="st-key-pbcard"] {
        background: #ffffff !important;
        border: 1px solid #e7e5f2 !important;
        border-radius: 16px !important;
        box-shadow: 0 6px 22px rgba(30,26,60,0.07) !important;
        padding: 14px 16px !important;
      }
      /* the ranked-pocket table also gets a solid backing */
      [data-testid="stDataFrame"] { background:#ffffff !important;
        border-radius: 12px; }
    </style>
    """
)

# Ambient molecular background: atoms drifting with bonds between near neighbours,
# painted on a canvas behind the whole app (self-contained JS, no CDN needed).
components.html(
    """
    <script>
    (function () {
      const doc = window.parent.document, win = window.parent;
      if (doc.getElementById('pb-bg')) return;              // guard against reruns
      const c = doc.createElement('canvas'); c.id = 'pb-bg';
      Object.assign(c.style, {position:'fixed', inset:'0', width:'100%',
        height:'100%', zIndex:'-1', pointerEvents:'none', opacity:'0.9'});
      doc.body.prepend(c);
      const ctx = c.getContext('2d');
      const reduce = win.matchMedia('(prefers-reduced-motion: reduce)').matches;
      const dpr = Math.min(win.devicePixelRatio || 1, 2);
      // element table: CPK-ish colour, relative abundance (w), radius, valence
      const EL = [
        {s:'C', c:[123,122,145], w:46, r:2.0, val:3},
        {s:'N', c:[74,107,214],  w:20, r:2.2, val:2},
        {s:'O', c:[224,82,82],   w:20, r:2.2, val:2},
        {s:'S', c:[216,167,46],  w:8,  r:2.6, val:2},
        {s:'P', c:[224,123,46],  w:6,  r:2.6, val:2}
      ];
      const ACC = [109,94,252];                      // accent-tinted bonds
      const N = 92, LINK = 185, SPEED = 0.20;
      let W, H, atoms;
      function pick(){ const t = EL.reduce((s,e)=>s+e.w,0); let r = Math.random()*t;
        for (const e of EL){ r -= e.w; if (r <= 0) return e; } return EL[0]; }
      function build(){
        W = doc.body.clientWidth; H = win.innerHeight;
        c.width = W*dpr; c.height = H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
        atoms = Array.from({length: N}, () => { const e = pick();
          return {x:Math.random()*W, y:Math.random()*H,
                  vx:(Math.random()-0.5)*SPEED, vy:(Math.random()-0.5)*SPEED, e}; });
      }
      function frame(){
        ctx.clearRect(0, 0, W, H);
        // valence-limited bonds: each atom bonds to its nearest `val` neighbours
        const seen = new Set();
        for (let i = 0; i < atoms.length; i++) {
          const a = atoms[i], near = [];
          for (let j = 0; j < atoms.length; j++) { if (i === j) continue;
            const b = atoms[j], d = Math.hypot(a.x-b.x, a.y-b.y);
            if (d < LINK) near.push({j, d, b}); }
          near.sort((p,q) => p.d - q.d);
          near.slice(0, a.e.val).forEach(({j, d, b}, k) => {
            const key = i < j ? i+':'+j : j+':'+i; if (seen.has(key)) return; seen.add(key);
            const fade = (1 - d/LINK) * 0.34;
            ctx.strokeStyle = 'rgba('+ACC[0]+','+ACC[1]+','+ACC[2]+','+fade.toFixed(3)+')';
            ctx.lineWidth = 1;
            const dbl = k === 0 && (a.e.s === 'O' || b.e.s === 'O') && d < LINK*0.55;
            if (dbl) {                                  // draw a double bond
              const ux = -(b.y-a.y)/d, uy = (b.x-a.x)/d, o = 1.9;
              ctx.beginPath(); ctx.moveTo(a.x+ux*o, a.y+uy*o); ctx.lineTo(b.x+ux*o, b.y+uy*o); ctx.stroke();
              ctx.beginPath(); ctx.moveTo(a.x-ux*o, a.y-uy*o); ctx.lineTo(b.x-ux*o, b.y-uy*o); ctx.stroke();
            } else {
              ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
            }
          });
        }
        // atoms: element-tinted dot with a soft halo
        for (const p of atoms) {
          if (!reduce) { p.x += p.vx; p.y += p.vy;
            if (p.x < 0 || p.x > W) p.vx *= -1; if (p.y < 0 || p.y > H) p.vy *= -1; }
          const [r,g,bl] = p.e.c, rad = p.e.r + 0.4;
          const grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, rad*3.4);
          grd.addColorStop(0, 'rgba('+r+','+g+','+bl+',0.5)');
          grd.addColorStop(1, 'rgba('+r+','+g+','+bl+',0)');
          ctx.fillStyle = grd; ctx.beginPath(); ctx.arc(p.x, p.y, rad*3.4, 0, 7); ctx.fill();
          ctx.fillStyle = 'rgba('+r+','+g+','+bl+',0.92)';
          ctx.beginPath(); ctx.arc(p.x, p.y, rad, 0, 7); ctx.fill();
        }
        if (!reduce) win.requestAnimationFrame(frame);
      }
      win.addEventListener('resize', build);
      build(); frame();
    })();
    </script>
    """,
    height=0,
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

# --------------------------------------------------------------------------- #
# Secondary pages (Gallery / About / Docs)
# --------------------------------------------------------------------------- #
_EXAMPLES = [
    ("2V5Z", "MAO-B", "Monoamine oxidase B — the running example; FAD-adjacent "
                      "inhibitor site."),
    ("2Z5X", "MAO-A", "Monoamine oxidase A; substrate / inhibitor cavity."),
    ("3EML", "Adenosine A2A", "A2A GPCR; orthosteric antagonist pocket."),
    ("3PBL", "Dopamine D3", "D3 receptor; orthosteric site."),
    ("2VT4", "β1-adrenergic", "β1 receptor; antagonist pocket."),
    ("2BYB", "MAO-B", "A second MAO-B crystal form for cross-checking."),
]


def _render_gallery() -> None:
    st.markdown("### Worked examples")
    st.caption("Structures with well-characterised binding sites, relevant to "
               "Parkinson's-disease targets. Copy a PDB ID into the Analyze "
               "page and run PocketBox on it.")
    cols = st.columns(3)
    for i, (pdb, name, note) in enumerate(_EXAMPLES):
        with cols[i % 3]:
            with st.container(border=True, key=f"pbcard-gal{i}"):
                st.markdown(f"**`{pdb}`** · {name}")
                st.caption(note)


def _render_about() -> None:
    st.markdown("### About & credits")
    st.markdown(
        "PocketBox wraps established detectors — **fpocket** and **P2Rank** — "
        "behind one page, adds a druggability-aware ranking, and emits the "
        "AutoDock Vina grid box most people end up computing by hand. "
        "Everything runs on open tools; nothing is sent to a third-party "
        "service.")
    st.markdown("**Built on**")
    tools = [
        ("fpocket / P2Rank", "Pocket detection (geometry + machine learning)."),
        ("RDKit", "Ligand parsing, descriptors, and 3D embedding."),
        ("AutoDock Vina", "The docking engine the exported box targets."),
        ("py3Dmol / 3Dmol.js", "In-browser 3D structure and box rendering."),
    ]
    cols = st.columns(2)
    for i, (n, w) in enumerate(tools):
        with cols[i % 2]:
            with st.container(border=True, key=f"pbcard-cred{i}"):
                st.markdown(f"**{n}**")
                st.caption(w)
    st.caption("Built by Ayoub Riad · Research use only — predictions are "
               "computational hypotheses, not clinical or diagnostic guidance.")


def _render_docs() -> None:
    st.markdown("### How the pipeline works")
    steps = [
        ("Fetch / read structure",
         "Download by PDB ID from RCSB, or read an uploaded .pdb file."),
        ("Detect pockets",
         "fpocket (geometry) or P2Rank (ML) enumerate candidate cavities."),
        ("Describe",
         "Per-pocket hydrophobicity, net charge, metals, volume, lining residues."),
        ("Rank vs ligand",
         "Blend druggability with ligand compatibility into a transparent score."),
        ("Compute Vina box",
         "Centre + size from the pocket's alpha-sphere extent, plus a buffer."),
        ("Export",
         "A ready-to-run Vina config; re-dock a known ligand to validate it."),
    ]
    for i, (title, body) in enumerate(steps, 1):
        with st.container(border=True, key=f"pbcard-doc{i}"):
            st.markdown(f"**{i} · {title}**")
            st.caption(body)
    st.markdown("**Command line**")
    st.code("python cli.py --pdb-id 2V5Z --ligand-class maob_inhibitor "
            "--backend p2rank --top 5", language="bash")


def render_secondary_page(name: str) -> None:
    {"Gallery": _render_gallery, "About": _render_about,
     "Docs": _render_docs}.get(name, _render_gallery)()


# --------------------------------------------------------------------------- #
# Sidebar: navigation + honest framing
# --------------------------------------------------------------------------- #
with st.sidebar:
    page = st.radio("Section", ["Analyze", "Gallery", "About", "Docs"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown("**About**")
    st.caption("PocketBox detects candidate ligand-binding pockets, ranks them "
               "by druggability, and emits a ready-to-run docking box.")
    st.warning("Ranking is a **shortlisting heuristic**, not a predictor of the "
               "true binding site. Confirm by docking; re-dock a known ligand "
               "to validate the box when a co-crystallised one exists.")
    st.caption("fpocket · P2Rank · RDKit · py3Dmol / 3Dmol.js")

# Analyze hero — only on the Analyze page.
if page == "Analyze":
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

# Gallery / About / Docs render here, then stop before the Analyze tool UI.
if page != "Analyze":
    render_secondary_page(page)
    st.stop()

# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
c1, c2 = st.columns(2)

with c1:
    with st.container(border=True, key="pbcard-structure"):
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
    with st.container(border=True, key="pbcard-ligand"):
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
    with st.container(border=True, key="pbcard-lig3d"):
        st.markdown('<div class="pb-sec">Ligand in 3D</div>',
                    unsafe_allow_html=True)
        st.caption("Ball-and-stick · drag to rotate · scroll to zoom")
        try:
            from pocketbox.visualize import build_ligand_view
            _show3d(build_ligand_view(lig_text, fmt=lig_fmt, width=1080,
                                      height=340), height=340)
        except Exception as e:
            st.info(f"Ligand viewer unavailable ({e}).")

with st.container(border=True, key="pbcard-detector"):
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

    with st.container(border=True, key="pbcard-detail"):
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
        "**X red · Y green · Z blue** (opposite walls share a colour), drawn "
        "solid with a bold outline.")
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

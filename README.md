# PocketBox

**Ligand-aware binding-pocket triage + AutoDock Vina docking-box export.**

Give PocketBox a protein (PDB ID or file) and a ligand (a class, or a SMILES).
It detects all pockets, ranks them for *compatibility* with that ligand,
shows the chosen pocket and its docking box in 3D, and exports a ready-to-run
Vina config — center `(x, y, z)` and size `(x, y, z)` included.

It runs as a Streamlit web app and as a command-line tool, and ships with a
Docker build so you can host a public demo.

---

## What it does — and what it does *not*

Three things here are well-posed and reliable:

1. **Pocket detection** from a structure — all cavities, with druggability
   scores, lining residues, and coordinates. Two interchangeable backends:
   **fpocket** (fast, geometry-based) and **P2Rank** (machine-learning).
2. **Docking-box geometry** — deterministic; the center and size fall straight
   out of the pocket's alpha-sphere coordinates.
3. **Visualization** of protein + pocket + box.

The part that is **a heuristic, not an oracle**, is *which* pocket a given
ligand binds. PocketBox **shortlists and ranks** pockets by blending the
fpocket druggability score with chemistry-grounded compatibility (hydrophobic
pockets favour lipophilic ligands; cationic pockets favour anionic ligands;
metal sites favour coordinating ligands; covalent warheads need a reactive
residue). That is a sensible **filter**, but it does **not** prove the true
binding site.

> **Use the ranking to pick a few candidates, then confirm by docking.** When a
> target has a co-crystallised ligand, re-dock it first and check RMSD (≲ 2 Å)
> to validate box placement before trusting scores on a novel compound.

The scoring is fully transparent — every term's contribution and a plain-text
reason are shown for each pocket — so it can be described honestly in a methods
section.

---

## Install (recommended: conda — includes fpocket)

```bash
conda env create -f environment.yml
conda activate pocketbox
streamlit run app.py
```

`fpocket` is a compiled binary and is **not** a pip package; the conda
environment above installs it from bioconda along with everything else.

### pip alternative

```bash
pip install -r requirements.txt          # python deps only
# then install fpocket separately, e.g.:
conda install -c bioconda fpocket         # or build from source
streamlit run app.py
```

---

## Command-line use

```bash
# by PDB ID, generic inhibitor, top 5 pockets (fpocket backend)
python cli.py --pdb-id 2V5Z --ligand-class small_molecule_inhibitor --top 5

# same, but use the P2Rank (machine-learning) detector
python cli.py --pdb-id 2V5Z --ligand-class small_molecule_inhibitor --backend p2rank

# local file + ligand as SMILES
python cli.py --pdb-file receptor.pdb --smiles "CC(=O)Oc1ccccc1C(=O)O"
```

Each top pocket is printed with its match score, druggability, and box, and a
`vina_pocket<N>.txt` config is written to `--outdir`.

### Choosing a detector

- **fpocket** — geometry/Voronoi-based, fast, reports a druggability score and a
  pocket volume. Default.
- **P2Rank** — machine-learning predictor (Java; CLI `prank`). It reports pocket
  centers and a calibrated ligandability *probability*, which PocketBox uses in
  the druggability slot. P2Rank does not report a pocket volume by default, so
  the size term in ranking is treated as "unknown" for that backend.

Both produce the same downstream `Pocket` objects, so ranking, the box, and the
viewer behave identically. Pick the backend in the app (radio button) or with
`--backend {fpocket,p2rank}` on the CLI. The conda environment installs both
(P2Rank needs Java, which the environment also pulls in). If your conda channels
don't carry `p2rank`, download the standalone P2Rank distribution and put its
`prank` launcher on PATH — I'm not 100% certain of bioconda's current P2Rank
packaging, so verify that step.

---

## Redocking validation (optional, local only)

`redock.py` re-docks a target's **native (co-crystallised) ligand** inside a box
centred on that ligand and reports, per engine, the top affinity and a
symmetry-corrected RMSD to the crystal pose (pass if ≲ 2 Å). This is the check
that turns box placement into something you've *validated* rather than asserted.

Three interchangeable engines share Vina's flag interface — **AutoDock Vina**,
**smina**, **QuickVina2** — selected with `--engine {vina,smina,qvina2,all}`:

```bash
python redock.py --pdb-id 2V5Z --ligand-resname A3T --engine all \
    --template-smiles "CC(=O)..."   # SMILES strongly recommended for reliable RMSD
```

RMSD uses RDKit's in-place, symmetry-aware `CalcRMS`; a *superimposing* RMSD
would hide a pose that landed in the wrong place, so it is deliberately not used.

Runtime requirements (on your machine, **not** the hosted demo): a prep tool
(ADFR/AutoDockTools `prepare_receptor`/`prepare_ligand`, or Open Babel) to build
PDBQT inputs, at least one docking engine, and Open Babel for the pose→PDB
conversion the RMSD step needs. Install whatever you already use — e.g. Open
Babel and smina are available via conda. I have **not** verified exact conda
package names for every engine (QuickVina2 in particular is often a manual
download whose binary may be `qvina2` or `qvina02`), so confirm those locally.

Docking is intentionally **not** exposed in the Streamlit app — running engines
on a shared hosted demo is slow and unsafe; keep redocking to the CLI/library.

## Deploy a public demo

A hosted demo needs fpocket installed on the host, so the container route is the
reliable one:

- **Docker / Hugging Face Spaces (Docker template):** the included `Dockerfile`
  uses a conda base and installs fpocket from bioconda — works out of the box.
  ```bash
  docker build -t pocketbox . && docker run -p 8501:8501 pocketbox
  ```
- **Streamlit Community Cloud:** straightforward for the Python deps, but it
  installs system packages via `apt`. I have *not* verified that fpocket is
  available through the default apt repositories, so the Docker route is the
  safer bet for a guaranteed-working public demo. If you confirm an apt source
  for fpocket, a `packages.txt` deploy would also work.

---

## How it works

```
PDB id / file
   → fetch & read structure        (pocketbox/fetch.py)
   → detect pockets               (pocketbox/detect.py · detect_p2rank.py)  fpocket OR P2Rank
   → describe each pocket          (pocketbox/descriptors.py)  hydrophobicity, charge, metals…
   → rank vs ligand profile        (pocketbox/match.py)    transparent weighted score + reasons
   → compute Vina box              (pocketbox/box.py)      center (x,y,z) + size (x,y,z)
   → 3D view  (py3Dmol / stmol)    (pocketbox/visualize.py)
   → export Vina config + receptor-prep instructions
```

Ligand descriptors (MW, logP, H-bond donors/acceptors, charge, aromatic rings,
metals) come from RDKit (`pocketbox/ligand.py`), which also holds the editable
ligand-class taxonomy.

---

## Extending the ligand taxonomy

Open `pocketbox/ligand.py` and add a `LigandProfile` to `LIGAND_CLASSES`. Each
profile states the kind of pocket the class is compatible with: a target
hydrophobic fraction, a charge preference, a size preference, and flags for
needing a metal / reactive residue / aromatic residues. The PD-relevant entries
(`maob_inhibitor`, `kinase_inhibitor`, `a2a_antagonist`) are starting points —
tune them to your own targets.

---

## Box-size conventions (please read)

The numbers PocketBox uses are **rules of thumb, not constants**: it sizes the
box to the pocket extent plus a buffer (default 5 Å), clamped to a sensible
range, ending up roughly 18–25 Å per side for typical pockets. Boxes that are
too large hurt both accuracy and speed. Tune per target and validate by
re-docking. Verify against primary docking literature for your system.

---

## Tested vs. not (be honest about it)

- **Verified (unit tests, `tests/` — 9/9 passing):** box geometry,
  ligand-compatibility ranking, the **P2Rank predictions-CSV parser**,
  **native-ligand extraction**, **docking-score parsing** (Vina/smina/QuickVina
  formats), and the **RMSD computation** (RDKit `CalcRMS`), plus the RDKit
  descriptor layer (aspirin → MW 180.16, logP 1.31).
- **Verified live (Python 3.13 + numpy + RDKit):** the **RCSB download path**
  (`pocketbox/fetch.py`) against the real endpoint, and the
  structure-parse → native-ligand-extract → box-geometry path on a genuine
  target (2V5Z, human MAO-B: 8,778 atoms; FAD + SAG resolved).
- **Not yet verified end-to-end:** the live **fpocket**, **P2Rank**, and
  **Vina/smina/QuickVina2** runs and the **PDBQT preparation** — these need
  their respective binaries on PATH (the geometry/ML detectors and docking
  engines are conda/Linux tools, not pip packages). The wrappers use the
  standard tool interfaces, but **try one real PDB end-to-end locally before
  relying on results.** Output formats and prep-tool CLIs have drifted across
  versions; the parsers/wrappers are defensive, but check them against your
  installed versions.

Run the tests:

```bash
python -m tests.test_core
```

---

## Acknowledgements / citing the upstream tools

PocketBox is glue around established tools — please cite them in any work that
uses it: **fpocket** (pocket detection), **RDKit** (cheminformatics),
**AutoDock Vina** (docking), **py3Dmol / 3Dmol.js** and **stmol** (3D
viewing), and **RCSB PDB** (structures). Refer to each project's official
repository and publication for the correct citation; this README intentionally
does not reproduce specific reference strings or DOIs to avoid getting them
wrong.

## License

MIT — see [LICENSE](LICENSE) (add your name).

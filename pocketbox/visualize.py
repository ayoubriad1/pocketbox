"""3D visualization with py3Dmol: cartoon + highlighted pocket + wireframe box.

Returns a py3Dmol view object. In Streamlit, render it with stmol.showmol(view).
Outside Streamlit you can call view.show() in a notebook, or view._make_html().
"""

from __future__ import annotations
from .box import Box


def build_view(pdb_text: str,
               box: Box | None = None,
               pocket_resids: list[int] | None = None,
               pocket_atom_coords=None,
               width: int = 800,
               height: int = 520):
    """Construct a py3Dmol view.

    pdb_text          : the full PDB file as a string.
    box               : Box to draw as a wireframe (optional).
    pocket_resids     : residue serial numbers to show as sticks (optional).
    pocket_atom_coords: iterable of (x,y,z) to render as translucent spheres,
                        used to mark the pocket cavity when resids are unknown.
    """
    try:
        import py3Dmol
    except ImportError as e:                                    # pragma: no cover
        raise ImportError(
            "py3Dmol is required for visualization (`pip install py3Dmol`).") from e

    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})

    if pocket_resids:
        view.addStyle({"resi": [str(r) for r in pocket_resids]},
                      {"stick": {"colorscheme": "orangeCarbon"}})

    if pocket_atom_coords is not None:
        for (x, y, z) in pocket_atom_coords:
            view.addSphere({"center": {"x": float(x), "y": float(y), "z": float(z)},
                            "radius": 0.4, "color": "orange", "opacity": 0.35})

    if box is not None:
        cx, cy, cz = box.center
        sx, sy, sz = box.size
        view.addBox({
            "center": {"x": cx, "y": cy, "z": cz},
            "dimensions": {"w": sx, "h": sy, "d": sz},
            "color": "magenta",
            "opacity": 0.9,
            "wireframe": True,
            "linewidth": 2.0,
        })

    view.zoomTo()
    return view


def build_ligand_view(mol_text: str,
                      fmt: str = "mol",
                      width: int = 760,
                      height: int = 360,
                      style: str = "ball_and_stick"):
    """Render a small molecule (SMILES-derived mol block or an uploaded
    mol/sdf/pdb) as an interactive, rotatable 3D model.

    mol_text : the molecule as text (MDL mol block or PDB).
    fmt      : 3Dmol model format ("mol", "sdf", "pdb").
    style    : "ball_and_stick" or "stick".
    """
    try:
        import py3Dmol
    except ImportError as e:                                    # pragma: no cover
        raise ImportError(
            "py3Dmol is required for the ligand viewer (`pip install py3Dmol`).") from e

    view = py3Dmol.view(width=width, height=height)
    view.addModel(mol_text, fmt)
    if style == "stick":
        view.setStyle({"stick": {"radius": 0.14}})
    else:
        view.setStyle({"stick": {"radius": 0.14},
                       "sphere": {"scale": 0.26}})
    view.setBackgroundColor("white")
    view.zoomTo()
    return view


def build_surface_view(pdb_text: str,
                       box: Box | None = None,
                       pocket_atom_coords=None,
                       width: int = 800,
                       height: int = 520,
                       surface_opacity: float = 0.5,
                       surface_color: str = "white"):
    """Render the protein as a translucent molecular surface so the pocket
    cavity is visible, with the docking box drawn at the pocket.

    The box is added in the model's own coordinate frame, so it stays locked
    to the active site when the view is rotated (protein and box rotate as one).

    pdb_text          : the full PDB file as a string.
    box               : Box to draw as a wireframe at the pocket (optional).
    pocket_atom_coords: iterable of (x,y,z) marking the pocket, shown as
                        translucent spheres to locate the cavity on the surface.
    """
    try:
        import py3Dmol
    except ImportError as e:                                    # pragma: no cover
        raise ImportError(
            "py3Dmol is required for visualization (`pip install py3Dmol`).") from e

    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_text, "pdb")
    # faint cartoon underneath gives structural context through the surface.
    view.setStyle({"cartoon": {"color": "spectrum"}})
    # translucent van-der-Waals surface over the whole protein -> reveals cavity.
    surftype = getattr(py3Dmol, "VDW", 1)
    view.addSurface(surftype, {"opacity": surface_opacity, "color": surface_color})

    if pocket_atom_coords is not None:
        for (x, y, z) in pocket_atom_coords:
            view.addSphere({"center": {"x": float(x), "y": float(y), "z": float(z)},
                            "radius": 1.2, "color": "orange", "opacity": 0.9})

    if box is not None:
        cx, cy, cz = box.center
        sx, sy, sz = box.size
        view.addBox({
            "center": {"x": cx, "y": cy, "z": cz},
            "dimensions": {"w": sx, "h": sy, "d": sz},
            "color": "magenta",
            "opacity": 0.95,
            "wireframe": True,
            "linewidth": 2.5,
        })

    view.zoomTo()
    return view

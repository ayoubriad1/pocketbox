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
            "opacity": 0.18,
            "wireframe": True,
        })

    view.zoomTo()
    return view

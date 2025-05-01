import os
import logging
from ..core import Pdb

logger = logging.getLogger(__name__)


def view_frustration_pymol(pdb: Pdb) -> None:
    # Implementation from visualization.py
    ...

def view_config_contacts_py3dmol(
    pdb: Pdb,
    central_chain: str,
    central_res: int,
    width: int = 800,
    height: int = 600,
    title: str = None
) -> None:
    """
    Visualize configurational contacts for a given residue using py3Dmol.

    This method loads the configurational frustration results and PDB structure,
    builds a summary table, displays it in the notebook, and launches a py3Dmol
    viewer with contacts drawn as cylinders whose thickness is proportional to
    frustration magnitude.

    Args:
        pdb: Pdb frustration object containing paths and data
        central_chain: Chain identifier for the residue of interest
        central_res: Residue number for the residue of interest
        width: Width of the viewer in pixels (default: 800)
        height: Height of the viewer in pixels (default: 600)
        title: Custom title for the visualization (default: auto-generated)
    """
    import os
    import pandas as pd
    import py3Dmol
    from IPython.display import display, HTML
    from .py3dmol_utils import (
        load_config_contacts,
        build_ca_map,
        create_summary_df,
        style_frustration,
        scale_radius,
    )

    # locate files based on pdb object
    frust_file = os.path.join(
        pdb.job_dir,
        "FrustrationData",
        f"{pdb.pdb_base}.pdb_{pdb.mode}"
    )
    pdb_file = os.path.join(
        pdb.job_dir,
        "VisualizationScripts",
        f"{pdb.pdb_base}.pdb"
    )

    # load and filter contacts
    contacts, contact_residues, res_names = load_config_contacts(
        frust_file, central_chain, central_res
    )
    central_resname = res_names.get((central_chain, central_res), '')
    central_label = f"{central_resname}_{central_res}{central_chain}"
    
    # Set descriptive title and caption
    if title is None:
        # Auto-generate descriptive titles
        table_caption = f"Contact {pdb.mode} frustration for residue {central_res} chain {central_chain} ({central_resname})"
        viewer_title = f"{central_resname}_{central_res}{central_chain} ({pdb.mode} frustration)"
    else:
        table_caption = title
        viewer_title = title
    
    print(f"Found {len(contacts)} contacts to {central_label}")

    # create and display summary table
    summary_df = create_summary_df(contacts).copy()
    # format magnitude
    summary_df['Magnitude'] = summary_df['Magnitude'].map(lambda x: f"{x:.3f}")
    styled = summary_df.style.map(
        style_frustration, subset=['State']
    ).set_caption(
        table_caption
    ).set_table_styles([
        {'selector': 'caption', 'props': [
            ('font-weight', 'bold'),
            ('font-size', '16px'),
            ('text-align', 'center')
        ]},
        {'selector': 'th', 'props': [
            ('font-weight', 'bold'),
            ('background-color', '#e6e6e6')
        ]}
    ])
    display(HTML(styled.to_html()))

    # build Cα coordinate map
    ca_map = build_ca_map(pdb_file)
    ctr = ca_map[(central_chain, central_res)]

    # prepare radius scaling
    mags = [mag for *_, mag, _ in contacts]
    min_mag, max_mag = min(mags), max(mags)

    # start py3Dmol viewer
    view = py3Dmol.view(width=width, height=height)
    with open(pdb_file) as f:
        view.addModel(f.read(), 'pdb')

    # global cartoon style
    view.setStyle({'cartoon': {'color': 'lightgray', 'opacity': 0.7}})

    # highlight central residue
    view.setStyle(
        {'chain': central_chain, 'resi': central_res},
        {
            'cartoon': {'color': 'yellow'},
            'stick': {'colorscheme': 'yellowCarbon', 'radius': 0.2}
        }
    )

    # style contact residues
    for ch, resnum in contact_residues:
        if (ch, resnum) == (central_chain, central_res):
            continue
        # find state
        state = next(
            (s for c, r, s, *_ in contacts if c == ch and r == resnum),
            'neutral'
        )
        if 'highly' in state.lower():
            cart_col = 'salmon'
        elif 'minimally' in state.lower():
            cart_col = 'lightgreen'
        else:
            cart_col = 'lightblue'
        view.setStyle(
            {'chain': ch, 'resi': resnum},
            {'stick': {'radius': 0.15}, 'cartoon': {'color': cart_col}}
        )

    # add cylinders for contacts
    color_map = {'highly': 'red', 'neutral': 'gray', 'minimally': 'green', 'native': 'blue'}
    for ch, resnum, state, mag, resname in contacts:
        pt = ca_map.get((ch, resnum))
        if not pt:
            continue
        col = next((c for k, c in color_map.items() if k in state.lower()), 'black')
        rad = scale_radius(mag, min_mag, max_mag)
        view.addCylinder({
            'start': {'x': ctr[0], 'y': ctr[1], 'z': ctr[2]},
            'end':   {'x': pt[0], 'y': pt[1], 'z': pt[2]},
            'radius': rad,
            'color': col,
            'fromCap': 1,
            'toCap': 1,
            'dashed': True
        })

    # add labels for contact residues
    for ch, resnum, state, mag, resname in contacts:
        pt = ca_map.get((ch, resnum))
        if not pt:
            continue
        # More informative label with magnitude
        label = f"{resname}_{resnum}{ch}\n({state.capitalize()})\n|{mag:.2f}|"
        view.addLabel(label, {
            'position': {'x': pt[0], 'y': pt[1], 'z': pt[2]},
            'backgroundColor': 'black',
            'fontColor': 'white',
            'fontSize': 12,
            'backgroundOpacity': 0.6,
            'inFront': True
        })

    # label central residue
    central_desc = f"{central_resname}_{central_res}{central_chain}\n(Center)"
    view.addLabel(
        central_desc,
        {'position': {'x': ctr[0], 'y': ctr[1], 'z': ctr[2]},
         'backgroundColor': 'gold', 'fontColor': 'black',
         'fontSize': 14, 'backgroundOpacity': 0.8, 'inFront': True}
    )

    # Add a title at the top of the viewer
    view.addLabel(
        viewer_title,
        {'position': {'x': ctr[0], 'y': ctr[1] - 25, 'z': ctr[2]},
         'backgroundColor': 'white',
         'fontColor': 'black',
         'fontSize': 16,
         'backgroundOpacity': 0.8,
         'inFront': True}
    )

    # zoom to all involved residues
    zoom_sel = {'or': [{ 'chain': c, 'resi': r } for c, r in contact_residues]}
    view.zoomTo(zoom_sel)

    # show viewer
    view.show()

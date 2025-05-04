import os
import logging
from ..core import Pdb
from Bio.SeqUtils import seq1

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
    title: str = None,
    show_neutral_labels: bool = True,
    show_neutral_contacts: bool = True
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
        show_neutral_labels: Whether to display labels for neutrally frustrated contacts (default: True)
        show_neutral_contacts: Whether to display cylinders for neutrally frustrated contacts (default: True)
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
        get_residue_name,
    )

    # Import seq1 for one-letter code conversion
    from Bio.SeqUtils import seq1

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
    # Get central residue name directly from PDB object for reliability
    central_resname = get_residue_name(pdb, central_chain, central_res)
    try:
        central_resname_one = seq1(central_resname)
    except KeyError:
        central_resname_one = central_resname # Fallback to 3-letter if non-standard
    central_label = f"{central_resname_one}{central_res}" # Removed chain
    
    # Set descriptive title and caption
    if title is None:
        # Auto-generate descriptive titles
        table_caption = f"Contact {pdb.mode} frustration for residue {central_resname_one}{central_res} ({central_resname})"
        viewer_title = f"{central_resname_one}{central_res} ({pdb.mode} frustration)" # Removed chain
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
        # Check if we should skip drawing neutral contacts
        state_lower = state.lower()
        is_neutral = 'highly' not in state_lower and 'minimally' not in state_lower
        if is_neutral and not show_neutral_contacts:
            continue # Skip adding cylinder for neutral contacts if flag is False

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
        # Determine label color based on frustration state
        state_lower = state.lower()
        if 'highly' in state_lower:
            label_font_color = 'red'
        elif 'minimally' in state_lower:
            label_font_color = 'lightgreen' # Or just 'green'?
        else:
            # Handle neutral state based on the flag
            if not show_neutral_labels:
                continue # Skip adding label for neutral contacts if flag is False
            label_background_color = 'gray' # Or 'lightblue' to match cartoon?

        # Set background color based on state
        if 'highly' in state_lower:
            label_background_color = 'red'
        elif 'minimally' in state_lower:
            label_background_color = 'green' 
        else:
             # Handle neutral state based on the flag for visibility
             if not show_neutral_labels:
                 continue # Skip adding label for neutral contacts if flag is False
             label_background_color = 'gray'

        # More informative label with magnitude
        label = f"{resname}{resnum}\n({state.capitalize()[:1]})\n|{mag:.2f}|" # Removed chain (ch)
        view.addLabel(label, {
            'position': {'x': pt[0], 'y': pt[1], 'z': pt[2]},
            'backgroundColor': label_background_color,
            'fontColor': 'white',
            'fontSize': 12,
            'backgroundOpacity': 0.6,
            'inFront': True
        })

    # label central residue
    central_desc = f"{central_resname_one}{central_res}" # Use 1-letter code, removed chain
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

def view_mutate_contacts_py3dmol(
    pdb: Pdb,
    chain: str,
    res_list: list = None,
    method: str = "threading",
    width: int = 800,
    height: int = 600,
    delta_threshold: float = 1.0,
    show_neutral_labels: bool = True,
    show_neutral_contacts: bool = True
) -> None:
    """
    Visualize the *change* in configurational contact frustration upon mutation,
    compared to the wild-type residue.

    Generates a separate viewer for each mutation variant at the specified positions.
    Contacts are colored based on the delta frustration (Mutant - WT):
    - Red: Delta < -delta_threshold (Frustration increased)
    - Blue: Delta > delta_threshold (Frustration decreased)
    - Gray: -delta_threshold <= Delta <= delta_threshold (Neutral change)

    Args:
        pdb: Pdb object with mutation data.
        chain: Chain identifier for the residue(s) of interest.
        res_list: List of residue numbers to visualize (defaults to all available).
        method: Mutation method key ("threading" or "modeller").
        width: Viewer width in pixels.
        height: Viewer height in pixels.
        delta_threshold: Absolute threshold for coloring contacts red/blue.
                         Contacts with |Delta Frst| > threshold are colored.
        show_neutral_labels: Whether to display labels for neutral delta frustration contacts (default: True).
        show_neutral_contacts: Whether to display cylinders for neutral delta frustration contacts (default: True).
    """
    import os
    import py3Dmol
    import pandas as pd
    import numpy as np
    import logging
    from .py3dmol_utils import build_ca_map, scale_radius, get_residue_name
    from Bio.SeqUtils import seq1

    logger = logging.getLogger(__name__)

    # --- 0. Load Base WT Frustration Data Once ---
    base_frust_file = os.path.join(pdb.job_dir, "FrustrationData", f"{pdb.pdb_base}.pdb_{pdb.mode}")
    if not os.path.exists(base_frust_file):
        raise FileNotFoundError(f"Base WT frustration file not found: {base_frust_file}")
    try:
        # Load the entire WT frustration file
        wt_base_df = pd.read_csv(base_frust_file, sep='\s+')
        # Basic validation of columns (adjust based on actual file format)
        required_wt_cols = {'Res1', 'Res2', 'ChainRes1', 'ChainRes2', 'FrstIndex'}
        if not required_wt_cols.issubset(wt_base_df.columns):
            raise ValueError(f"Base WT file {base_frust_file} missing required columns (needs: {required_wt_cols})")
    except Exception as e:
        logger.error(f"Error loading base WT frustration file {base_frust_file}: {e}")
        raise

    # --- 1. Input Validation and Data Loading --- 
    if not hasattr(pdb, 'Mutations'):
        raise ValueError("No mutation data found. Run mutate_res() first.")
    if method not in pdb.Mutations:
        raise ValueError(f"No mutation data found for method '{method}'.")
    if not pdb.Mutations[method]:
        raise ValueError(f"No mutation data found for method '{method}'.")

    # Find available mutation files for the specified chain
    available_mutations = {}
    for key, data in pdb.Mutations[method].items():
        if data.get('Chain') == chain:
            try:
                res_num = int(key.split('_')[1])
                mut_file = data.get('File')
                if mut_file and os.path.exists(mut_file):
                    available_mutations[res_num] = mut_file
                else:
                    logger.warning(f"Mutation file for {key} not found or invalid: {mut_file}")
            except (IndexError, ValueError):
                logger.warning(f"Skipping malformed mutation key: {key}")
                continue

    if not available_mutations:
        raise ValueError(f"No valid mutation files found for chain '{chain}', method '{method}'")

    # Determine which residues to process
    if res_list:
        process_residues = []
        for res in res_list:
            if res in available_mutations:
                process_residues.append(res)
            else:
                print(f"Warning: No mutation data file found for residue {res} in chain {chain}. Skipping.")
    else:
        process_residues = sorted(list(available_mutations.keys()))

    if not process_residues:
        print(f"No valid residues found to visualize.")
        return

    print(f"Preparing delta frustration visualizations for residues: {process_residues}")

    # Load Base PDB coordinates
    pdb_file = os.path.join(pdb.job_dir, 'VisualizationScripts', f"{pdb.pdb_base}.pdb")
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"Base PDB file not found: {pdb_file}")
    ca_map = build_ca_map(pdb_file)

    # --- 2. Process Each Residue Position --- 
    for res_pos in process_residues:
        print(f"\n--- Processing residue {res_pos}{chain} ---")
        mut_file = available_mutations[res_pos]

        # Load all mutation data for this position
        try:
            df = pd.read_csv(mut_file, sep='\s+')
            
            # Create new Central_AA and contact mapping, including cases where residue is in Res1 or Res2
            mask1 = (df['Res1'] == res_pos) & (df['ChainRes1'] == chain)
            mask2 = (df['Res2'] == res_pos) & (df['ChainRes2'] == chain)
            df = df[mask1 | mask2].copy()
            df['Central_AA'] = np.where(mask1, df['AA1'], df['AA2'])
            df['Contact_Res'] = np.where(mask1, df['Res2'], df['Res1']).astype(int)
            df['Contact_Chain'] = np.where(mask1, df['ChainRes2'], df['ChainRes1'])
            df['Contact_AA'] = np.where(mask1, df['AA2'], df['AA1'])
            
        except Exception as e:
            logger.error(f"Error loading data from {mut_file}: {e}")
            continue

        # Identify native residue three-letter code using helper function
        native_three = get_residue_name(pdb, chain, res_pos)
        if not native_three:
            logger.error(f"Could not determine native residue for {res_pos}{chain} from PDB object. Skipping.")
            continue

        # Convert three-letter code to one-letter code
        try:
            native_aa = seq1(native_three)
        except KeyError:
            logger.error(f"Could not convert native residue {native_three} to one-letter code. Skipping.")
            continue

        # Filter the BASE WT data for contacts involving the current res_pos and chain
        wt_contacts_df = wt_base_df[
            ((wt_base_df['Res1'] == res_pos) & (wt_base_df['ChainRes1'] == chain)) |
            ((wt_base_df['Res2'] == res_pos) & (wt_base_df['ChainRes2'] == chain))
        ].copy()

        if wt_contacts_df.empty:
            logger.warning(f"No WT contacts found for residue {res_pos}{chain} in the base frustration file. Skipping delta calculation for this residue.")
            continue

        # Create the WT frustration map { (contact_chain, contact_res_num): frustration_index }
        wt_frustration_map = {}
        for _, row in wt_contacts_df.iterrows():
            if row['Res1'] == res_pos and row['ChainRes1'] == chain:
                contact_key = (row['ChainRes2'], row['Res2'])
            else:
                contact_key = (row['ChainRes1'], row['Res1'])
            wt_frustration_map[contact_key] = row['FrstIndex']

        # --- 3. Process Each Mutation Variant --- 
        mutation_variants = sorted([aa for aa in df['Central_AA'].unique()]) # Keep WT in the list for now

        for mut_aa in mutation_variants:
            # Skip comparing WT to itself
            if mut_aa == native_aa:
                continue

            mut_data = df[df['Central_AA'] == mut_aa]
            if mut_data.empty:
                continue

            # Create a map for the specific mutant's frustration values
            mut_frustration_map = mut_data.set_index(['Contact_Chain', 'Contact_Res'])['FrstIndex'].to_dict()

            # Get the union of contact keys from WT and this mutant
            wt_contact_keys = set(wt_frustration_map.keys())
            mut_contact_keys = set(mut_frustration_map.keys())
            all_relevant_contact_keys = wt_contact_keys.union(mut_contact_keys)

            # Create a new viewer for this specific mutation comparison
            view = py3Dmol.view(width=width, height=height)
            with open(pdb_file) as f:
                view.addModel(f.read(), 'pdb')

            # Base style
            view.setStyle({'cartoon': {'color': 'lightgray', 'opacity': 0.7}})
            # Highlight the central mutated residue
            view.setStyle(
                {'chain': chain, 'resi': res_pos},
                {'cartoon': {'color': 'yellow'}, 'stick': {'radius': 0.3, 'colorscheme': 'yellowCarbon'}}
            )

            print(f"  Visualizing Delta: {native_aa} -> {mut_aa} ...")
            rendered_contacts = 0

            # Calculate and draw delta frustration contacts
            for contact_key in all_relevant_contact_keys:
                contact_chain, contact_res = contact_key
                mut_frst = mut_frustration_map.get(contact_key, 0) # Assume 0 if contact absent in MUTANT
                wt_frst = wt_frustration_map.get(contact_key, 0) # Assume 0 if contact absent in WT
                delta_frst = mut_frst - wt_frst

                # Determine state and color based on delta thresholds
                if delta_frst <= -delta_threshold:
                    state_str = 'highly'
                elif delta_frst >= delta_threshold:
                    state_str = 'minimally'
                else:
                    state_str = 'neutral'
                color_map = {'highly': 'red', 'minimally': 'green', 'neutral': 'gray'}
                color = color_map[state_str]

                # Determine radius based on magnitude of delta
                max_possible_delta = 8 # Assuming FrstIndex ranges roughly from -4 to 4
                radius = scale_radius(abs(delta_frst), 0, max_possible_delta, min_rad=0.05, max_rad=0.3)

                # Get coordinates
                start_coords = ca_map.get((chain, res_pos))
                end_coords = ca_map.get(contact_key)

                if not start_coords or not end_coords:
                    continue

                # Skip drawing cylinder if neutral and flag is False
                if state_str == 'neutral' and not show_neutral_contacts:
                    pass # Don't draw cylinder, but might still draw label later
                else:
                    # Draw cylinder
                    view.addCylinder({
                        'start': {'x': start_coords[0], 'y': start_coords[1], 'z': start_coords[2]},
                        'end':   {'x': end_coords[0], 'y': end_coords[1], 'z': end_coords[2]},
                        'radius': radius,
                        'color': color,
                        'dashed': True,
                        'fromCap': 1, # Rounded caps
                        'toCap': 1   # Rounded caps
                    })

                # Get contact residue name using PDB object
                try:
                    contact_aa_name = pdb.atom[
                        (pdb.atom['chain'] == contact_chain) &
                        (pdb.atom['res_num'] == contact_res) &
                        (pdb.atom['atom_name'] == 'CA')
                    ]['res_name'].iloc[0]
                except IndexError:
                    # Try to get from mut_data if available, otherwise use generic name
                    contact_aa_name = mut_data.loc[mut_data['Contact_Res'] == contact_res, 'Contact_AA'].iloc[0] if contact_key in mut_contact_keys else "UNK" 
                
                # Add label to contact residue (show delta) with state-based background
                # Determine label background color based on delta_frst
                if delta_frst <= -delta_threshold:
                    label_background_color = 'red'
                elif delta_frst >= delta_threshold:
                    label_background_color = 'green'
                else:
                    label_background_color = 'gray'
                    # Skip adding label if neutral and flag is False
                    if not show_neutral_labels:
                        continue # Skip to next contact in the loop

                # Adjust label opacity based on significance
                label_opacity = 0.7 if state_str != 'neutral' else 0.5

                label_text = f"{contact_aa_name}{contact_res}\nΔFrst: {delta_frst:+.2f}"
                view.addLabel(label_text, {
                    'position': {'x': end_coords[0], 'y': end_coords[1], 'z': end_coords[2]},
                    'backgroundColor': label_background_color,
                    'fontColor': 'white',
                    'fontSize': 10,
                    'backgroundOpacity': label_opacity,
                    'inFront': True
                })

                # Style contact residue based on DELTA frustration state (to match cylinders/labels)
                if delta_frst <= -delta_threshold:
                     contact_cartoon_color = 'red' # Or 'salmon' if preferred
                elif delta_frst >= delta_threshold:
                     contact_cartoon_color = 'green' # Or 'lightgreen'
                else:
                     contact_cartoon_color = 'gray' # Or 'lightblue'
                view.setStyle(
                    {'chain': contact_chain, 'resi': contact_res},
                    {'stick': {'radius': 0.1}, 'cartoon': {'color': contact_cartoon_color, 'opacity': 0.8}}
                )
                
                rendered_contacts += 1

            # Add Title to the viewer
            title = f"Delta Frustration: {native_aa} -> {mut_aa} at {res_pos}{chain}"
            center_coords = ca_map.get((chain, res_pos))
            if center_coords:
                view.addLabel(title, {
                    'position': {'x': center_coords[0], 'y': center_coords[1] + 15, 'z': center_coords[2]},
                    'backgroundColor': 'white',
                    'fontColor': 'black',
                    'fontSize': 16,
                    'backgroundOpacity': 0.8,
                    'inFront': True
                })
            
            # Add Legend
            legend_y_start = center_coords[1] - 15 if center_coords else 0
            legend_x_start = center_coords[0] - 15 if center_coords else 0
            view.addLabel("ΔFrst Legend:", {
                'position': {'x': legend_x_start, 'y': legend_y_start, 'z': center_coords[2] if center_coords else 0},
                'backgroundColor': 'white', 'fontColor': 'black', 'fontSize': 12, 'backgroundOpacity': 0.7, 'borderColor': 'lightgrey', 'borderWidth': 1
            })
            view.addLabel(f"  Increased (Δ < {-delta_threshold:.1f})", {
                'position': {'x': legend_x_start, 'y': legend_y_start - 2, 'z': center_coords[2] if center_coords else 0},
                'backgroundColor': 'red', 'fontColor': 'white', 'fontSize': 10, 'backgroundOpacity': 0.8
            })
            view.addLabel(f"  Decreased (Δ ≥ {delta_threshold:.1f})", {
                'position': {'x': legend_x_start, 'y': legend_y_start - 4, 'z': center_coords[2] if center_coords else 0},
                'backgroundColor': 'green', 'fontColor': 'white', 'fontSize': 10, 'backgroundOpacity': 0.8
            })
            view.addLabel("  Neutral", {
                'position': {'x': legend_x_start, 'y': legend_y_start - 6, 'z': center_coords[2] if center_coords else 0},
                'backgroundColor': 'gray', 'fontColor': 'white', 'fontSize': 10, 'backgroundOpacity': 0.8
            })


            # Zoom and Show
            zoom_selection = {'or': [{'chain': chain, 'resi': res_pos}]}
            for contact_key in all_relevant_contact_keys:
                zoom_selection['or'].append({'chain': contact_key[0], 'resi': contact_key[1]})
            
            if rendered_contacts > 0:
                print(f"    Rendered {rendered_contacts} contacts showing delta frustration.")
                view.zoomTo(zoom_selection)
                view.show()
            else:
                print(f"    No contacts rendered for {native_aa} -> {mut_aa}.")

        print(f"--- Finished residue {res_pos}{chain} --- " )

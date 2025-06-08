import streamlit as st
import pandas as pd
from pymatgen.core import Structure


def calculate_achievable_concentrations(target_concentrations, total_sites):
    achievable_concentrations = {}
    achievable_counts = {}
    remaining_sites = total_sites

    sorted_elements = sorted(target_concentrations.items(), key=lambda x: x[1], reverse=True)

    for i, (element, target_frac) in enumerate(sorted_elements):
        if i == len(sorted_elements) - 1:
            achievable_counts[element] = remaining_sites
        else:
            count = int(round(target_frac * total_sites))
            achievable_counts[element] = count
            remaining_sites -= count

    for element, count in achievable_counts.items():
        achievable_concentrations[element] = count / total_sites

    return achievable_concentrations, achievable_counts


def intro_text():
    st.warning("Please upload at least one structure file to use the SQS Transformation tool.")

    st.markdown("""

     This tool provides GUI for generation of special quasi random (SQS) structure using [Icet python package](https://icet.materialsmodeling.org/index.html).

     **Article for Icet: ÅNGQVIST, Mattias, et al. ICET–A Python library for constructing and sampling alloy cluster expansions. Advanced Theory and Simulations, 2019.**
     ### 🔄 Global Composition Mode
     - Specify overall composition for the entire structure
     - Elements can occupy any crystallographic site
     - Currently, only option with the specified supercell is integrated 
     - If the specified atomic concentrations cannot be achieved within the given supercell, they are automatically adjusted to the closest possible values compatible with that cell.

     ### 🎯 Sublattice-Specific Mode
     - Control which elements can occupy which atomic site
     - Set different compositions for different crystallographic sites

     ### Key Features:
     **🔬 ICET Integration**
     **🎯 Sublattice Control**
     **📊 (P)RDF Calculation**
     **💾 Download SQS**

     ### How to Use:

     1. Upload a structure file (.cif, .poscar, etc.)
     2. Choose between Global or Sublattice-Specific mode
     3. **For Sublattice Mode**: Configure you can configure composition for different atomic sites
     4. **For Global Mode**: Set overall target composition
     5. Select ICET algorithm and parameters
     6. Generate and download your SQS structure
     """)


import streamlit.components.v1 as components
import numpy as np
import py3Dmol
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write


def structure_preview(working_structure):
    #st.subheader("Structure Preview")

    lattice = working_structure.lattice
    st.write(f"**Lattice parameters:**")
    st.write(f"a = {lattice.a:.4f} Å, b = {lattice.b:.4f} Å, c = {lattice.c:.4f} Å")
    st.write(f"α = {lattice.alpha:.2f}°, β = {lattice.beta:.2f}°, γ = {lattice.gamma:.2f}°")

    st.write("**Structure visualization:**")

    try:
        from io import StringIO

        jmol_colors = {
            'H': '#FFFFFF', 'He': '#D9FFFF', 'Li': '#CC80FF', 'Be': '#C2FF00', 'B': '#FFB5B5',
            'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D', 'F': '#90E050', 'Ne': '#B3E3F5',
            'Na': '#AB5CF2', 'Mg': '#8AFF00', 'Al': '#BFA6A6', 'Si': '#F0C8A0', 'P': '#FF8000',
            'S': '#FFFF30', 'Cl': '#1FF01F', 'Ar': '#80D1E3', 'K': '#8F40D4', 'Ca': '#3DFF00',
            'Fe': '#E06633', 'Co': '#F090A0', 'Ni': '#50D050', 'Cu': '#C88033', 'Zn': '#7D80B0'
        }

        def add_box(view, cell, color='black', linewidth=2):
            vertices = np.array([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
            ])
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]
            ]
            cart_vertices = np.dot(vertices, cell)
            for edge in edges:
                start, end = cart_vertices[edge[0]], cart_vertices[edge[1]]
                view.addCylinder({
                    'start': {'x': start[0], 'y': start[1], 'z': start[2]},
                    'end': {'x': end[0], 'y': end[1], 'z': end[2]},
                    'radius': 0.05,
                    'color': color
                })

        structure_ase = AseAtomsAdaptor.get_atoms(working_structure)
        xyz_io = StringIO()
        write(xyz_io, structure_ase, format="xyz")
        xyz_str = xyz_io.getvalue()

        view = py3Dmol.view(width=400, height=400)
        view.addModel(xyz_str, "xyz")
        view.setStyle({'model': 0}, {"sphere": {"radius": 0.4, "colorscheme": "Jmol"}})

        cell = structure_ase.get_cell()
        add_box(view, cell, color='black', linewidth=2)

        view.zoomTo()
        view.zoom(1.2)

        html_string = view._make_html()
        components.html(html_string, height=420, width=420)

        unique_elements = sorted(set(structure_ase.get_chemical_symbols()))
        legend_html = "<div style='display: flex; flex-wrap: wrap; align-items: center; justify-content: center; margin-top: 10px;'>"
        for elem in unique_elements:
            color = jmol_colors.get(elem, "#CCCCCC")
            legend_html += (
                f"<div style='margin-right: 15px; display: flex; align-items: center;'>"
                f"<div style='width: 20px; height: 20px; background-color: {color}; margin-right: 5px; border: 1px solid black; border-radius: 50%;'></div>"
                f"<span style='font-weight: bold;'>{elem}</span></div>"
            )
        legend_html += "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error visualizing structure: {e}")
        st.info("3D visualization is not available, but you can still generate the SQS structure.")


def sqs_visualization(result):
    try:
        from io import StringIO

        jmol_colors = {
            'H': '#FFFFFF', 'He': '#D9FFFF', 'Li': '#CC80FF', 'Be': '#C2FF00', 'B': '#FFB5B5',
            'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D', 'F': '#90E050', 'Ne': '#B3E3F5',
            'Na': '#AB5CF2', 'Mg': '#8AFF00', 'Al': '#BFA6A6', 'Si': '#F0C8A0', 'P': '#FF8000',
            'S': '#FFFF30', 'Cl': '#1FF01F', 'Ar': '#80D1E3', 'K': '#8F40D4', 'Ca': '#3DFF00',
            'Sc': '#E6E6E6', 'Ti': '#BFC2C7', 'V': '#A6A6AB', 'Cr': '#8A99C7', 'Mn': '#9C7AC7',
            'Fe': '#E06633', 'Co': '#F090A0', 'Ni': '#50D050', 'Cu': '#C88033', 'Zn': '#7D80B0',
            'Ga': '#C28F8F', 'Ge': '#668F8F', 'As': '#BD80E3', 'Se': '#FFA100', 'Br': '#A62929',
            'Kr': '#5CB8D1', 'Rb': '#702EB0', 'Sr': '#00FF00', 'Y': '#94FFFF', 'Zr': '#94E0E0',
            'Nb': '#73C2C9', 'Mo': '#54B5B5', 'Tc': '#3B9E9E', 'Ru': '#248F8F', 'Rh': '#0A7D8C',
            'Pd': '#006985', 'Ag': '#C0C0C0', 'Cd': '#FFD98F', 'In': '#A67573', 'Sn': '#668080',
            'Sb': '#9E63B5', 'Te': '#D47A00', 'I': '#940094', 'Xe': '#429EB0', 'Cs': '#57178F',
            'Ba': '#00C900', 'La': '#70D4FF', 'Ce': '#FFFFC7', 'Pr': '#D9FFC7', 'Nd': '#C7FFC7',
            'Pm': '#A3FFC7', 'Sm': '#8FFFC7', 'Eu': '#61FFC7', 'Gd': '#45FFC7', 'Tb': '#30FFC7',
            'Dy': '#1FFFC7', 'Ho': '#00FF9C', 'Er': '#00E675', 'Tm': '#00D452', 'Yb': '#00BF38',
            'Lu': '#00AB24', 'Hf': '#4DC2FF', 'Ta': '#4DA6FF', 'W': '#2194D6', 'Re': '#267DAB',
            'Os': '#266696', 'Ir': '#175487', 'Pt': '#D0D0E0', 'Au': '#FFD123', 'Hg': '#B8B8D0',
            'Tl': '#A6544D', 'Pb': '#575961', 'Bi': '#9E4FB5', 'Po': '#AB5C00', 'At': '#754F45',
            'Rn': '#428296', 'Fr': '#420066', 'Ra': '#007D00', 'Ac': '#70ABFA', 'Th': '#00BAFF',
            'Pa': '#00A1FF', 'U': '#008FFF', 'Np': '#0080FF', 'Pu': '#006BFF', 'Am': '#545CF2',
            'Cm': '#785CE3', 'Bk': '#8A4FE3', 'Cf': '#A136D4', 'Es': '#B31FD4', 'Fm': '#B31FBA',
            'Md': '#B30DA6', 'No': '#BD0D87', 'Lr': '#C70066', 'Rf': '#CC0059', 'Db': '#D1004F',
            'Sg': '#D90045', 'Bh': '#E00038', 'Hs': '#E6002E', 'Mt': '#EB0026'
        }

        def add_box(view, cell, color='black', linewidth=2):
            vertices = np.array([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
            ])
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]
            ]
            cart_vertices = np.dot(vertices, cell)
            for edge in edges:
                start, end = cart_vertices[edge[0]], cart_vertices[edge[1]]
                view.addCylinder({
                    'start': {'x': start[0], 'y': start[1], 'z': start[2]},
                    'end': {'x': end[0], 'y': end[1], 'z': end[2]},
                    'radius': 0.05,
                    'color': color
                })

        structure_ase = AseAtomsAdaptor.get_atoms(result['structure'])
        xyz_io = StringIO()
        write(xyz_io, structure_ase, format="xyz")
        xyz_str = xyz_io.getvalue()

        view = py3Dmol.view(width=600, height=400)
        view.addModel(xyz_str, "xyz")
        view.setStyle({'model': 0}, {"sphere": {"radius": 0.3, "colorscheme": "Jmol"}})

        cell = structure_ase.get_cell()
        add_box(view, cell, color='black', linewidth=2)

        view.zoomTo()
        view.zoom(1.2)

        html_string = view._make_html()
        components.html(html_string, height=420, width=620)

        unique_elements = sorted(set(structure_ase.get_chemical_symbols()))
        legend_html = "<div style='display: flex; flex-wrap: wrap; align-items: center; justify-content: center; margin-top: 10px;'>"
        for elem in unique_elements:
            color = jmol_colors.get(elem, "#CCCCCC")
            legend_html += (
                f"<div style='margin-right: 15px; display: flex; align-items: center;'>"
                f"<div style='width: 20px; height: 20px; background-color: {color}; margin-right: 5px; border: 1px solid black; border-radius: 50%;'></div>"
                f"<span style='font-weight: bold;'>{elem}</span></div>"
            )
        legend_html += "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error visualizing SQS structure: {e}")


def generated_SQS_information(result):
    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.write("**Final Composition:**")
        comp = result['structure'].composition
        comp_data = []

        target_concentrations = result['target_concentrations']
        achievable_concentrations = result['achievable_concentrations']

        if isinstance(target_concentrations, dict) and any(isinstance(v, dict) for v in target_concentrations.values()):
            global_target = {}
            global_achievable = {}

            for sublattice_id, sublattice_conc in target_concentrations.items():
                for element, conc in sublattice_conc.items():
                    if element in global_target:
                        global_target[element] += conc
                    else:
                        global_target[element] = conc

            for sublattice_id, sublattice_conc in achievable_concentrations.items():
                for element, conc in sublattice_conc.items():
                    if element in global_achievable:
                        global_achievable[element] += conc
                    else:
                        global_achievable[element] = conc

            num_sublattices = len(target_concentrations)
            for element in global_target:
                global_target[element] /= num_sublattices
            for element in global_achievable:
                global_achievable[element] /= num_sublattices

        else:
            global_target = target_concentrations
            global_achievable = achievable_concentrations

        for el, amt in comp.items():
            target_frac = global_target.get(el.symbol, 0.0)
            achievable_frac = global_achievable.get(el.symbol, 0.0)
            actual_frac = amt / comp.num_atoms

            comp_data.append({
                "Element": el.symbol,
                "Count": int(amt),
                "Actual": f"{actual_frac:.4f}",
                "Target": f"{target_frac:.4f}",
                "Match": "✅" if abs(actual_frac - target_frac) < 0.01 else "⚠️"
            })
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True)

    with col_info2:
        st.write("**Lattice Parameters:**")
        sqs_lattice = result['structure'].lattice
        st.write(f"a = {sqs_lattice.a:.4f} Å")
        st.write(f"b = {sqs_lattice.b:.4f} Å")
        st.write(f"c = {sqs_lattice.c:.4f} Å")
        st.write(f"α = {sqs_lattice.alpha:.2f}°")
        st.write(f"β = {sqs_lattice.beta:.2f}°")
        st.write(f"γ = {sqs_lattice.gamma:.2f}°")
        st.write(f"Volume = {sqs_lattice.volume:.2f} Ų")

    if isinstance(target_concentrations, dict) and any(isinstance(v, dict) for v in target_concentrations.values()):
        with st.expander("🎯 Sublattice-Specific Composition Details", expanded=False):
            st.write("**Sublattice Breakdown:**")
            sublattice_data = []

            for sublattice_id in sorted(target_concentrations.keys()):
                target_sub = target_concentrations[sublattice_id]
                achievable_sub = achievable_concentrations.get(sublattice_id, {})

                for element in target_sub:
                    target_val = target_sub[element]
                    achievable_val = achievable_sub.get(element, 0.0)
                    status = "✅" if abs(target_val - achievable_val) < 0.01 else "⚠️"

                    sublattice_data.append({
                        "Sublattice": sublattice_id,
                        "Element": element,
                        "Target": f"{target_val:.4f}",
                        "Achievable": f"{achievable_val:.4f}",
                        "Status": status
                    })

            if sublattice_data:
                sublattice_df = pd.DataFrame(sublattice_data)
                st.dataframe(sublattice_df, use_container_width=True)


def icet_results_short_sum(result):
    st.success(
        f"✅ SQS structure generated successfully in {result['elapsed_time']:.1f} seconds!. Algorithm used: {result['algorithm']}"
        f"Method: {result['method']}. Structure contains {len(result['structure'])} atoms.")

    if result.get('progress_data') and result['progress_data']['scores']:
        st.subheader("SQS Generation Summary")
        progress_data = result['progress_data']
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Total Steps", 1000*len(progress_data['steps']))
        with col_stat2:
            if progress_data['scores']:
                st.metric("Best Score", f"{min(progress_data['scores']):.4f}")


import logging
import threading
import queue
import re
from icet.input_output.logging_tools import set_log_config


class StreamlitLogHandler(logging.Handler):

    def __init__(self, message_queue):
        super().__init__()
        self.message_queue = message_queue

    def emit(self, record):
        try:
            msg = self.format(record)
            self.message_queue.put(msg)
        except Exception:
            pass

def parse_icet_log_message(message):
    pattern = r'MC step (\d+)/(\d+) \((\d+) accepted trials, temperature ([\d.-]+)\), best score: ([\d.-]+)'
    match = re.search(pattern, message)

    if match:
        return {
            'current_step': int(match.group(1)),
            'total_steps': int(match.group(2)),
            'accepted_trials': int(match.group(3)),
            'temperature': float(match.group(4)),
            'best_score': float(match.group(5)),
            'message': message
        }
    return None


from icet.input_output.logging_tools import set_log_config


def setup_icet_logging(message_queue):
    set_log_config(level='INFO')
    icet_logger = logging.getLogger('icet.target_cluster_vector_annealing')
    handler = StreamlitLogHandler(message_queue)
    handler.setLevel(logging.INFO)
    icet_logger.addHandler(handler)
    return handler


def has_partial_occupancies(structure):
    for site in structure:
        if not site.is_ordered:
            return True
    return False


def get_sublattice_composition_options():
    common_elements = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
        'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
        'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Vac'
    ]
    return common_elements


import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def remove_vacancies_from_structure(structure):
    non_vacancy_indices = []
    for i, site in enumerate(structure):
        if site.is_ordered:
            if site.specie.symbol != 'X':
                non_vacancy_indices.append(i)
        else:
            has_non_vacancy = any(sp.symbol != 'X' for sp in site.species)
            if has_non_vacancy:
                non_vacancy_indices.append(i)

    if len(non_vacancy_indices) == len(structure):
        return structure

    new_lattice = structure.lattice
    new_species = []
    new_coords = []

    for i in non_vacancy_indices:
        site = structure[i]
        new_coords.append(site.frac_coords)
        if site.is_ordered:
            new_species.append(site.specie)
        else:

            filtered_species = {sp: occ for sp, occ in site.species.items() if sp.symbol != 'X'}

            total_occ = sum(filtered_species.values())
            if total_occ > 0:
                normalized_species = {sp: occ / total_occ for sp, occ in filtered_species.items()}
                new_species.append(normalized_species)
            else:

                continue

    new_structure = Structure(new_lattice, new_species, new_coords)
    return new_structure


def render_site_sublattice_selector(working_structure, all_sites):
    st.markdown(
        """
        <hr style="border: none; height: 6px; background-color: #3399ff; border-radius: 8px; margin: 20px 0;">
        """,
        unsafe_allow_html=True
    )
    st.subheader("4️⃣ Step 4: Create Sublattices - Select Elements and Concentrations")
    st.info(
        "Select atomic sites and specify elements with concentrations. Sites with identical compositions will form sublattices.")

    if "site_assignments" not in st.session_state:
        st.session_state.site_assignments = {}

    with st.expander("📋 All Atomic Sites", expanded=False):
        site_data = []
        for site_info in all_sites:
            site_data.append({
                "Site Index": site_info['site_index'],
                "Current Element": site_info['element'],
                "Wyckoff Letter": site_info['wyckoff_letter'],
                "Coordinates": f"({site_info['coords'][0]:.3f}, {site_info['coords'][1]:.3f}, {site_info['coords'][2]:.3f})"
            })
        site_df = pd.DataFrame(site_data)
        st.dataframe(site_df, use_container_width=True)

    assigned_sites = set()
    for assignment_key in st.session_state.site_assignments.keys():
        assigned_sites.update(assignment_key)

    unassigned_sites = [site['site_index'] for site in all_sites if site['site_index'] not in assigned_sites]

    st.write(
        "**Assign Elements to Sites. Then confirm it with the 'Set elements' button. After that, you can assign another sites differently if needed:**")

    if unassigned_sites:
        site_options = unassigned_sites
        site_labels = [f"Site {site['site_index']}: {site['element']} (Wyckoff {site['wyckoff_letter']})"
                       for site in all_sites if site['site_index'] in unassigned_sites]

        selected_sites = st.multiselect(
            f"Select from {len(unassigned_sites)} unassigned sites:",
            options=site_options,
            format_func=lambda x: site_labels[unassigned_sites.index(x)],
            key="selected_sites"
        )

        if selected_sites:
            st.write(f"**Configure elements for sites: {selected_sites}**")

            common_elements = [
                'H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F',
                'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Ga', 'Ge', 'As', 'Se', 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
                'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
                'Cs', 'Ba', 'La', 'Ce', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
                'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'U'
            ]
            all_elements = ['X'] + sorted(common_elements)

            col1, col2 = st.columns([2, 1])
            with col1:
                selected_elements = st.multiselect(
                    "Select elements (use 'X' for vacancy):",
                    options=all_elements,
                    default=[],
                    key="elements_multiselect",
                    help="Example: Select 'Fe' and 'Ga' or 'Ti' and 'Al'"
                )

            with col2:
                if st.button("Set Elements", key="set_elements_btn", type='secondary'):
                    if selected_elements:

                        elements_list = sorted(selected_elements)

                        assignment_key = tuple(sorted(selected_sites))
                        st.session_state.site_assignments[assignment_key] = {
                            'elements': elements_list,
                            'concentrations': {}
                        }
                        st.success(
                            f"Elements {elements_list} assigned to sites {selected_sites} (sorted alphabetically)")

                        st.rerun()
                    else:
                        st.warning("Please select at least one element.")
    else:
        st.info("✅ All sites have been assigned. Configure concentrations below or remove assignments to modify.")

    assignments_to_remove = []
    for assignment_key, assignment_data in st.session_state.site_assignments.items():
        sites_list = list(assignment_key)
        elements = assignment_data['elements']

        with st.expander(f"Sites {sites_list}: {', '.join(elements)}", expanded=True):

            if len(elements) == 1:
                assignment_data['concentrations'] = {elements[0]: 1.0}
                st.write(f"**{elements[0]}: 1.000** (single element)")
            else:
                st.write("**Set concentrations:**")

                if not assignment_data['concentrations'] or set(assignment_data['concentrations'].keys()) != set(
                        elements):
                    equal_conc = 1.0 / len(elements)
                    assignment_data['concentrations'] = {elem: equal_conc for elem in elements}

                remaining = 1.0
                new_concentrations = {}

                for i, elem in enumerate(elements[:-1]):
                    current_val = assignment_data['concentrations'].get(elem, 0.0)
                    conc_val = st.slider(
                        f"{elem}:",
                        min_value=0.0,
                        max_value=remaining,
                        value=min(current_val, remaining),
                        step=0.01,
                        format="%.3f",
                        key=f"conc_{assignment_key}_{elem}"
                    )
                    new_concentrations[elem] = conc_val
                    remaining -= conc_val

                last_elem = elements[-1]
                new_concentrations[last_elem] = max(0.0, remaining)
                st.write(f"**{last_elem}: {remaining:.3f}**")

                assignment_data['concentrations'] = new_concentrations


            if st.button(f"❌ Remove Assignment", key=f"remove_{assignment_key}"):
                assignments_to_remove.append(assignment_key)

    for key in assignments_to_remove:
        del st.session_state.site_assignments[key]
        st.rerun()
    if st.session_state.site_assignments:
        st.write("**Current Assignments Summary:**")
        summary_data = []
        for assignment_key, assignment_data in st.session_state.site_assignments.items():
            sites_str = ", ".join(map(str, assignment_key))
            elements_str = ", ".join(assignment_data['elements'])
            conc_str = ", ".join([f"{elem}: {conc:.3f}" for elem, conc in assignment_data['concentrations'].items()])

            summary_data.append({
                "Sites": sites_str,
                "Elements": elements_str,
                "Concentrations": conc_str
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

    final_assigned_sites = set()
    for assignment_key in st.session_state.site_assignments.keys():
        final_assigned_sites.update(assignment_key)

    final_unassigned_sites = [site['site_index'] for site in all_sites if
                              site['site_index'] not in final_assigned_sites]

    if final_unassigned_sites:
        st.info(f"**Unassigned sites** (keeping original elements): {final_unassigned_sites}")

    chemical_symbols = []
    sublattice_compositions = {}
    first_occurrence_order = []

    for site in all_sites:
        site_idx = site['site_index']

        site_elements = None
        site_concentrations = None

        for assignment_key, assignment_data in st.session_state.site_assignments.items():
            if site_idx in assignment_key:
                site_elements = assignment_data['elements']
                site_concentrations = assignment_data['concentrations']
                break

        if site_elements is None:
            original_element = site['element']
            chemical_symbols.append([original_element])
        else:
            # Assigned site - use specified elements
            # Filter out 'X' (vacancy) for chemical symbols and SORT ALPHABETICALLY
            # valid_elements = [elem for elem in site_elements if elem != 'X']
            # if not valid_elements:
            #    valid_elements = ['H']  # Dummy for vacancy-only
            # valid_elements = sorted(valid_elements)  # SORT ALPHABETICALLY TO MATCH ICET
            valid_elements = sorted(site_elements)
            chemical_symbols.append(valid_elements)

            if len(site_elements) > 1:  # Only multi-element sites form sublattices
                # SORT ELEMENTS ALPHABETICALLY TO MATCH ICET
                sorted_site_elements = sorted(site_elements)
                elements_signature = frozenset(sorted_site_elements)

                # Check if this is the first occurrence of this element combination
                if elements_signature not in sublattice_compositions:
                    sublattice_compositions[elements_signature] = site_concentrations.copy()
                    first_occurrence_order.append(elements_signature)

    # Create sublattice labels based on ALPHABETICAL ORDER OF FIRST ELEMENT (matching ICET's logic)
    target_concentrations = {}
    sublattice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    sorted_sublattices = []
    for elements_signature in first_occurrence_order:
        elements_list = sorted(list(elements_signature))  # Get sorted elements
        first_element = elements_list[0]  # First element alphabetically
        sorted_sublattices.append((first_element, elements_signature))

    # Sort by first element alphabetically
    sorted_sublattices.sort(key=lambda x: x[0])

    # Assign sublattice letters in alphabetical order of first element
    for i, (first_element, elements_signature) in enumerate(sorted_sublattices):
        if i < len(sublattice_letters):
            sublattice_id = sublattice_letters[i]
            target_concentrations[sublattice_id] = sublattice_compositions[elements_signature]

    is_configured = True

    for assignment_key, assignment_data in st.session_state.site_assignments.items():
        total_conc = sum(assignment_data['concentrations'].values())
        if abs(total_conc - 1.0) > 0.001:
            is_configured = False
            sites_str = ", ".join(map(str, assignment_key))
            st.warning(f"⚠️ Concentrations for sites {sites_str} must sum to 1.0 (currently {total_conc:.3f})")

    if is_configured and target_concentrations:
        st.success("✅ Site assignment configuration is complete!")

        with st.expander("🎯 Generated Configuration", expanded=False):
            st.write("**Chemical Symbols (for ICET ClusterSpace):**")
            st.code(f"chemical_symbols = {chemical_symbols}")

            st.write("**Target Concentrations (for SQS generation):**")
            st.code(f"target_concentrations = {target_concentrations}")

            if target_concentrations:
                st.write("**Sublattice Summary (ordered by first element alphabetically to match ICET):**")
                for sublattice_id, conc in target_concentrations.items():
                    elements = sorted(list(conc.keys()))  # Show sorted elements
                    first_element = elements[0]
                    st.write(f"- **Sublattice {sublattice_id}**: {elements} (first element: '{first_element}')")

                st.info(
                    "💡 **Note**: Sublattices are assigned A, B, C... based on alphabetical order of their first element to match ICET's behavior.")

    elif not st.session_state.site_assignments:
        is_configured = False
        st.info("⚙️ Please assign elements to at least one atomic site.")

    return chemical_symbols, target_concentrations, is_configured


import streamlit as st
import pandas as pd
import numpy as np
from ase.build import make_supercell
import icet
import logging
import threading
import queue
import time
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from icet.tools.structure_generation import (
    generate_sqs,
    generate_sqs_from_supercells,
    generate_sqs_by_enumeration
)
from ase import Atoms


def pymatgen_to_ase(structure):
    symbols = [site.specie.symbol if site.is_ordered else site.species.elements[0].symbol for site in structure]
    positions = structure.cart_coords
    cell = structure.lattice.matrix
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=[True, True, True])


def calculate_supercell_factor(transformation_matrix):

    is_diagonal = True
    for i in range(3):
        for j in range(3):
            if i != j and abs(transformation_matrix[i][j]) > 1e-10:
                is_diagonal = False
                break
        if not is_diagonal:
            break

    if is_diagonal:
        return int(round(transformation_matrix[0][0] * transformation_matrix[1][1] * transformation_matrix[2][2]))
    else:
        return int(round(abs(np.linalg.det(transformation_matrix))))


def calculate_achievable_concentrations_sublattice(target_concentrations, chemical_symbols, transformation_matrix,
                                                   primitive_structure):
    supercell_factor = calculate_supercell_factor(transformation_matrix)
    achievable_concentrations = {}
    adjustment_info = []

    sublattice_mapping = {}  # {sublattice_letter: {'elements': set, 'site_indices': list}}
    sublattice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    unique_combinations = {}  # {frozenset(elements): [site_indices]}

    for site_idx, site_elements in enumerate(chemical_symbols):
        if len(site_elements) > 1:
            sorted_elements = sorted(site_elements)
            elements_signature = frozenset(sorted_elements)

            if elements_signature not in unique_combinations:
                unique_combinations[elements_signature] = []
            unique_combinations[elements_signature].append(site_idx)

    sorted_combinations = []
    for elements_signature, site_indices in unique_combinations.items():
        elements_list = sorted(list(elements_signature))
        first_element = elements_list[0]
        sorted_combinations.append((first_element, elements_signature, site_indices))

    sorted_combinations.sort(key=lambda x: x[0])

    for i, (first_element, elements_signature, site_indices) in enumerate(sorted_combinations):
        if i < len(sublattice_letters):
            sublattice_letter = sublattice_letters[i]
            sublattice_mapping[sublattice_letter] = {
                'elements': set(elements_signature),
                'site_indices': site_indices
            }

    for sublattice_letter, target_conc in target_concentrations.items():
        if sublattice_letter not in sublattice_mapping:
            st.warning(f"Sublattice {sublattice_letter} not found in chemical symbols")
            achievable_concentrations[sublattice_letter] = {}
            continue

        mapping_info = sublattice_mapping[sublattice_letter]
        sites_in_primitive = len(mapping_info['site_indices'])
        total_sites_in_supercell = sites_in_primitive * supercell_factor

        if total_sites_in_supercell == 0:
            st.warning(f"No sites found for sublattice {sublattice_letter}")
            achievable_concentrations[sublattice_letter] = {}
            continue

        sublattice_achievable = {}
        sublattice_counts = {}

        elements = list(target_conc.keys())
        exact_counts = {}

        for element in elements:
            exact_counts[element] = target_conc[element] * total_sites_in_supercell

        integer_counts = {}
        remainders = {}
        total_assigned = 0

        for element in elements:
            integer_counts[element] = int(exact_counts[element])
            remainders[element] = exact_counts[element] - integer_counts[element]
            total_assigned += integer_counts[element]

        remaining_atoms = total_sites_in_supercell - total_assigned
        if remaining_atoms > 0:
            sorted_by_remainder = sorted(remainders.items(), key=lambda x: x[1], reverse=True)

            for i in range(remaining_atoms):
                element = sorted_by_remainder[i % len(sorted_by_remainder)][0]
                integer_counts[element] += 1
        elif remaining_atoms < 0:
            sorted_by_remainder = sorted(remainders.items(), key=lambda x: x[1])

            for i in range(abs(remaining_atoms)):
                element = sorted_by_remainder[i % len(sorted_by_remainder)][0]
                if integer_counts[element] > 0:
                    integer_counts[element] -= 1

        sublattice_counts = integer_counts

        total_check = sum(sublattice_counts.values())
        if total_check != total_sites_in_supercell:
            st.error(
                f"Atom count mismatch in sublattice {sublattice_letter}: {total_check} != {total_sites_in_supercell}")

        for element, count in sublattice_counts.items():
            sublattice_achievable[element] = count / total_sites_in_supercell

        achievable_concentrations[sublattice_letter] = sublattice_achievable

        for element in target_conc:
            target_val = target_conc[element]
            achievable_val = sublattice_achievable.get(element, 0.0)
            if abs(target_val - achievable_val) > 0.001:
                adjustment_info.append({
                    "Sublattice": sublattice_letter,
                    "Element": element,
                    "Target": f"{target_val:.3f}",
                    "Achievable": f"{achievable_val:.3f}",
                    "Atom Count": sublattice_counts.get(element, 0),
                    "Total Sites": total_sites_in_supercell
                })

    return achievable_concentrations, adjustment_info


def calculate_global_concentrations_from_sublattices(target_concentrations, chemical_symbols, transformation_matrix,
                                                     primitive_structure):
    atoms = pymatgen_to_ase(primitive_structure)
    supercell_factor = calculate_supercell_factor(transformation_matrix)

    global_element_counts = {}

    achievable_concentrations, _ = calculate_achievable_concentrations_sublattice(
        target_concentrations, chemical_symbols, transformation_matrix, primitive_structure
    )

    for site_idx, site_elements in enumerate(chemical_symbols):
        if len(site_elements) == 1:
            element = site_elements[0]
            sites_in_supercell = supercell_factor
            if element in global_element_counts:
                global_element_counts[element] += sites_in_supercell
            else:
                global_element_counts[element] = sites_in_supercell
        else:
            sorted_elements = sorted(site_elements)
            elements_signature = frozenset(sorted_elements)

            found_sublattice = None
            for sublattice_letter, achievable_conc in achievable_concentrations.items():
                if set(achievable_conc.keys()) == set(sorted_elements):
                    found_sublattice = sublattice_letter
                    break

            if found_sublattice:
                sites_in_supercell = supercell_factor
                for element, concentration in achievable_concentrations[found_sublattice].items():
                    # Use the EXACT atom count from achievable concentrations
                    element_count = concentration * sites_in_supercell
                    if element in global_element_counts:
                        global_element_counts[element] += element_count
                    else:
                        global_element_counts[element] = element_count

    total_sites = len(atoms) * supercell_factor
    global_concentrations = {}

    for element, count in global_element_counts.items():
        global_concentrations[element] = count / total_sites

    return global_concentrations


def display_sublattice_preview(target_concentrations, chemical_symbols, transformation_matrix, primitive_structure):
    try:
        if not target_concentrations:
            st.info("No sublattice concentrations configured yet.")
            return

        achievable_concentrations, adjustment_info = calculate_achievable_concentrations_sublattice(
            target_concentrations, chemical_symbols, transformation_matrix, primitive_structure
        )

        st.write("**Sublattice Concentrations:**")

        sublattice_data = []
        supercell_factor = int(transformation_matrix[0][0]) * int(transformation_matrix[1][1]) * int(
            transformation_matrix[2][2])

        sublattice_mapping = {}  # {sublattice_letter: {'elements': set, 'site_indices': list}}
        sublattice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

        unique_combinations = {}  # {frozenset(elements): [site_indices]}

        for site_idx, site_elements in enumerate(chemical_symbols):
            if len(site_elements) > 1:  # Multi-element site
                sorted_elements = sorted(site_elements)
                elements_signature = frozenset(sorted_elements)

                if elements_signature not in unique_combinations:
                    unique_combinations[elements_signature] = []
                unique_combinations[elements_signature].append(site_idx)
        sorted_combinations = []
        for elements_signature, site_indices in unique_combinations.items():
            elements_list = sorted(list(elements_signature))
            first_element = elements_list[0]
            sorted_combinations.append((first_element, elements_signature, site_indices))

        sorted_combinations.sort(key=lambda x: x[0])  # Sort by first element

        for i, (first_element, elements_signature, site_indices) in enumerate(sorted_combinations):
            if i < len(sublattice_letters):
                sublattice_letter = sublattice_letters[i]
                sublattice_mapping[sublattice_letter] = {
                    'elements': set(elements_signature),
                    'site_indices': site_indices
                }

        for sublattice_letter, target_conc in target_concentrations.items():
            achievable_conc = achievable_concentrations.get(sublattice_letter, {})
            if sublattice_letter in sublattice_mapping:
                sites_count = len(sublattice_mapping[sublattice_letter]['site_indices'])
                total_sublattice_sites = sites_count * supercell_factor

                for element in target_conc.keys():
                    target_frac = target_conc[element]
                    achievable_frac = achievable_conc.get(element, 0.0)
                    atom_count = int(round(achievable_frac * total_sublattice_sites))

                    status = "✅" if abs(target_frac - achievable_frac) < 0.01 else "⚠️"

                    sublattice_data.append({
                        "Sublattice": sublattice_letter,
                        "Element": element,
                        "Target": f"{target_frac:.3f}",
                        "Achievable": f"{achievable_frac:.3f}",
                        "Atoms": atom_count,
                        "Sites": total_sublattice_sites,
                        "Status": status
                    })

        if sublattice_data:
            sublattice_df = pd.DataFrame(sublattice_data)
            st.dataframe(sublattice_df, use_container_width=True)

        global_concentrations = calculate_global_concentrations_from_sublattices(
            target_concentrations, chemical_symbols, transformation_matrix, primitive_structure
        )

        if global_concentrations:
            st.write("**Overall Composition from Sublattices:**")

            atoms = pymatgen_to_ase(primitive_structure)
            total_sites = len(atoms) * supercell_factor

            overall_comp_data = []
            total_global_atoms = 0

            for element in sorted(global_concentrations.keys()):
                global_fraction = global_concentrations[element]
                atom_count = int(round(global_fraction * total_sites))
                total_global_atoms += atom_count

                overall_comp_data.append({
                    "Element": element,
                    "Fraction": f"{global_fraction:.3f}",
                    "Percentage": f"{global_fraction * 100:.1f}%",
                    "Atom Count": atom_count
                })

            if overall_comp_data:
                overall_comp_df = pd.DataFrame(overall_comp_data)
                st.dataframe(overall_comp_df, use_container_width=True)

                st.info(f"**Total atoms in supercell:** {total_global_atoms} / {total_sites}")
        else:
            st.warning("Could not calculate overall composition.")

        # Show adjustment information if any
        #if adjustment_info:
        #    st.warning("⚠️ **Concentration Adjustments Required:**")
        #    adj_df = pd.DataFrame(adjustment_info)
        #    st.dataframe(adj_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error calculating composition preview: {e}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")


class ProgressTracker:

    def __init__(self):
        self.data = {
            'steps': [],
            'scores': [],
            'temperatures': [],
            'accepted_trials': [],
            'timestamps': []
        }
        self.lock = threading.Lock()
        self.last_update = 0

    def add_data_point(self, step, score, temperature, accepted_trials):

        with self.lock:
            self.data['steps'].append(step)
            self.data['scores'].append(score)
            self.data['temperatures'].append(temperature)
            self.data['accepted_trials'].append(accepted_trials)
            self.data['timestamps'].append(time.time())

    def get_data_copy(self):

        with self.lock:
            return {key: val.copy() for key, val in self.data.items()}

    def has_new_data(self, min_interval=0.5):

        current_time = time.time()
        if current_time - self.last_update >= min_interval:
            self.last_update = current_time
            return True
        return False


def create_optimized_chart(progress_data, title="SQS Optimization Progress (Live)"):
    if not progress_data['steps'] or len(progress_data['steps']) < 2:
        return None

    max_points = 1000
    steps = progress_data['steps']
    scores = progress_data['scores']
    temps = progress_data['temperatures']

    if len(steps) > max_points:
        step_size = len(steps) // max_points
        indices = range(0, len(steps), step_size)

        steps = [steps[i] for i in indices]
        scores = [scores[i] for i in indices]
        temps = [temps[i] for i in indices]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=scores,
            mode='lines',
            name='Best Score',
            line=dict(color='blue', width=2),
            hovertemplate='Step: %{x}<br>Best Score: %{y:.4f}<extra></extra>'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=temps,
            mode='lines',
            name='Temperature',
            line=dict(color='red', width=2),
            hovertemplate='Step: %{x}<br>Temperature: %{y:.3f}<extra></extra>'
        ),
        secondary_y=True
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16)
        ),
        xaxis_title='MC Step',
        height=300,
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98),
        # Optimize for performance
        uirevision='constant',  # Maintains zoom/pan state
    )

    fig.update_yaxes(title_text="Best Score", secondary_y=False, color='blue')
    fig.update_yaxes(title_text="Temperature", secondary_y=True, color='red')

    return fig


def thread_for_graph(last_update_time, message_queue, progress_data, progress_placeholder, status_placeholder,
                     chart_placeholder, update_interval):
    if isinstance(last_update_time, list):
        current_last_update = last_update_time[0]
    else:
        current_last_update = last_update_time

    try:
        messages_processed = 0
        max_messages = 20
        chart_updated = False

        while not message_queue.empty() and messages_processed < max_messages:
            message = message_queue.get_nowait()
            parsed = parse_icet_log_message(message)

            if parsed:
                progress_data['steps'].append(parsed['current_step'])
                progress_data['scores'].append(parsed['best_score'])
                progress_data['temperatures'].append(parsed['temperature'])
                progress_data['accepted_trials'].append(parsed['accepted_trials'])

                if progress_placeholder:
                    progress = min(parsed['current_step'] / max(1, parsed['total_steps'] - 1), 1.0)
                    progress_placeholder.progress(progress)

                if status_placeholder:
                    status_placeholder.text(
                        f"🔄 Step {parsed['current_step']}/{parsed['total_steps']} | "
                        f"Best Score: {parsed['best_score']:.4f} | "
                        f"Temperature: {parsed['temperature']:.3f} | "
                        f"Accepted: {parsed['accepted_trials']}"
                    )

                chart_updated = True

            messages_processed += 1

    except queue.Empty:
        pass

    current_time = time.time()
    if chart_placeholder and chart_updated and (current_time - current_last_update) > update_interval:
        if len(progress_data['steps']) > 1:
            try:
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                steps = progress_data['steps']
                scores = progress_data['scores']
                temps = progress_data['temperatures']

                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=scores,
                        mode='lines',
                        name='Best Score',
                        line=dict(color='blue', width=2),
                        hovertemplate='Step: %{x}<br>Best Score: %{y:.4f}<extra></extra>'
                    ),
                    secondary_y=False
                )

                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=temps,
                        mode='lines',
                        name='Temperature',
                        line=dict(color='red', width=2),
                        hovertemplate='Step: %{x}<br>Temperature: %{y:.3f}<extra></extra>'
                    ),
                    secondary_y=True
                )

                fig.update_layout(
                    title=dict(
                        text='SQS Optimization Progress (Live)',
                        font=dict(size=16)
                    ),
                    xaxis_title='MC Step',
                    height=300,
                    hovermode='x unified',
                    legend=dict(x=0.02, y=0.98)
                )

                fig.update_yaxes(title_text="Best Score", secondary_y=False, color='blue')
                fig.update_yaxes(title_text="Temperature", secondary_y=True, color='red')

                chart_placeholder.plotly_chart(fig, use_container_width=True,
                                               key=f"live_chart_{int(current_time * 1000)}")

                if isinstance(last_update_time, list):
                    last_update_time[0] = current_time

            except Exception as e:
                if status_placeholder:
                    status_placeholder.text(f"Continuing optimization... (chart update paused)")


def create_final_chart(progress_data, title="SQS Optimization Results"):
    if not progress_data['steps'] or len(progress_data['steps']) < 2:
        return None

    steps = progress_data['steps']
    scores = progress_data['scores']
    temps = progress_data['temperatures']

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=scores,
            mode='lines',
            name='Best Score',
            line=dict(color='blue', width=2),
            hovertemplate='Step: %{x}<br>Best Score: %{y:.4f}<extra></extra>'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=temps,
            mode='lines',
            name='Temperature',
            line=dict(color='red', width=2),
            hovertemplate='Step: %{x}<br>Temperature: %{y:.3f}<extra></extra>'
        ),
        secondary_y=True
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16)
        ),
        xaxis_title='MC Step',
        height=300,
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98)
    )

    fig.update_yaxes(title_text="Best Score", secondary_y=False, color='blue')
    fig.update_yaxes(title_text="Temperature", secondary_y=True, color='red')

    return fig


def generate_sqs_with_icet_progress_multi(primitive_structure, target_concentrations, transformation_matrix,
                                          cutoffs, method="monte_carlo", n_steps=10000, random_seed=42,
                                          progress_placeholder=None, chart_placeholder=None, status_placeholder=None):
    atoms = pymatgen_to_ase(primitive_structure)

    supercell = make_supercell(atoms, transformation_matrix)
    total_sites = len(supercell)

    achievable_concentrations, achievable_counts = calculate_achievable_concentrations(
        target_concentrations, total_sites)

    concentration_adjusted = False
    for element in target_concentrations:
        if abs(target_concentrations[element] - achievable_concentrations[element]) > 0.001:
            concentration_adjusted = True
            break

    #if concentration_adjusted:
    #    st.warning("⚠️ **Concentration Adjustment**: Target concentrations adjusted to achievable integer atom counts:")
    #    adj_data = []
    #    for element in sorted(target_concentrations.keys()):
    #        adj_data.append({
    #            "Element": element,
    #            "Target": f"{target_concentrations[element]:.3f}",
    #            "Achievable": f"{achievable_concentrations[element]:.3f}",
    #            "Atom Count": achievable_counts[element]
    #        })
    #    adj_df = pd.DataFrame(adj_data)
    #    st.dataframe(adj_df, use_container_width=True)

    all_elements = list(achievable_concentrations.keys())
    chemical_symbols = [all_elements for _ in range(len(atoms))]

    cs = icet.ClusterSpace(atoms, cutoffs, chemical_symbols)

    if random_seed > 0:
        random.seed(random_seed)
        np.random.seed(random_seed)

    message_queue = queue.Queue()
    log_handler = setup_icet_logging(message_queue)

    progress_data = {
        'steps': [],
        'scores': [],
        'temperatures': [],
        'accepted_trials': []
    }

    def run_sqs_generation():
        try:
            if method == "supercell_specific":
                supercells = [supercell]
                return generate_sqs_from_supercells(
                    cluster_space=cs,
                    supercells=supercells,
                    target_concentrations=achievable_concentrations,
                    n_steps=n_steps
                )
            else:
                return generate_sqs(
                    cluster_space=cs,
                    max_size=total_sites,
                    target_concentrations=achievable_concentrations,
                    n_steps=n_steps,
                    include_smaller_cells=False
                )
        except Exception as e:
            message_queue.put(f"ERROR: {str(e)}")
            raise e

    sqs_result = [None]
    exception_result = [None]

    def generation_thread():
        try:
            sqs_result[0] = run_sqs_generation()
        except Exception as e:
            exception_result[0] = e

    thread = threading.Thread(target=generation_thread)
    thread.start()

    last_update_time = [time.time()]
    update_interval = 0.5

    while thread.is_alive():
        thread_for_graph_multi_run(last_update_time, message_queue, progress_data, progress_placeholder,
                                   status_placeholder, chart_placeholder, update_interval)
        time.sleep(0.1)

    thread.join()
    remaining_messages = 0
    max_remaining = 50
    while not message_queue.empty() and remaining_messages < max_remaining:
        try:
            message = message_queue.get_nowait()
            parsed = parse_icet_log_message(message)
            if parsed:
                progress_data['steps'].append(parsed['current_step'])
                progress_data['scores'].append(parsed['best_score'])
                progress_data['temperatures'].append(parsed['temperature'])
                progress_data['accepted_trials'].append(parsed['accepted_trials'])
            remaining_messages += 1
        except queue.Empty:
            break

    if progress_placeholder:
        progress_placeholder.progress(1.0)

    if status_placeholder:
        if progress_data['scores']:
            best_score = min(progress_data['scores'])
            final_step = max(progress_data['steps']) if progress_data['steps'] else n_steps
            status_placeholder.text(
                f"✅ Run completed! Final step: {final_step}/{n_steps} | Best Score: {best_score:.4f}")
        else:
            status_placeholder.text("✅ SQS generation completed!")

    if chart_placeholder and len(progress_data['steps']) > 1:
        try:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            steps = progress_data['steps']
            scores = progress_data['scores']
            temps = progress_data['temperatures']

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=scores,
                    mode='lines',
                    name='Best Score',
                    line=dict(color='blue', width=1),
                    hovertemplate='Step: %{x}<br>Best Score: %{y:.4f}<extra></extra>'
                ),
                secondary_y=False
            )

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=temps,
                    mode='lines',
                    name='Temperature',
                    line=dict(color='red', width=1),
                    hovertemplate='Step: %{x}<br>Temperature: %{y:.3f}<extra></extra>'
                ),
                secondary_y=True
            )
            font_size = 14
            fig.update_layout(
                title=dict(
                    text='✅ Final SQS Optimization Results',
                    font=dict(size=font_size, family="Arial Black")
                ),
                xaxis_title='MC Step',
                height=300,
                hovermode='x unified',
                legend=dict(
                    x=0.02,
                    y=0.98,
                    font=dict(size=font_size, family="Arial Black")
                ),
                font=dict(size=font_size, family="Arial"),
                xaxis=dict(
                    title_font=dict(size=font_size, family="Arial Black"),
                    tickfont=dict(size=font_size, family="Arial")
                ),
                yaxis=dict(
                    title_font=dict(size=font_size, family="Arial Black"),
                    tickfont=dict(size=font_size, family="Arial")
                )
            )

            fig.update_yaxes(
                title_text="Best Score",
                secondary_y=False,
                color='blue',
                title_font=dict(size=font_size, family="Arial Black"),
                tickfont=dict(size=font_size, family="Arial")
            )
            fig.update_yaxes(
                title_text="Temperature",
                secondary_y=True,
                color='red',
                title_font=dict(size=font_size, family="Arial Black"),
                tickfont=dict(size=font_size, family="Arial")
            )

            final_chart_key = f"final_multi_chart_{getattr(st.session_state, 'current_multi_run', 0)}_{int(time.time() * 1000)}"
            chart_placeholder.plotly_chart(fig, use_container_width=True, key=final_chart_key)

        except Exception as e:
            st.warning(f"Could not update final chart: {e}")

    icet_logger = logging.getLogger('icet.target_cluster_vector_annealing')
    icet_logger.removeHandler(log_handler)

    if exception_result[0]:
        raise exception_result[0]

    return sqs_result[0], cs, achievable_concentrations, progress_data


from pymatgen.analysis.local_env import VoronoiNN
from matminer.featurizers.structure import PartialRadialDistributionFunction
from itertools import combinations
from collections import defaultdict
import plotly.graph_objects as go


def calculate_sqs_prdf(structure, cutoff=10.0, bin_size=0.1):
    try:

        elements = list(set([site.specie.symbol for site in structure if site.is_ordered]))

        species_combinations = list(combinations(elements, 2)) + [(s, s) for s in elements]

        prdf_featurizer = PartialRadialDistributionFunction(cutoff=cutoff, bin_size=bin_size)
        prdf_featurizer.fit([structure])

        prdf_data = prdf_featurizer.featurize(structure)
        feature_labels = prdf_featurizer.feature_labels()

        prdf_dict = defaultdict(list)
        distance_dict = {}

        for i, label in enumerate(feature_labels):
            parts = label.split(" PRDF r=")
            element_pair = tuple(parts[0].split("-"))
            distance_range = parts[1].split("-")
            bin_center = (float(distance_range[0]) + float(distance_range[1])) / 2
            prdf_dict[element_pair].append(prdf_data[i])

            if element_pair not in distance_dict:
                distance_dict[element_pair] = []
            distance_dict[element_pair].append(bin_center)

        return prdf_dict, distance_dict, species_combinations

    except Exception as e:
        st.error(f"Error calculating PRDF: {e}")
        return None, None, None


def calculate_and_display_sqs_prdf(sqs_structure, cutoff=10.0, bin_size=0.1):
    try:
        with st.expander("📊 PRDF Analysis of Generated SQS"):
            with st.spinner("Calculating PRDF..."):
                prdf_dict, distance_dict, species_combinations = calculate_sqs_prdf(
                    sqs_structure, cutoff=cutoff, bin_size=bin_size
                )

                if prdf_dict is not None:
                    import matplotlib.pyplot as plt
                    import numpy as np

                    colors = plt.cm.tab10.colors

                    def rgb_to_hex(color):
                        return '#%02x%02x%02x' % (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

                    font_dict = dict(size=18, color="black")

                    fig_combined = go.Figure()

                    for idx, (pair, prdf_values) in enumerate(prdf_dict.items()):
                        hex_color = rgb_to_hex(colors[idx % len(colors)])

                        fig_combined.add_trace(go.Scatter(
                            x=distance_dict[pair],
                            y=prdf_values,
                            mode='lines+markers',
                            name=f"{pair[0]}-{pair[1]}",
                            line=dict(color=hex_color, width=2),
                            marker=dict(size=6)
                        ))

                    fig_combined.update_layout(
                        title={'text': "SQS PRDF: All Element Pairs", 'font': font_dict},
                        xaxis_title={'text': "Distance (Å)", 'font': font_dict},
                        yaxis_title={'text': "PRDF Intensity", 'font': font_dict},
                        hovermode='x',
                        font=font_dict,
                        xaxis=dict(tickfont=font_dict),
                        yaxis=dict(tickfont=font_dict, range=[0, None]),
                        hoverlabel=dict(font=font_dict),
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.2,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=16)
                        )
                    )

                    st.plotly_chart(fig_combined, use_container_width=True)

                    import pandas as pd
                    import base64

                    for pair, prdf_values in prdf_dict.items():
                        df = pd.DataFrame()
                        df["Distance (Å)"] = distance_dict[pair]
                        df["PRDF"] = prdf_values

                        csv = df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        filename = f"SQS_{pair[0]}_{pair[1]}_prdf.csv"
                        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {pair[0]}-{pair[1]} PRDF data</a>'
                        st.markdown(href, unsafe_allow_html=True)

                    return True

                else:
                    st.error("Failed to calculate PRDF")
                    return False

    except Exception as e:
        st.error(f"Error calculating PRDF: {e}")
        return False


from io import StringIO
from ase.constraints import FixAtoms


def create_sqs_download_section(result, selected_file):
    st.subheader("📥 Download SQS Structure")

    col_download_format, col_download_button = st.columns([1, 1])

    with col_download_format:
        file_format = st.radio(
            f"Select file **format**",
            ("CIF", "VASP", "LAMMPS", "XYZ"),
            horizontal=True,
            key="sqs_download_format"
        )

    file_content = None
    download_file_name = None
    mime = "text/plain"

    sqs_structure = result['structure']

    try:
        if file_format == "CIF":
            from pymatgen.io.cif import CifWriter

            download_file_name = f"SQS_{selected_file.split('.')[0]}.cif"
            mime = "chemical/x-cif"
            file_content = result['cif_content']

        elif file_format == "VASP":
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.io import write
            from io import StringIO

            mime = "text/plain"
            download_file_name = f"SQS_{selected_file.split('.')[0]}.poscar"

            current_ase_structure = AseAtomsAdaptor.get_atoms(sqs_structure)

            col_vasp1, col_vasp2 = st.columns([1, 1])
            with col_vasp1:
                use_fractional = st.checkbox(
                    "Output POSCAR with fractional coordinates",
                    value=True,
                    key="sqs_poscar_fractional"
                )

            with col_vasp2:
                from ase.constraints import FixAtoms
                use_selective_dynamics = st.checkbox(
                    "Include Selective dynamics (all atoms free)",
                    value=False,
                    key="sqs_poscar_sd"
                )
                if use_selective_dynamics:
                    constraint = FixAtoms(indices=[])
                    current_ase_structure.set_constraint(constraint)

            out = StringIO()
            write(out, current_ase_structure, format="vasp", direct=use_fractional, sort=True)
            file_content = out.getvalue()

        elif file_format == "LAMMPS":
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.io import write
            from io import StringIO

            mime = "text/plain"
            download_file_name = f"SQS_{selected_file.split('.')[0]}.lmp"

            current_ase_structure = AseAtomsAdaptor.get_atoms(sqs_structure)

            st.markdown("**LAMMPS Export Options**")
            col_lmp1, col_lmp2 = st.columns([1, 1])

            with col_lmp1:
                atom_style = st.selectbox(
                    "Select atom_style",
                    ["atomic", "charge", "full"],
                    index=0,
                    key="sqs_lammps_atom_style"
                )
                units = st.selectbox(
                    "Select units",
                    ["metal", "real", "si"],
                    index=0,
                    key="sqs_lammps_units"
                )

            with col_lmp2:
                include_masses = st.checkbox(
                    "Include atomic masses",
                    value=True,
                    key="sqs_lammps_masses"
                )
                force_skew = st.checkbox(
                    "Force triclinic cell (skew)",
                    value=False,
                    key="sqs_lammps_skew"
                )

            out = StringIO()
            write(
                out,
                current_ase_structure,
                format="lammps-data",
                atom_style=atom_style,
                units=units,
                masses=include_masses,
                force_skew=force_skew
            )
            file_content = out.getvalue()

        elif file_format == "XYZ":
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.io import write
            from io import StringIO

            mime = "text/plain"
            download_file_name = f"SQS_{selected_file.split('.')[0]}.xyz"

            current_ase_structure = AseAtomsAdaptor.get_atoms(sqs_structure)

            out = StringIO()
            write(out, current_ase_structure, format="xyz")
            file_content = out.getvalue()

    except Exception as e:
        st.error(f"Error generating {file_format} file: {e}")
        st.error(
            f"There was an error processing the SQS structure for {file_format} format. "
            f"Please try a different format or check the structure validity."
        )

    with col_download_button:
        if file_content is not None:
            st.download_button(
                label=f"📥 Download {file_format} file",
                data=file_content,
                file_name=download_file_name,
                type="primary",
                mime=mime,
                key=f"sqs_download_{file_format.lower()}"
            )
        else:
            st.info(f"Select {file_format} format to enable download")


def pymatgen_to_ase(structure):
    from ase import Atoms
    import numpy as np

    symbols = [str(site.specie) for site in structure]
    positions = [site.coords for site in structure]
    cell = structure.lattice.matrix

    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=cell,
        pbc=True
    )
    return atoms



def thread_for_graph_multi_run(last_update_time, message_queue, progress_data, progress_placeholder, status_placeholder,
                               chart_placeholder, update_interval):
    if isinstance(last_update_time, list):
        current_last_update = last_update_time[0]
    else:
        current_last_update = last_update_time

    try:
        messages_processed = 0
        max_messages = 20
        chart_updated = False
        latest_parsed = None

        while not message_queue.empty() and messages_processed < max_messages:
            message = message_queue.get_nowait()
            parsed = parse_icet_log_message(message)

            if parsed:
                progress_data['steps'].append(parsed['current_step'])
                progress_data['scores'].append(parsed['best_score'])
                progress_data['temperatures'].append(parsed['temperature'])
                progress_data['accepted_trials'].append(parsed['accepted_trials'])
                latest_parsed = parsed

                if progress_placeholder:
                    try:
                        progress = min(parsed['current_step'] / max(1, parsed['total_steps'] - 1), 1.0)
                        progress_placeholder.progress(progress)
                    except:
                        pass

                if status_placeholder:
                    try:
                        status_placeholder.text(
                            f"🔄 MC Step {parsed['current_step']}/{parsed['total_steps']} | "
                            f"Best Score: {parsed['best_score']:.4f} | "
                            f"Temperature: {parsed['temperature']:.3f} | "
                            f"Accepted: {parsed['accepted_trials']}"
                        )
                    except Exception as e:
                        try:
                            status_placeholder.text(f"🔄 MC Step {parsed['current_step']}/{parsed['total_steps']}")
                        except:
                            pass

                chart_updated = True

            messages_processed += 1

    except queue.Empty:
        pass
    except Exception as e:
        if status_placeholder and latest_parsed:
            try:
                status_placeholder.text(f"🔄 MC Step {latest_parsed['current_step']}/{latest_parsed['total_steps']}")
            except:
                pass

    current_time = time.time()
    if chart_placeholder and chart_updated and (current_time - current_last_update) > update_interval:
        if len(progress_data['steps']) > 1:
            try:
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                steps = progress_data['steps']
                scores = progress_data['scores']
                temps = progress_data['temperatures']

                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=scores,
                        mode='lines',
                        name='Best Score',
                        line=dict(color='blue', width=4),
                        hovertemplate='Step: %{x}<br>Best Score: %{y:.4f}<extra></extra>'
                    ),
                    secondary_y=False
                )

                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=temps,
                        mode='lines',
                        name='Temperature',
                        line=dict(color='red', width=4),
                        hovertemplate='Step: %{x}<br>Temperature: %{y:.3f}<extra></extra>'
                    ),
                    secondary_y=True
                )

                font_ss = 14
                fig.update_layout(
                    title=dict(
                        text='SQS Optimization Progress (Live)',
                        font=dict(size=font_ss, family="Arial Black")
                    ),
                    xaxis_title='MC Step',
                    height=350,
                    hovermode='x unified',
                    legend=dict(
                        x=0.02,
                        y=0.98,
                        font=dict(size=font_ss, family="Arial Black")
                    ),
                    font=dict(size=font_ss, family="Arial"),
                    xaxis=dict(
                        title_font=dict(size=font_ss, family="Arial Black"),
                        tickfont=dict(size=font_ss, family="Arial")
                    ),
                    yaxis=dict(
                        title_font=dict(size=font_ss, family="Arial Black"),


                        tickfont=dict(size=font_ss, family="Arial")
                    )
                )

                fig.update_yaxes(
                    title_text="Best Score",
                    secondary_y=False,
                    color='blue',
                    title_font=dict(size=font_ss, family="Arial Black"),
                    tickfont=dict(size=font_ss, family="Arial")
                )
                fig.update_yaxes(
                    title_text="Temperature",
                    secondary_y=True,
                    color='red',
                    title_font=dict(size=font_ss, family="Arial Black"),
                    tickfont=dict(size=font_ss, family="Arial")
                )

                chart_key = f"multi_run_chart_{getattr(st.session_state, 'current_multi_run', 0)}_{int(current_time)}"
                chart_placeholder.plotly_chart(fig, use_container_width=True, key=chart_key)

                if isinstance(last_update_time, list):
                    last_update_time[0] = current_time

            except Exception as e:
                if status_placeholder and latest_parsed:
                    try:
                        status_placeholder.text(f"🔄 MC Step {latest_parsed['current_step']}/{latest_parsed['total_steps']} | Score: {latest_parsed['best_score']:.4f}")
                    except:
                        pass

    return latest_parsed


def generate_structure_file_content_multi(structure, file_format):
    try:
        if file_format == "CIF":
            from pymatgen.io.cif import CifWriter
            cif_writer = CifWriter(structure)
            return cif_writer.__str__()

        elif file_format == "VASP":
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.io import write
            from io import StringIO

            ase_structure = AseAtomsAdaptor.get_atoms(structure)
            out = StringIO()
            write(out, ase_structure, format="vasp", direct=True, sort=True)
            return out.getvalue()

        elif file_format == "LAMMPS":
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.io import write
            from io import StringIO

            ase_structure = AseAtomsAdaptor.get_atoms(structure)
            out = StringIO()
            write(out, ase_structure, format="lammps-data", atom_style="atomic", units="metal")
            return out.getvalue()

        elif file_format == "XYZ":
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.io import write
            from io import StringIO

            ase_structure = AseAtomsAdaptor.get_atoms(structure)
            out = StringIO()
            write(out, ase_structure, format="xyz")
            return out.getvalue()

        else:
            return "Unsupported format"

    except Exception as e:
        return f"Error generating {file_format}: {str(e)}"




def generate_sqs_with_icet_progress_sublattice(primitive_structure, chemical_symbols, target_concentrations,
                                               transformation_matrix, cutoffs, method="monte_carlo",
                                               n_steps=10000, random_seed=42, progress_placeholder=None,
                                               chart_placeholder=None, status_placeholder=None):
    atoms = pymatgen_to_ase(primitive_structure)

    supercell = make_supercell(atoms, transformation_matrix)
    total_sites = len(supercell)

    achievable_concentrations, adjustment_info = calculate_achievable_concentrations_sublattice(
        target_concentrations, chemical_symbols, transformation_matrix, primitive_structure
    )

    #if adjustment_info:
    #    st.warning(
    #        "⚠️ **Sublattice Concentration Adjustment**: Target concentrations adjusted to achievable integer atom counts:")
    #    adj_df = pd.DataFrame(adjustment_info)
    #    st.dataframe(adj_df, use_container_width=True)

    try:
        cs = icet.ClusterSpace(atoms, cutoffs, chemical_symbols)
    except Exception as e:
        st.error(f"Error creating ClusterSpace: {e}")
        st.write(f"Chemical symbols: {chemical_symbols}")
        st.write(f"Atoms symbols: {atoms.get_chemical_symbols()}")
        raise

    # Display sublattice information
    st.write("**Sublattice Configuration:**")
    sublattice_info_data = []
    for sublattice_id, sublattice_conc in achievable_concentrations.items():
        elements = list(sublattice_conc.keys())
        conc_str = ", ".join([f"{elem}: {conc:.3f}" for elem, conc in sublattice_conc.items()])
        sublattice_info_data.append({
            "Sublattice": sublattice_id,
            "Elements": ", ".join(elements),
            "Concentrations": conc_str
        })

    if sublattice_info_data:
        sublattice_df = pd.DataFrame(sublattice_info_data)
        st.dataframe(sublattice_df, use_container_width=True)

    if random_seed > 0:
        random.seed(random_seed)
        np.random.seed(random_seed)

    message_queue = queue.Queue()
    log_handler = setup_icet_logging(message_queue)

    progress_data = {
        'steps': [],
        'scores': [],
        'temperatures': [],
        'accepted_trials': []
    }

    def run_sqs_generation():
        time.sleep(5)
        try:
            if method == "supercell_specific":
                supercells = [supercell]
                return generate_sqs_from_supercells(
                    cluster_space=cs,
                    supercells=supercells,
                    target_concentrations=achievable_concentrations,
                    n_steps=n_steps
                )
            elif method == "enumeration":
                return generate_sqs_by_enumeration(
                    cluster_space=cs,
                    max_size=total_sites,
                    target_concentrations=achievable_concentrations
                )
            else:
                return generate_sqs(
                    cluster_space=cs,
                    max_size=total_sites,
                    target_concentrations=achievable_concentrations,
                    n_steps=n_steps,
                    include_smaller_cells=False
                )
        except Exception as e:
            message_queue.put(f"ERROR: {str(e)}")
            raise e

    sqs_result = [None]
    exception_result = [None]

    def generation_thread():
        try:
            sqs_result[0] = run_sqs_generation()
        except Exception as e:
            exception_result[0] = e

    thread = threading.Thread(target=generation_thread)
    thread.start()

    last_update_time = [time.time()]
    update_interval = 0.5

    while thread.is_alive():
        thread_for_graph(
            last_update_time,
            message_queue,
            progress_data,
            progress_placeholder,
            status_placeholder,
            chart_placeholder,
            update_interval
        )
        time.sleep(0.1)

    thread.join()

    remaining_messages = 0
    max_remaining = 50
    while not message_queue.empty() and remaining_messages < max_remaining:
        try:
            message = message_queue.get_nowait()
            parsed = parse_icet_log_message(message)
            if parsed:
                progress_data['steps'].append(parsed['current_step'])
                progress_data['scores'].append(parsed['best_score'])
                progress_data['temperatures'].append(parsed['temperature'])
                progress_data['accepted_trials'].append(parsed['accepted_trials'])
            remaining_messages += 1
        except queue.Empty:
            break

    if progress_placeholder:
        progress_placeholder.progress(1.0)

    if status_placeholder:
        if progress_data['scores']:
            best_score = min(progress_data['scores'])
            final_step = max(progress_data['steps']) if progress_data['steps'] else n_steps
            status_placeholder.text(
                f"✅ Generation completed! Final step: {final_step+1000}/{n_steps} | Best Score: {best_score:.4f}")
        else:
            status_placeholder.text("✅ SQS generation completed!")

    if chart_placeholder and len(progress_data['steps']) > 1:
        with chart_placeholder.container():
            st.write("**Final SQS Optimization Results:**")
            final_fig = create_final_chart(progress_data, title="Final SQS Optimization Results")
            if final_fig:
                current_run = getattr(st.session_state, 'current_multi_run', 0)
                chart_key = f"sqs_sublattice_final_chart_run_{current_run}_{int(time.time() * 1000)}"
                st.plotly_chart(final_fig, use_container_width=True, key=chart_key)
            else:
                st.info("Optimization completed - see live chart above for progress details.")

            if progress_data['scores']:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Steps", 1000*len(progress_data['steps']))
                with col2:
                    st.metric("Best Score", f"{min(progress_data['scores']):.4f}")
                with col3:
                    st.metric("Final Score", f"{progress_data['scores'][-1]:.4f}")

    icet_logger = logging.getLogger('icet.target_cluster_vector_annealing')
    if log_handler in icet_logger.handlers:
        icet_logger.removeHandler(log_handler)

    if exception_result[0]:
        raise exception_result[0]

    return sqs_result[0], cs, achievable_concentrations, progress_data

def generate_sqs_with_icet_progress_sublattice_multi(primitive_structure, chemical_symbols, target_concentrations,
                                                     transformation_matrix, cutoffs, method="monte_carlo",
                                                     n_steps=10000, random_seed=42, progress_placeholder=None,
                                                     chart_placeholder=None, status_placeholder=None):

    atoms = pymatgen_to_ase(primitive_structure)

    supercell = make_supercell(atoms, transformation_matrix)
    total_sites = len(supercell)

    achievable_concentrations, adjustment_info = calculate_achievable_concentrations_sublattice(
        target_concentrations, chemical_symbols, transformation_matrix, primitive_structure
    )

    #if adjustment_info:
    #    st.warning(
    #        "⚠️ **Sublattice Concentration Adjustment**: Target concentrations adjusted to achievable integer atom counts:")
    #    adj_df = pd.DataFrame(adjustment_info)
    #    st.dataframe(adj_df, use_container_width=True)

    try:
        cs = icet.ClusterSpace(atoms, cutoffs, chemical_symbols)
    except Exception as e:
        st.error(f"Error creating ClusterSpace: {e}")
        st.write(f"Chemical symbols: {chemical_symbols}")
        st.write(f"Atoms symbols: {atoms.get_chemical_symbols()}")
        raise

    # Display sublattice information
    #st.write("**Sublattice Configuration:**")
    #sublattice_info_data = []
    #for sublattice_id, sublattice_conc in achievable_concentrations.items():
    #    elements = list(sublattice_conc.keys())
    #    conc_str = ", ".join([f"{elem}: {conc:.3f}" for elem, conc in sublattice_conc.items()])
    #    sublattice_info_data.append({
    #        "Sublattice": sublattice_id,
    #        "Elements": ", ".join(elements),
    #        "Concentrations": conc_str
    #    })

    #if sublattice_info_data:
    #    sublattice_df = pd.DataFrame(sublattice_info_data)
    #    st.dataframe(sublattice_df, use_container_width=True)

    if random_seed > 0:
        random.seed(random_seed)
        np.random.seed(random_seed)

    message_queue = queue.Queue()
    log_handler = setup_icet_logging(message_queue)

    progress_data = {
        'steps': [],
        'scores': [],
        'temperatures': [],
        'accepted_trials': []
    }

    def run_sqs_generation():
        time.sleep(5)
        try:
            if method == "supercell_specific":
                supercells = [supercell]
                return generate_sqs_from_supercells(
                    cluster_space=cs,
                    supercells=supercells,
                    target_concentrations=achievable_concentrations,
                    n_steps=n_steps
                )
            elif method == "enumeration":
                return generate_sqs_by_enumeration(
                    cluster_space=cs,
                    max_size=total_sites,
                    target_concentrations=achievable_concentrations
                )
            else:  # monte_carlo
                return generate_sqs(
                    cluster_space=cs,
                    max_size=total_sites,
                    target_concentrations=achievable_concentrations,
                    n_steps=n_steps,
                    include_smaller_cells=False
                )
        except Exception as e:
            message_queue.put(f"ERROR: {str(e)}")
            raise e

    sqs_result = [None]
    exception_result = [None]

    def generation_thread():
        try:
            sqs_result[0] = run_sqs_generation()
        except Exception as e:
            exception_result[0] = e

    thread = threading.Thread(target=generation_thread)
    thread.start()

    last_update_time = [time.time()]
    update_interval = 0.5

    while thread.is_alive():
        thread_for_graph_multi_run(
            last_update_time,
            message_queue,
            progress_data,
            progress_placeholder,
            status_placeholder,
            chart_placeholder,
            update_interval
        )
        time.sleep(0.1)

    thread.join()

    remaining_messages = 0
    max_remaining = 50
    while not message_queue.empty() and remaining_messages < max_remaining:
        try:
            message = message_queue.get_nowait()
            parsed = parse_icet_log_message(message)
            if parsed:
                progress_data['steps'].append(parsed['current_step'])
                progress_data['scores'].append(parsed['best_score'])
                progress_data['temperatures'].append(parsed['temperature'])
                progress_data['accepted_trials'].append(parsed['accepted_trials'])
            remaining_messages += 1
        except queue.Empty:
            break

    if progress_placeholder:
        progress_placeholder.progress(1.0)

    if status_placeholder:
        if progress_data['scores']:
            best_score = min(progress_data['scores'])
            final_step = max(progress_data['steps']) if progress_data['steps'] else n_steps
            status_placeholder.text(
                f"✅ Run completed! Final step: {final_step}/{n_steps} | Best Score: {best_score:.4f}")
        else:
            status_placeholder.text("✅ SQS generation completed!")

    if chart_placeholder and len(progress_data['steps']) > 1:
        try:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            steps = progress_data['steps']
            scores = progress_data['scores']
            temps = progress_data['temperatures']

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=scores,
                    mode='lines',
                    name='Best Score',
                    line=dict(color='blue', width=1),
                    hovertemplate='Step: %{x}<br>Best Score: %{y:.4f}<extra></extra>'
                ),
                secondary_y=False
            )

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=temps,
                    mode='lines',
                    name='Temperature',
                    line=dict(color='red', width=1),
                    hovertemplate='Step: %{x}<br>Temperature: %{y:.3f}<extra></extra>'
                ),
                secondary_y=True
            )
            font_sizz = 12
            fig.update_layout(
                title=dict(
                    text='✅ Final SQS Optimization Results (Sublattice)',
                    font=dict(size=font_sizz, family="Arial Black")
                ),
                xaxis_title='MC Step',
                height=300,
                hovermode='x unified',
                legend=dict(
                    x=0.02,
                    y=0.98,
                    font=dict(size=font_sizz, family="Arial Black")
                ),
                font=dict(size=font_sizz, family="Arial"),
                xaxis=dict(
                    title_font=dict(size=font_sizz, family="Arial Black"),
                    tickfont=dict(size=font_sizz, family="Arial")
                ),
                yaxis=dict(
                    title_font=dict(size=font_sizz, family="Arial Black"),
                    tickfont=dict(size=font_sizz, family="Arial")
                )
            )

            fig.update_yaxes(
                title_text="Best Score",
                secondary_y=False,
                color='blue',
                title_font=dict(size=font_sizz, family="Arial Black"),
                tickfont=dict(size=font_sizz, family="Arial")
            )
            fig.update_yaxes(
                title_text="Temperature",
                secondary_y=True,
                color='red',
                title_font=dict(size=font_sizz, family="Arial Black"),
            )

            current_run = getattr(st.session_state, 'current_multi_run', 0)
            final_chart_key = f"final_sublattice_multi_chart_run_{current_run}_{int(time.time() * 1000)}"
            chart_placeholder.plotly_chart(fig, use_container_width=True, key=final_chart_key)

        except Exception as e:
            st.warning(f"Could not update final chart: {e}")

    icet_logger = logging.getLogger('icet.target_cluster_vector_annealing')
    if log_handler in icet_logger.handlers:
        icet_logger.removeHandler(log_handler)

    if exception_result[0]:
        raise exception_result[0]

    return sqs_result[0], cs, achievable_concentrations, progress_data

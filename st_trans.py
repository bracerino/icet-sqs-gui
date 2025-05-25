import streamlit as st
import io
import numpy as np
import random

from helpers import *
import time
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Element, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
import py3Dmol
import streamlit.components.v1 as components
import pandas as pd
import icet
from icet.tools.structure_generation import (
    generate_sqs,
    generate_sqs_from_supercells,
    generate_sqs_by_enumeration
)
from icet.input_output.logging_tools import set_log_config
from ase import Atoms
from ase.build import make_supercell
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
import logging
import threading
import queue
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_all_sites(structure):
    #
    try:
        sga = SpacegroupAnalyzer(structure)
        sym_data = sga.get_symmetry_dataset()
        wyckoffs = sym_data.get("wyckoffs", ["?"] * len(structure))
    except Exception:
        wyckoffs = ["?"] * len(structure)

    all_sites = []
    for i, site in enumerate(structure):

        if site.is_ordered:
            element = site.specie.symbol
        else:
            element = ", ".join(f"{sp.symbol}:{occ:.3f}" for sp, occ in site.species.items())

        all_sites.append({
            "site_index": i,
            "wyckoff_letter": wyckoffs[i],
            "element": element,
            "coords": site.frac_coords
        })

    return all_sites


def get_unique_sites(structure):
    try:
        analyzer = SpacegroupAnalyzer(structure)
        symmetry_data = analyzer.get_symmetry_dataset()
        wyckoff_letters = symmetry_data["wyckoffs"]
        equivalent_sites = analyzer.get_symmetrized_structure().equivalent_sites
        equivalent_indices = analyzer.get_symmetrized_structure().equivalent_indices

        unique_sites = []
        for i, equiv_indices in enumerate(equivalent_indices):
            site_index = equiv_indices[0]
            site = structure[site_index]

            if site.is_ordered:
                element = site.specie.symbol
            else:
                element = ", ".join([f"{sp.symbol}: {occ:.3f}" for sp, occ in site.species.items()])

            wyckoff = wyckoff_letters[site_index]
            coords = site.frac_coords
            unique_sites.append({
                'wyckoff_index': i,
                'site_index': site_index,
                'wyckoff_letter': wyckoff,
                'element': element,
                'coords': coords,
                'multiplicity': len(equiv_indices),
                'equivalent_indices': equiv_indices
            })

        return unique_sites
    except Exception as e:
        unique_sites = []
        for i, site in enumerate(structure):
            if site.is_ordered:
                element = site.specie.symbol
            else:
                element = ", ".join([f"{sp.symbol}: {occ:.3f}" for sp, occ in site.species.items()])

            unique_sites.append({
                'wyckoff_index': i,
                'site_index': i,
                'wyckoff_letter': "?",
                'element': element,
                'coords': site.frac_coords,
                'multiplicity': 1,
                'equivalent_indices': [i]
            })

        return unique_sites


def ase_to_pymatgen(atoms):
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    lattice = Lattice(cell)
    return Structure(lattice, symbols, positions, coords_are_cartesian=True)


def calculate_achievable_concentrations(target_concentrations, total_sites):
    """Calculate achievable integer atom counts and corresponding concentrations"""
    achievable_concentrations = {}
    achievable_counts = {}
    remaining_sites = total_sites

    # Sort by target count to allocate largest fractions first
    sorted_elements = sorted(target_concentrations.items(), key=lambda x: x[1], reverse=True)

    for i, (element, target_frac) in enumerate(sorted_elements):
        if i == len(sorted_elements) - 1:  # Last element gets remaining sites
            achievable_counts[element] = remaining_sites
        else:
            count = int(round(target_frac * total_sites))
            achievable_counts[element] = count
            remaining_sites -= count

    # Calculate achievable fractions
    for element, count in achievable_counts.items():
        achievable_concentrations[element] = count / total_sites

    return achievable_concentrations, achievable_counts


def generate_sqs_with_icet_progress(primitive_structure, target_concentrations, transformation_matrix,
                                    cutoffs, method="monte_carlo", n_steps=10000, random_seed=42,
                                    progress_placeholder=None, chart_placeholder=None, status_placeholder=None):
    # Convert to ASE
    atoms = pymatgen_to_ase(primitive_structure)

    supercell = make_supercell(atoms, transformation_matrix)
    total_sites = len(supercell)

    achievable_concentrations, achievable_counts = calculate_achievable_concentrations(
        target_concentrations, total_sites)

    # Check if adjustment was needed
    concentration_adjusted = False
    for element in target_concentrations:
        if abs(target_concentrations[element] - achievable_concentrations[element]) > 0.001:
            concentration_adjusted = True
            break

    if concentration_adjusted:
        st.warning("⚠️ **Concentration Adjustment**: Target concentrations adjusted to achievable integer atom counts:")
        adj_data = []
        for element in sorted(target_concentrations.keys()):
            adj_data.append({
                "Element": element,
                "Target": f"{target_concentrations[element]:.3f}",
                "Achievable": f"{achievable_concentrations[element]:.3f}",
                "Atom Count": achievable_counts[element]
            })
        adj_df = pd.DataFrame(adj_data)
        st.dataframe(adj_df, use_container_width=True)

    all_elements = list(achievable_concentrations.keys())
    chemical_symbols = [all_elements for _ in range(len(atoms))]

    cs = icet.ClusterSpace(atoms, cutoffs, chemical_symbols)
    # st.write("CLUSTER SPACE")
    # st.write(cs)

    if random_seed > 0:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Setup logging for progress tracking
    message_queue = queue.Queue()
    log_handler = setup_icet_logging(message_queue)

    # Storage for progress data
    progress_data = {
        'steps': [],
        'scores': [],
        'temperatures': [],
        'accepted_trials': []
    }

    # Function to run ICET generation in thread
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

    # Start SQS generation in separate thread
    sqs_result = [None]
    exception_result = [None]

    def generation_thread():
        try:
            sqs_result[0] = run_sqs_generation()
        except Exception as e:
            exception_result[0] = e

    thread = threading.Thread(target=generation_thread)
    thread.start()

    # Monitor progress
    last_update_time = time.time()
    update_interval = 0.5  # Update every 0.5 seconds

    while thread.is_alive():
        thread_for_graph(last_update_time, message_queue, progress_data, progress_placeholder, status_placeholder,
                         chart_placeholder,
                         update_interval)
        time.sleep(0.1)  # Short sleep to prevent busy waiting

    # Wait for thread to complete
    thread.join()

    # Clean up logging
    icet_logger = logging.getLogger('icet.target_cluster_vector_annealing')
    icet_logger.removeHandler(log_handler)

    if exception_result[0]:
        raise exception_result[0]

    # Final chart update

    return sqs_result[0], cs, achievable_concentrations, progress_data


def render_sqs_module():
    st.markdown(
        """
        <hr style="border: none; height: 6px; background-color: #3399ff; border-radius: 8px; margin: 20px 0;">
        """,
        unsafe_allow_html=True
    )

    st.title("🎲 Special Quasi-Random Structure (SQS) Generation")

    st.info("""
        Special Quasi-Random Structures (SQS) approximate random alloys by matching the correlation functions 
        of a truly random alloy in a finite supercell.
    """)

    # Initialize SQS mode without clearing data
    if "sqs_mode_initialized" not in st.session_state:
        if "calc_xrd" not in st.session_state:
            st.session_state.calc_xrd = False
        st.session_state.sqs_mode_initialized = True

    if 'full_structures' in st.session_state and st.session_state['full_structures']:
        file_options = list(st.session_state['full_structures'].keys())

        selected_sqs_file = st.selectbox(
            "Select structure for SQS transformation:",
            file_options,
            key="sqs_structure_selector"
        )

        try:
            sqs_structure = st.session_state['full_structures'][selected_sqs_file]

            st.write(f"**Selected structure:** {sqs_structure.composition.reduced_formula}")
            st.write(f"**Number of atoms:** {len(sqs_structure)}")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Structure Preparation")

                reduce_to_primitive = st.checkbox(
                    "Convert to primitive cell before SQS transformation",
                    value=True,
                    help="This will convert the structure to its primitive cell before applying SQS transformation."
                )

                if reduce_to_primitive:
                    analyzer = SpacegroupAnalyzer(sqs_structure)
                    primitive_structure = analyzer.get_primitive_standard_structure()
                    st.write(f"**Primitive cell contains {len(primitive_structure)} atoms**")
                    working_structure = primitive_structure
                else:
                    working_structure = sqs_structure

                try:
                    analyzer = SpacegroupAnalyzer(working_structure)
                    spg_symbol = analyzer.get_space_group_symbol()
                    spg_number = analyzer.get_space_group_number()
                    st.write(f"**Space group:** {spg_symbol} (#{spg_number})")
                except:
                    st.write("**Space group:** Could not determine")

                unique_sites = get_unique_sites(working_structure)
                all_sites = get_all_sites(working_structure)

                st.subheader("Wyckoff Positions Analysis")

                site_data = []
                for site_info in unique_sites:
                    site_data.append({
                        "Wyckoff Index": site_info['wyckoff_index'],
                        "Wyckoff Letter": site_info['wyckoff_letter'],
                        "Current Element": site_info['element'],
                        "Coordinates": f"({site_info['coords'][0]:.3f}, {site_info['coords'][1]:.3f}, {site_info['coords'][2]:.3f})",
                        "Multiplicity": site_info['multiplicity'],
                        "Site Indices": str(site_info['equivalent_indices'])
                    })

                site_df = pd.DataFrame(site_data)
                st.dataframe(site_df, use_container_width=True)

            st.markdown(
                """
                <hr style="border: none; height: 6px; background-color: #3399ff; border-radius: 8px; margin: 20px 0;">
                """,
                unsafe_allow_html=True
            )

            # ICET Method Selection
            st.subheader("✅ Step 1: Select Icet SQS Method")
            colzz, colss = st.columns([1,1])
            with colzz:

                sqs_method = st.radio(
                    "ICET SQS Method:",
                    ["Supercell-Specific", "Maximum n. of atoms (Not yet implemented)","Enumeration (Small Systems, Not yet implemented)"],
                    index=0,
                )

            with colss:
                st.write("**Cluster Cutoff Parameters**")
                colaa, colbb = st.columns([1, 1])
                with colaa:
                    cutoff_pair = st.number_input("Pair cutoff (Å):", min_value=1.0, max_value=10.0, value=7.0,
                                                  step=0.5)
                with colbb:
                    cutoff_triplet = st.number_input("Triplet cutoff (Å):", min_value=1.0, max_value=8.0, value=4.0,
                                                     step=0.5)
            n_steps = 10000
            # Map to internal method names
            method_map = {
                "Maximum n. of atoms (Not yet implemented)": "monte_carlo",
                "Supercell-Specific": "supercell_specific",
                "Enumeration (Small Systems, Not yet implemented)": "enumeration"
            }
            internal_method = method_map[sqs_method]
            #internal_method = 'Ss'

            if internal_method != "enumeration":
                n_steps = st.number_input(
                    "Number of Monte Carlo steps:",
                    min_value=1000,
                    max_value=1000000,
                    value=10000,
                    step=1000,
                    help="More steps generally lead to better SQS structures"
                )
            else:
                st.info("ℹ️ **Enumeration:** Will try all possible arrangements to find the optimal SQS.")
                try:
                    ase_atoms = pymatgen_to_ase(working_structure)
                   # supercell_preview = make_supercell(ase_atoms, transformation_matrix)
                   # if len(supercell_preview) > 20:
                   #     st.warning("⚠️ Warning: Enumeration may be very slow for systems with >20 atoms!")
                except:
                    pass

            random_seed = st.number_input(
                "Random seed (0 for random):",
                min_value=0,
                max_value=9999,
                value=42,
                help="Set a specific seed for reproducible results, or 0 for random"
            )

            cutoffs = [cutoff_pair, cutoff_triplet]

            st.subheader("✅ Step 2: Select Composition Mode")
            composition_mode = st.radio(
                "Choose composition specification mode:",
                [
                    "🔄 Global Composition",
                    "🎯 Sublattice-Specific"
                ],
                index=0,
                help="Global: Specify overall composition. Sublattice: Control each atomic position separately."
            )
            st.subheader("✅ Step 3: Supercell Configuration")

            col_x, col_y, col_z = st.columns(3)
            with col_x:
                nx = st.number_input("x-axis multiplier", value=2, min_value=1, max_value=10, step=1,
                                     key="nx_global")
            with col_y:
                ny = st.number_input("y-axis multiplier", value=2, min_value=1, max_value=10, step=1,
                                     key="ny_global")
            with col_z:
                nz = st.number_input("z-axis multiplier", value=2, min_value=1, max_value=10, step=1,
                                     key="nz_global")

            transformation_matrix = np.array([
                [nx, 0, 0],
                [0, ny, 0],
                [0, 0, nz]
            ])

            det = np.linalg.det(transformation_matrix)
            vol_or = nx * ny * nz
            st.write(f"**Supercell size:** {nx}×{ny}×{nz}")

            ase_atoms = pymatgen_to_ase(working_structure)
            supercell_preview = make_supercell(ase_atoms, transformation_matrix)
            st.write(f"**Preview: Supercell will contain {len(supercell_preview)} atoms**")
            # Get all elements in the structure
            all_elements = set()
            for site in working_structure:
                if site.is_ordered:
                    all_elements.add(site.specie.symbol)
                else:
                    for sp in site.species:
                        all_elements.add(sp.symbol)

            # Composition input
            if "sqs_composition_default" not in st.session_state:
                st.session_state.sqs_composition_default = ", ".join(sorted(list(all_elements)))


            if "previous_composition_mode" not in st.session_state:
                st.session_state.previous_composition_mode = composition_mode

            if st.session_state.previous_composition_mode != composition_mode:
                # Clear SQS results when switching modes to prevent conflicts
                if "sqs_results" in st.session_state:
                    st.session_state.sqs_results = {}
                st.session_state.previous_composition_mode = composition_mode
                st.rerun()

            use_sublattice_mode = composition_mode.startswith("🎯")

            if composition_mode == "🔄 Global Composition":
                # composition_input = st.text_input(
                #   "Enter elements for SQS (comma-separated, use 'X' for vacancy):",
                #   value=st.session_state.sqs_composition_default,
                #   key="sqs_composition_global",
                #   help="Example: 'Fe, Ni' for Fe-Ni alloy or 'O, X' for oxygen with vacancies"
                # )

                common_elements = [
                    'H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F',
                    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                    'Ga', 'Ge', 'As', 'Se', 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
                    'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
                    'Cs', 'Ba', 'La', 'Ce', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
                    'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'U', 'X'
                ]

                # Add vacancy and sort
                all_elements = sorted(common_elements)

                structure_elements = set()
                for site in working_structure:
                    if site.is_ordered:
                        structure_elements.add(site.specie.symbol)
                    else:
                        for sp in site.species:
                            structure_elements.add(sp.symbol)

                # Multiselect with structure elements as default
                st.subheader("✅ Step 4: Select Elements and Concentrations")
                element_list = st.multiselect(
                    "Select elements for SQS (use 'X' for vacancy):",
                    options=all_elements,
                    default=sorted(list(structure_elements)),
                    key="sqs_composition_global",
                    help="Example: Select 'Fe' and 'Ni' for Fe-Ni alloy, or 'O' and 'X' for oxygen with vacancies"
                )
                composition_input = ", ".join(element_list)
                # element_list = [e.strip() for e in composition_input.split(",")]

                # Composition sliders
                st.write("**Set target composition fractions:**")
                cols = st.columns(len(element_list))
                target_concentrations = {}

                remaining = 1.0
                for j, elem in enumerate(element_list[:-1]):
                    with cols[j]:
                        frac_val = st.slider(
                            f"{elem}:",
                            min_value=0.0,
                            max_value=remaining,
                            value=min(1.0 / len(element_list), remaining),
                            step=0.01,
                            format="%.2f",
                            key=f"sqs_comp_global_{elem}"
                        )
                        target_concentrations[elem] = frac_val
                        remaining -= frac_val

                if element_list:
                    last_elem = element_list[-1]
                    target_concentrations[last_elem] = max(0.0, remaining)
                    with cols[-1]:
                        st.write(f"**{last_elem}: {target_concentrations[last_elem]:.2f}**")

            else:
                composition_input = []
                chem_symbols, target_concentrations, otrs = render_site_sublattice_selector(working_structure,
                                                                                            all_sites)

            if composition_mode == "🔄 Global Composition":
                try:
                    # Calculate achievable concentrations
                    total_sites = len(supercell_preview)
                    st.write("**Target vs Achievable Concentrations:**")

                    conc_data = []
                    for element, target_frac in target_concentrations.items():
                        target_count = target_frac * total_sites
                        achievable_count = int(round(target_count))
                        achievable_frac = achievable_count / total_sites

                        status = "✅ Exact" if abs(target_frac - achievable_frac) < 0.01 else "⚠️ Rounded"

                        conc_data.append({
                            "Element": element,
                            "Target (%)": f"{target_frac * 100:.1f}",
                            "Achievable (%)": f"{achievable_frac * 100:.1f}",
                            "Atom Count": achievable_count,
                            "Status": status
                        })

                    conc_df = pd.DataFrame(conc_data)
                    st.dataframe(conc_df, use_container_width=True)

                except Exception as e:
                    st.error(f"Error creating supercell preview: {e}")
            else:
                display_sublattice_preview(target_concentrations, chem_symbols, transformation_matrix,
                                           working_structure)
            #st.subheader("SQS Generation Parameters")









            run_sqs = st.button("Generate SQS Structure", type="primary")

            # Initialize SQS results storage
            if "sqs_results" not in st.session_state:
                st.session_state.sqs_results = {}

            # Create configuration key for caching results
            config_str = composition_input
            current_config_key = f"{selected_sqs_file}_{reduce_to_primitive}_{nx}_{ny}_{nz}_{config_str}_{internal_method}_{n_steps}_{random_seed}"

            col_prdf1, col_prdf2 = st.columns(2)

            with col_prdf1:
                prdf_cutoff = st.number_input(
                    "⚙️ PRDF Cutoff (Å)",
                    min_value=1.0,
                    max_value=20.0,
                    value=10.0,
                    step=1.0,
                    format="%.1f",
                    key="sqs_prdf_cutoff",
                    help="Maximum distance for PRDF calculation"
                )

            with col_prdf2:
                prdf_bin_size = st.number_input(
                    "⚙️ PRDF Bin Size (Å)",
                    min_value=0.001,
                    max_value=1.000,
                    value=0.100,
                    step=0.010,
                    format="%.3f",
                    key="sqs_prdf_bin_size",
                    help="Resolution of distance bins"
                )

            # Generate SQS when button is clicked
            if run_sqs:
                # Create placeholders for progress tracking
                progress_container = st.container()
                with progress_container:
                    st.subheader("🔄 SQS Generation Progress")

                    if internal_method != "enumeration":
                        # Real-time progress tracking for ICET Monte Carlo methods
                        col_prog1, col_prog2 = st.columns([1, 1])

                        with col_prog1:
                            st.write("**Progress:**")
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                        with col_prog2:
                            st.write("**Live Statistics:**")
                            stats_placeholder = st.empty()

                        st.write("**Real-time Optimization Charts:**")
                        chart_placeholder = st.empty()
                    else:
                        # Simple progress for enumeration
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        chart_placeholder = None

                try:
                    start_time = time.time()
                    if not use_sublattice_mode:
                        if internal_method != "enumeration":
                            # Use ICET algorithms with progress tracking
                            status_text.text("🚀 Starting ICET SQS generation with real-time progress...")

                            sqs_atoms, cluster_space, achievable_concentrations, progress_data = generate_sqs_with_icet_progress(
                                primitive_structure=working_structure,
                                target_concentrations=target_concentrations,
                                transformation_matrix=transformation_matrix,
                                cutoffs=cutoffs,
                                method=internal_method,
                                n_steps=n_steps,
                                random_seed=random_seed,
                                progress_placeholder=progress_bar,
                                chart_placeholder=chart_placeholder,
                                status_placeholder=status_text
                            )

                            steps, energies, best_energy = [], [], None
                            if progress_data['scores']:
                                best_energy = min(progress_data['scores'])
                    else:
                        sqs_atoms, cluster_space, achievable_concentrations, progress_data = generate_sqs_with_icet_progress_sublattice(
                            primitive_structure=working_structure,
                            chemical_symbols=chem_symbols,
                            target_concentrations=target_concentrations,
                            transformation_matrix=transformation_matrix,
                            cutoffs=cutoffs,
                            method=internal_method,
                            n_steps=n_steps,
                            random_seed=random_seed,
                            progress_placeholder=progress_bar,
                            chart_placeholder=chart_placeholder,
                            status_placeholder=status_text
                        )
                        steps, energies, best_energy = [], [], None
                        if progress_data['scores']:
                            best_energy = min(progress_data['scores'])

                    elapsed_time = time.time() - start_time

                    sqs_result_with_vacancies = ase_to_pymatgen(sqs_atoms)
                    if not use_sublattice_mode:
                        all_used_elements = element_list
                    else:
                        # Sublattice mode - extract elements from chemical symbols
                        all_used_elements = []
                        for site_elements in chem_symbols:
                            for elem in site_elements:
                                if elem not in all_used_elements:
                                    all_used_elements.append(elem)

                    if 'X' in all_used_elements:
                        st.info("🔍 Removing vacancy sites ('X') from the final SQS structure...")

                        # Count vacancies before removal
                        vacancy_count = 0
                        for site in sqs_result_with_vacancies:
                            if site.is_ordered and site.specie.symbol == 'X':
                                vacancy_count += 1
                            elif not site.is_ordered:
                                for sp, occ in site.species.items():
                                    if sp.symbol == 'X':
                                        vacancy_count += occ

                        if vacancy_count > 0:
                            st.write(f"**Found {vacancy_count:.1f} vacancy sites to remove**")
                            sqs_result = remove_vacancies_from_structure(sqs_result_with_vacancies)
                            st.write(
                                f"**Final structure: {len(sqs_result_with_vacancies)} → {len(sqs_result)} atoms (removed {len(sqs_result_with_vacancies) - len(sqs_result)} vacancy sites)**")
                        else:
                            st.write("**No vacancy sites found**")
                            sqs_result = sqs_result_with_vacancies
                    else:
                        sqs_result = sqs_result_with_vacancies

                    calculate_and_display_sqs_prdf(
                        sqs_result,
                        cutoff=prdf_cutoff,
                        bin_size=prdf_bin_size
                    )
                    cluster_vector = cluster_space.get_cluster_vector(sqs_atoms)

                    # Create CIF
                    result_name = f"SQS_{selected_sqs_file.split('.')[0]}"
                    cif_writer = CifWriter(sqs_result)
                    cif_content = cif_writer.__str__()

                    # Store results
                    st.session_state.sqs_results[current_config_key] = {
                        'structure': sqs_result,
                        'cif_content': cif_content,
                        'cluster_vector': cluster_vector,
                        'elapsed_time': elapsed_time,
                        'result_name': result_name,
                        'algorithm': "ICET",
                        'method': sqs_method,
                        'target_concentrations': target_concentrations,
                        'achievable_concentrations': achievable_concentrations,
                        'best_energy': best_energy,
                        'optimization_plot': None,
                        'progress_data': progress_data
                    }

                    progress_container.empty()
                    st.success(f"🎉 SQS generation completed successfully!")

                except Exception as e:
                    st.error(f"Error generating SQS structure: {e}")
                    import traceback
                    st.error(traceback.format_exc())

            if current_config_key in st.session_state.sqs_results:
                result = st.session_state.sqs_results[current_config_key]

                icet_results_short_sum(result)

                st.subheader("Generated SQS Structure Information")
                # *** IMPORTED *** Information about the generated SQS structure
                generated_SQS_information(result)
                # ****************

                # Visualization
                st.write("**SQS Structure Visualization:**")
                # *** IMPORTED ***  Visualization of the generated SQS structure
                sqs_visualization(result)
                # ****************

                create_sqs_download_section(result, selected_sqs_file)

            with col2:
                # **** IMPORTED **** Visualization of the uploaded structure
                structure_preview(working_structure)
                # ****************


        except Exception as e:
            st.error(f"Error loading structure: {e}")
            import traceback
            st.error(traceback.format_exc())
    else:

        # **** IMPORTED **** Introduction text when nothing is yet uploaded
        intro_text()


def check_sqs_mode(calc_mode):
    if "previous_calc_mode" not in st.session_state:
        st.session_state.previous_calc_mode = calc_mode.copy()
    if "🎲 SQS Transformation" in calc_mode and "🎲 SQS Transformation" not in st.session_state.previous_calc_mode:
        st.cache_data.clear()
        st.cache_resource.clear()
        calc_mode = ["🎲 SQS Transformation"]
        if "sqs_mode_initialized" in st.session_state:
            del st.session_state.sqs_mode_initialized
        st.rerun()

    if "🎲 SQS Transformation" in calc_mode and len(calc_mode) > 1:
        calc_mode = ["🎲 SQS Transformation"]
        st.rerun()

    st.session_state.previous_calc_mode = calc_mode.copy()
    return calc_mode

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



if "previous_generation_mode" not in st.session_state:
    st.session_state.previous_generation_mode = "Single Run"


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


def create_bulk_download_zip_fixed(results, download_format, options=None):
    import zipfile
    from io import BytesIO
    import time

    if options is None:
        options = {}

    try:
        with st.spinner(f"Creating ZIP with {len(results)} {download_format} files..."):
            zip_buffer = BytesIO()

            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for result in results:
                    try:
                        file_content = generate_structure_file_content_with_options(
                            result['structure'],
                            download_format,
                            options
                        )

                        file_extension = get_file_extension(download_format)
                        filename = f"SQS_run_{result['run_number']}_seed_{result['seed']}.{file_extension}"
                        zip_file.writestr(filename, file_content)

                    except Exception as e:
                        error_filename = f"ERROR_run_{result['run_number']}_seed_{result['seed']}.txt"
                        error_content = f"Error generating {download_format} file: {str(e)}"
                        zip_file.writestr(error_filename, error_content)

            zip_buffer.seek(0)

            timestamp = int(time.time())
            zip_filename = f"SQS_multi_run_{download_format}_{timestamp}.zip"

            st.download_button(
                label=f"📥 Download ZIP ({len(results)} files)",
                data=zip_buffer.getvalue(),
                file_name=zip_filename,
                mime="application/zip",
                type="primary",
                key=f"zip_download_{timestamp}",
                help=f"Download ZIP file containing all {len(results)} SQS structures in {download_format} format"
            )

            st.success(f"✅ ZIP file with {len(results)} {download_format} structures ready!")

    except Exception as e:
        st.error(f"Error creating ZIP file: {e}")
        st.error("Please try again or check your structure files.")
def get_file_extension(file_format):
    extensions = {
        "CIF": "cif",
        "VASP": "poscar",
        "LAMMPS": "lmp",
        "XYZ": "xyz"
    }
    return extensions.get(file_format, "txt")


def generate_structure_file_content_with_options(structure, file_format, options=None):
    if options is None:
        options = {}

    try:
        if file_format == "CIF":
            from pymatgen.io.cif import CifWriter
            cif_writer = CifWriter(structure)
            return cif_writer.__str__()

        elif file_format == "VASP":
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.io import write
            from ase.constraints import FixAtoms
            from io import StringIO

            ase_structure = AseAtomsAdaptor.get_atoms(structure)


            use_fractional = options.get('use_fractional', True)
            use_selective_dynamics = options.get('use_selective_dynamics', False)

            if use_selective_dynamics:
                constraint = FixAtoms(indices=[])
                ase_structure.set_constraint(constraint)

            out = StringIO()
            write(out, ase_structure, format="vasp", direct=use_fractional, sort=True)
            return out.getvalue()

        elif file_format == "LAMMPS":
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.io import write
            from io import StringIO

            ase_structure = AseAtomsAdaptor.get_atoms(structure)
            atom_style = options.get('atom_style', 'atomic')
            units = options.get('units', 'metal')
            include_masses = options.get('include_masses', True)
            force_skew = options.get('force_skew', False)

            out = StringIO()
            write(
                out,
                ase_structure,
                format="lammps-data",
                atom_style=atom_style,
                units=units,
                masses=include_masses,
                force_skew=force_skew
            )
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
            return f"Unsupported format: {file_format}"

    except Exception as e:
        return f"Error generating {file_format}: {str(e)}"
def create_bulk_download_zip(results, download_format, options=None):
    import zipfile
    from io import BytesIO

    if options is None:
        options = {}

    try:
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for result in results:
                try:
                    file_content = generate_structure_file_content_with_options(
                        result['structure'],
                        download_format,
                        options
                    )

                    file_extension = get_file_extension(download_format)
                    filename = f"SQS_run_{result['run_number']}_seed_{result['seed']}.{file_extension}"
                    zip_file.writestr(filename, file_content)

                except Exception as e:
                    error_filename = f"ERROR_run_{result['run_number']}_seed_{result['seed']}.txt"
                    error_content = f"Error generating {download_format} file: {str(e)}"
                    zip_file.writestr(error_filename, error_content)

        zip_buffer.seek(0)

        timestamp = int(time.time())
        options_str = ""
        if download_format == "VASP" and options:
            coord_type = "frac" if options.get('use_fractional', True) else "cart"
            sd_type = "sd" if options.get('use_selective_dynamics', False) else "nosd"
            options_str = f"_{coord_type}_{sd_type}"
        elif download_format == "LAMMPS" and options:
            atom_style = options.get('atom_style', 'atomic')
            units = options.get('units', 'metal')
            options_str = f"_{atom_style}_{units}"

        zip_filename = f"SQS_multi_run_{download_format}{options_str}_{timestamp}.zip"

        zip_key = f"zip_download_bulk_{download_format}_{timestamp}"

        st.download_button(
            label=f"📥 Download {download_format} ZIP ({len(results)} files)",
            data=zip_buffer.getvalue(),
            file_name=zip_filename,
            mime="application/zip",
            type="primary",
            key=zip_key,
            help=f"Download ZIP file containing all {len(results)} successful SQS structures in {download_format} format"
        )

        st.success(f"✅ ZIP file with {len(results)} {download_format} structures ready for download!")

    except Exception as e:
        st.error(f"Error creating ZIP file: {e}")
        st.error("Please try again or contact support if the problem persists.")


def display_multi_run_results(all_results=None, download_format="CIF"):
    import numpy as np
    if all_results is None:
        if "multi_run_results" in st.session_state and st.session_state.multi_run_results:
            all_results = st.session_state.multi_run_results
        else:
            return

    if not all_results:
        return

    results_data = []
    valid_results = [r for r in all_results if r.get('best_score') is not None]

    for result in all_results:
        if result.get('best_score') is not None:
            results_data.append({
                "Run": result['run_number'],
                "Seed": result['seed'],
                "Best Score": f"{result['best_score']:.4f}",
                "Time (s)": f"{result['elapsed_time']:.1f}",
                "Atoms": len(result['structure']) if result.get('structure') else 0,
                "Status": "✅ Success"
            })
        else:
            results_data.append({
                "Run": result['run_number'],
                "Seed": result['seed'],
                "Best Score": "Failed",
                "Time (s)": f"{result.get('elapsed_time', 0):.1f}",
                "Atoms": 0,
                "Status": "❌ Error"
            })

    if results_data:
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)

    if valid_results:
        best_result = min(valid_results, key=lambda x: x['best_score'])
        st.success(f"🥇 **Best Result:** Run {best_result['run_number']} with score {best_result['best_score']:.4f}")

        scores = [r['best_score'] for r in valid_results]
        if len(scores) > 1:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Score", f"{min(scores):.4f}")
            with col2:
                st.metric("Worst Score", f"{max(scores):.4f}")
            with col3:
                st.metric("Average Score", f"{np.mean(scores):.4f}")
            with col4:
                st.metric("Std Dev", f"{np.std(scores):.4f}")

    st.subheader("📥 Download Individual Structures")

    col_format, col_options = st.columns([1, 2])

    with col_format:
        selected_format = st.selectbox(
            "Select format:",
            ["CIF", "VASP", "LAMMPS", "XYZ"],
            index=0,
            key="multi_run_individual_format"
        )

    format_options = {}
    with col_options:
        if selected_format == "VASP":
            st.write("**VASP Options:**")
            col_vasp1, col_vasp2 = st.columns(2)
            with col_vasp1:
                format_options['use_fractional'] = st.checkbox(
                    "Fractional coordinates",
                    value=True,
                    key="multi_vasp_fractional"
                )
            with col_vasp2:
                format_options['use_selective_dynamics'] = st.checkbox(
                    "Selective dynamics",
                    value=False,
                    key="multi_vasp_selective"
                )

        elif selected_format == "LAMMPS":
            st.write("**LAMMPS Options:**")
            col_lmp1, col_lmp2 = st.columns(2)
            with col_lmp1:
                format_options['atom_style'] = st.selectbox(
                    "Atom style:",
                    ["atomic", "charge", "full"],
                    index=0,
                    key="multi_lammps_atom_style"
                )
                format_options['units'] = st.selectbox(
                    "Units:",
                    ["metal", "real", "si"],
                    index=0,
                    key="multi_lammps_units"
                )
            with col_lmp2:
                format_options['include_masses'] = st.checkbox(
                    "Include masses",
                    value=True,
                    key="multi_lammps_masses"
                )
                format_options['force_skew'] = st.checkbox(
                    "Force triclinic",
                    value=False,
                    key="multi_lammps_skew"
                )
        else:
            st.write("")

    successful_results = [r for r in all_results if r.get('structure') is not None]

    if successful_results:
        best_run_number = min(successful_results, key=lambda x: x.get('best_score', float('inf'))).get('run_number')

        num_cols = min(4, len(successful_results))
        cols = st.columns(num_cols)

        for idx, result in enumerate(successful_results):
            with cols[idx % num_cols]:
                is_best = (result['run_number'] == best_run_number)
                button_type = "primary" if is_best else "secondary"
                label = f"📥 Run {result['run_number']}" + (" 🥇" if is_best else "")
                label += f"\nScore: {result['best_score']:.4f}"

                try:
                    file_content = generate_structure_file_content_with_options(
                        result['structure'],
                        selected_format,
                        format_options
                    )

                    file_extension = get_file_extension(selected_format)
                    filename = f"SQS_run_{result['run_number']}_seed_{result['seed']}.{file_extension}"

                    unique_key = f"download_run_{result['run_number']}_{result['seed']}_{selected_format}_{hash(str(format_options))}"

                    st.download_button(
                        label=label,
                        data=file_content,
                        file_name=filename,
                        mime="text/plain",
                        type=button_type,
                        key=unique_key,
                        help=f"Download {selected_format} structure from run {result['run_number']}"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.warning("No successful runs are available for download.")

    if valid_results:
        best_result = min(valid_results, key=lambda x: x['best_score'])

        st.subheader("🏆 Best Structure Analysis")
        colp1, colp2 = st.columns([1, 1])
        with colp2:
            #with st.expander(f"📊 Structure Details for Best Run {best_result['run_number']}", expanded=False):
            st.markdown(f"📊 Structure Details for Best Run {best_result['run_number']}")
            col_info1, col_info2 = st.columns(2)

            with col_info1:
                st.write("**Structure Information:**")
                best_structure = best_result['structure']
                comp = best_structure.composition
                comp_data = []

                for el, amt in comp.items():
                    actual_frac = amt / comp.num_atoms
                    comp_data.append({
                        "Element": el.symbol,
                        "Count": int(amt),
                        "Fraction": f"{actual_frac:.4f}"
                    })
                comp_df = pd.DataFrame(comp_data)
                st.dataframe(comp_df, use_container_width=True)

            with col_info2:
                st.write("**Lattice Parameters:**")
                lattice = best_structure.lattice
                st.write(f"a = {lattice.a:.4f} Å")
                st.write(f"b = {lattice.b:.4f} Å")
                st.write(f"c = {lattice.c:.4f} Å")
                st.write(f"α = {lattice.alpha:.2f}°")
                st.write(f"β = {lattice.beta:.2f}°")
                st.write(f"γ = {lattice.gamma:.2f}°")
                st.write(f"Volume = {lattice.volume:.2f} Ų")
        with colp1:
            st.write("**3D Structure Visualization:**")
            try:
                from io import StringIO
                import py3Dmol
                import streamlit.components.v1 as components
                from pymatgen.io.ase import AseAtomsAdaptor
                from ase.io import write
                import numpy as np

                jmol_colors = {
                    'H': '#FFFFFF', 'He': '#D9FFFF', 'Li': '#CC80FF', 'Be': '#C2FF00', 'B': '#FFB5B5',
                    'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D', 'F': '#90E050', 'Ne': '#B3E3F5',
                    'Na': '#AB5CF2', 'Mg': '#8AFF00', 'Al': '#BFA6A6', 'Si': '#F0C8A0', 'P': '#FF8000',
                    'S': '#FFFF30', 'Cl': '#1FF01F', 'Ar': '#80D1E3', 'K': '#8F40D4', 'Ca': '#3DFF00',
                    'Sc': '#E6E6E6', 'Ti': '#BFC2C7', 'V': '#A6A6AB', 'Cr': '#8A99C7', 'Mn': '#9C7AC7',
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

                structure_ase = AseAtomsAdaptor.get_atoms(best_structure)
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
                st.error(f"Error visualizing best structure: {e}")

        st.write("**PRDF Analysis:**")
        try:
            prdf_cutoff = st.session_state.get('sqs_prdf_cutoff', 10.0)
            prdf_bin_size = st.session_state.get('sqs_prdf_bin_size', 0.1)

            calculate_and_display_sqs_prdf(best_structure, cutoff=prdf_cutoff, bin_size=prdf_bin_size)

        except Exception as e:
            st.error(f"Error calculating PRDF for best structure: {e}")
            st.info("PRDF analysis could not be completed for the best structure.")

    if successful_results:
        st.subheader("📦 Bulk Download")

        col_bulk_format, col_bulk_options = st.columns([1, 2])

        with col_bulk_format:
            bulk_format = st.selectbox(
                "Bulk format:",
                ["CIF", "VASP", "LAMMPS", "XYZ"],
                index=0,
                key="bulk_format_selector"
            )

        bulk_options = {}
        with col_bulk_options:
            if bulk_format == "VASP":
                st.write("**VASP Bulk Options:**")
                col_bulk_vasp1, col_bulk_vasp2 = st.columns(2)
                with col_bulk_vasp1:
                    bulk_options['use_fractional'] = st.checkbox(
                        "Fractional coordinates",
                        value=True,
                        key="bulk_vasp_fractional"
                    )
                with col_bulk_vasp2:
                    bulk_options['use_selective_dynamics'] = st.checkbox(
                        "Selective dynamics",
                        value=False,
                        key="bulk_vasp_selective"
                    )

            elif bulk_format == "LAMMPS":
                st.write("**LAMMPS Bulk Options:**")
                col_bulk_lmp1, col_bulk_lmp2 = st.columns(2)
                with col_bulk_lmp1:
                    bulk_options['atom_style'] = st.selectbox(
                        "Atom style:",
                        ["atomic", "charge", "full"],
                        index=0,
                        key="bulk_lammps_atom_style"
                    )
                    bulk_options['units'] = st.selectbox(
                        "Units:",
                        ["metal", "real", "si"],
                        index=0,
                        key="bulk_lammps_units"
                    )
                with col_bulk_lmp2:
                    bulk_options['include_masses'] = st.checkbox(
                        "Include masses",
                        value=True,
                        key="bulk_lammps_masses"
                    )
                    bulk_options['force_skew'] = st.checkbox(
                        "Force triclinic",
                        value=False,
                        key="bulk_lammps_skew"
                    )

        if st.button("📥 Download all structures as ZIP", type="primary", key="bulk_download_button"):
            create_bulk_download_zip_fixed(successful_results, bulk_format, bulk_options)


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
        time.sleep(1)
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
                f"✅ Run completed! Final step: {final_step+1000}/{n_steps} | Best Score: {best_score:.4f}")
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
            font_sizz = 14
            fig.update_layout(
                title=dict(
                    text='✅ Final SQS Optimization Results (Sublattice)',
                    font=dict(size=font_sizz, family="Arial Black")
                ),
                xaxis_title='MC Step',
                height=300,
                width=600,
                hovermode='x unified',
                legend=dict(
                    x=1.02,
                    y=1,
                    xanchor='left',
                    font=dict(size=10, family="Arial")
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
                tickfont=dict(size=font_sizz, family="Arial")
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


def handle_multi_run_button(working_structure, target_concentrations, transformation_matrix,
                            cutoffs, internal_method, n_steps, num_runs, multi_run_base_seed,
                            use_sublattice_mode, chem_symbols, multi_run_download_format):
    st.session_state.multi_run_in_progress = True

    st.info("🧹 Starting new multi-run generation...")
    st.session_state.multi_run_results = []
    st.session_state.multi_run_active = True
    st.session_state.multi_run_completed = False

    if "persistent_prdf_data" in st.session_state:
        st.session_state.persistent_prdf_data = None
    if "sqs_results" in st.session_state:
        st.session_state.sqs_results = {}

    run_multiple_sqs_generations(
        primitive_structure=working_structure,
        target_concentrations=target_concentrations,
        transformation_matrix=transformation_matrix,
        cutoffs=cutoffs,
        method=internal_method,
        n_steps=n_steps,
        num_runs=num_runs,
        base_seed=multi_run_base_seed,
        use_sublattice_mode=use_sublattice_mode,
        chem_symbols=chem_symbols,
        download_format=multi_run_download_format
    )

    st.session_state.multi_run_completed = True
    st.session_state.multi_run_in_progress = False




def run_multiple_sqs_generations(primitive_structure, target_concentrations, transformation_matrix,
                                 cutoffs, method, n_steps, num_runs, base_seed,
                                 use_sublattice_mode=False, chem_symbols=None, download_format="CIF"):
    st.session_state.multi_run_active = True
    st.session_state.multi_run_completed = False

    st.subheader("🔄 Multi-Run SQS Generation Progress")
    overall_progress_container = st.container()
    with overall_progress_container:
        overall_progress = st.progress(0)
        overall_status = st.empty()
    run_progress_container = st.container()

    all_results = []

    for run_idx in range(num_runs):
        current_run = run_idx + 1
        overall_progress.progress(run_idx / num_runs)
        overall_status.text(f"**Running SQS Generation {current_run} of {num_runs}**")

        if base_seed == 0:
            current_seed = random.randint(1, 9999)
        else:
            current_seed = base_seed + run_idx * 42

        st.session_state.current_multi_run = current_run

        with run_progress_container:
            st.write(f"### 🏃 Run {current_run}: Seed {current_seed}")
            run_progress = st.progress(0)
            run_status = st.empty()
            chart_placeholder = st.empty()
            run_status.text(f"🚀 Starting optimization with seed {current_seed}...")

        try:
            start_time = time.time()

            if use_sublattice_mode:
                sqs_atoms, _, _, progress_data = generate_sqs_with_icet_progress_sublattice_multi(
                    primitive_structure=primitive_structure, chemical_symbols=chem_symbols,
                    target_concentrations=target_concentrations, transformation_matrix=transformation_matrix,
                    cutoffs=cutoffs, method=method, n_steps=n_steps, random_seed=current_seed,
                    progress_placeholder=run_progress, chart_placeholder=chart_placeholder,
                    status_placeholder=run_status
                )
            else:
                sqs_atoms, _, _, progress_data = generate_sqs_with_icet_progress_multi(
                    primitive_structure=primitive_structure, target_concentrations=target_concentrations,
                    transformation_matrix=transformation_matrix, cutoffs=cutoffs, method=method,
                    n_steps=n_steps, random_seed=current_seed, progress_placeholder=run_progress,
                    chart_placeholder=chart_placeholder, status_placeholder=run_status
                )

            elapsed_time = time.time() - start_time
            sqs_result = ase_to_pymatgen(sqs_atoms)

            elements_with_potential_vacancies = []
            if use_sublattice_mode:
                for site_elements in chem_symbols:
                    elements_with_potential_vacancies.extend(site_elements)
            else:
                elements_with_potential_vacancies = list(target_concentrations.keys())

            if 'X' in set(elements_with_potential_vacancies):
                sqs_result = remove_vacancies_from_structure(sqs_result)

            best_score = min(progress_data['scores']) if progress_data.get('scores') else None
            file_content = generate_structure_file_content_multi(sqs_result, download_format)

            all_results.append({
                'run_number': current_run, 'seed': current_seed, 'structure': sqs_result,
                'best_score': best_score, 'elapsed_time': elapsed_time, 'file_content': file_content,
                'format': download_format, 'progress_data': progress_data
            })
            run_status.text(f"✅ Run {current_run} completed! Score: {best_score:.4f} (Time: {elapsed_time:.1f}s)")
            run_progress.progress(1.0)
            time.sleep(0.5)

        except Exception as e:
            st.error(f"❌ Error in run {current_run}: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            all_results.append({
                'run_number': current_run, 'seed': current_seed, 'structure': None, 'best_score': None,
                'elapsed_time': time.time() - start_time if 'start_time' in locals() else 0,
                'file_content': f"Error: {str(e)}",
                'format': download_format, 'progress_data': {'scores': [], 'steps': []}
            })
            continue

    overall_progress.progress(1.0)
    overall_status.text(f"🎉 **All {num_runs} runs completed!**")

    st.session_state.multi_run_results = all_results
    st.session_state.multi_run_active = False


def check_multi_run_completion():
    if st.session_state.get("multi_run_completed", False):
        st.session_state.multi_run_completed = False
        # Trigger UI refresh
        #st.rerun()

def generate_sqs_with_icet_progress(primitive_structure, target_concentrations, transformation_matrix,
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
                f"✅ Generation completed! Final step: {final_step}/{n_steps} | Best Score: {best_score:.4f}")
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
                    text='✅ Final SQS Optimization Results (Global)',
                    font=dict(size=font_size, family="Arial Black")
                ),
                xaxis_title='MC Step',
                height=300,
                hovermode='x unified',
                legend=dict(x=0.02, y=0.98),
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

            final_chart_key = f"final_global_single_chart_{int(time.time() * 1000)}"
            chart_placeholder.plotly_chart(fig, use_container_width=True, key=final_chart_key)

        except Exception as e:
            st.warning(f"Could not update final chart: {e}")

    icet_logger = logging.getLogger('icet.target_cluster_vector_annealing')
    if log_handler in icet_logger.handlers:
        icet_logger.removeHandler(log_handler)

    if exception_result[0]:
        raise exception_result[0]

    return sqs_result[0], cs, achievable_concentrations, progress_data

if "persistent_prdf_data" not in st.session_state:
    st.session_state.persistent_prdf_data = None
if "prdf_structure_key" not in st.session_state:
    st.session_state.prdf_structure_key = None


def render_sqs_module():
    check_multi_run_completion()


    st.title("🎲 Special Quasi-Random Structure (SQS) Generation using Icet Package")
    st.markdown(
        """
        <hr style="border: none; height: 6px; background-color: #3399ff; border-radius: 8px; margin: 20px 0;">
        """,
        unsafe_allow_html=True
    )
    st.info("""
        Special Quasi-Random Structures (SQS) approximate random alloys by matching the correlation functions 
        of a truly random alloy in a finite supercell.
    """)


    if "sqs_mode_initialized" not in st.session_state:
        if "calc_xrd" not in st.session_state:
            st.session_state.calc_xrd = False
        st.session_state.sqs_mode_initialized = True

    if 'full_structures' in st.session_state and st.session_state['full_structures']:
        file_options = list(st.session_state['full_structures'].keys())

        selected_sqs_file = st.sidebar.selectbox(
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
                    value=False,
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
            with col2:
                structure_preview(working_structure)
            st.markdown(
                """
                <hr style="border: none; height: 6px; background-color: #3399ff; border-radius: 8px; margin: 20px 0;">
                """,
                unsafe_allow_html=True
            )

            st.subheader("1️⃣ Step 1: Select Icet SQS Method")
            colzz, colss = st.columns([1, 1])
            with colzz:

                sqs_method = st.radio(
                    "ICET SQS Method:",
                    ["Supercell-Specific", "Maximum n. of atoms (Not yet implemented)",
                     "Enumeration (For max 24 atoms)"],
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
            method_map = {
                "Maximum n. of atoms (Not yet implemented)": "monte_carlo",
                "Supercell-Specific": "supercell_specific",
                "Enumeration (For max 24 atoms)": "enumeration"
            }
            internal_method = method_map[sqs_method]
            colsz, colsc = st.columns(2)
            with colsc:
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
                    except:
                        pass
            with colsz:
                generation_mode = st.radio(
                    "Choose generation mode:",
                    ["Single Run", "Multiple Runs"],
                    key="generation_mode_selector",
                    help="Single: Generate one SQS. Multiple: Generate several with different seeds."
                )
                if generation_mode == "Multiple Runs":
                    col_runs, col_seed, col_format = st.columns(3)

                    with col_runs:
                        num_runs = st.number_input("Number of runs:", min_value=2, max_value=20, value=5, step=1)
                    with col_seed:
                        multi_run_base_seed = st.number_input("Base seed (0 for random):", min_value=0, max_value=9999,
                                                              value=42)
                    with col_format:
                        #multi_run_download_format = st.selectbox("Download format:", ["CIF", "VASP", "LAMMPS", "XYZ"])
                        multi_run_download_format = 'CIF'
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
            with colsc:
                if generation_mode == "Single Run":
                    random_seed = st.number_input(
                        "Random seed (0 for random):",
                        min_value=0,
                        max_value=9999,
                        value=42,
                        help="Set a specific seed for reproducible results, or 0 for random"
                    )

            cutoffs = [cutoff_pair, cutoff_triplet]
            st.markdown(
                """
                <hr style="border: none; height: 6px; background-color: #3399ff; border-radius: 8px; margin: 20px 0;">
                """,
                unsafe_allow_html=True
            )
            st.subheader("2️⃣ Step 2: Select Composition Mode")
            colb1, colb2 = st.columns([1,1])
            with colb1:
                composition_mode = st.radio(
                    "Choose composition specification mode:",
                    [
                        "🔄 Global Composition",
                        "🎯 Sublattice-Specific"
                    ],
                    index=1,
                    key="composition_mode_radio",
                    help="Global: Specify overall composition. Sublattice: Control each atomic position separately."
                )
            with colb2:
                with st.expander("ℹ️ Composition Mode Details", expanded=False):
                    st.markdown("""
                    ### 🔄 Global Composition
                    - Specify the target composition for the entire structure (e.g., 50% Fe, 50% Ni)
                    - All crystallographic sites can be occupied by any of the selected elements
                    - Elements are distributed randomly throughout the structure according to the specified fractions
                    - **Example:** Fe₀.₅Ni₀.₅ random alloy where Fe and Ni atoms can occupy any position

                    ---

                    ### 🎯 Sublattice-Specific  
                    - Control which elements can occupy specific crystallographic sites (Wyckoff positions)
                    - Set different compositions for different atomic sublattices
                    - **Example:** In a perovskite ABO₃, control A-site (Ba/Sr) and B-site (Ti/Zr) compositions independently

                    ---

                    ### 🔑 Key Difference
                    **Global** treats all sites equally, while **Sublattice-Specific** allows site-dependent element distributions based on crystallographic positions.

                    ### 💡 When to Use Which?
                    - **Choose Global** for: e.g., binary/ternary alloys, solid solutions, when all sites are equivalent
                    - **Choose Sublattice** for: e.g., intermetallic compounds, ceramics, when different sites have chemical preferences
                    """)

            st.markdown(
                """
                <hr style="border: none; height: 6px; background-color: #3399ff; border-radius: 8px; margin: 20px 0;">
                """,
                unsafe_allow_html=True
            )
            st.subheader("3️⃣ Step 3: Supercell Configuration")

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

            all_elements = set()
            for site in working_structure:
                if site.is_ordered:
                    all_elements.add(site.specie.symbol)
                else:
                    for sp in site.species:
                        all_elements.add(sp.symbol)

            if "sqs_composition_default" not in st.session_state:
                st.session_state.sqs_composition_default = ", ".join(sorted(list(all_elements)))

            if "previous_composition_mode" not in st.session_state:
                st.session_state.previous_composition_mode = composition_mode

            if st.session_state.previous_composition_mode != composition_mode:
                if "sqs_results" in st.session_state:
                    st.session_state.sqs_results = {}
                st.session_state.previous_composition_mode = composition_mode
                #st.rerun()

            use_sublattice_mode = composition_mode.startswith("🎯")

            target_concentrations = {}
            chem_symbols = None
            otrs = None

            if composition_mode == "🔄 Global Composition":
                common_elements = [
                    'H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F',
                    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                    'Ga', 'Ge', 'As', 'Se', 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
                    'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
                    'Cs', 'Ba', 'La', 'Ce', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
                    'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'U', 'X'
                ]

                all_elements = sorted(common_elements)

                structure_elements = set()
                for site in working_structure:
                    if site.is_ordered:
                        structure_elements.add(site.specie.symbol)
                    else:
                        for sp in site.species:
                            structure_elements.add(sp.symbol)
                st.markdown(
                    """
                    <hr style="border: none; height: 6px; background-color: #3399ff; border-radius: 8px; margin: 20px 0;">
                    """,
                    unsafe_allow_html=True
                )
                st.subheader("4️⃣ Step 4: Select Elements and Concentrations")
                element_list = st.multiselect(
                    "Select elements for SQS (use 'X' for vacancy):",
                    options=all_elements,
                    default=sorted(list(structure_elements)),
                    key="sqs_composition_global",
                    help="Example: Select 'Fe' and 'Ni' for Fe-Ni alloy, or 'O' and 'X' for oxygen with vacancies"
                )
                composition_input = ", ".join(element_list)

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





           # if "previous_generation_mode" not in st.session_state:
           #     st.session_state.previous_generation_mode = generation_mode

            #if st.session_state.previous_generation_mode != generation_mode:
            #    if "sqs_results" in st.session_state:
            #        st.session_state.sqs_results = {}
            #    st.session_state.previous_generation_mode = generation_mode
            st.markdown(
                """
                <hr style="border: none; height: 6px; background-color: #3399ff; border-radius: 8px; margin: 20px 0;">
                """,
                unsafe_allow_html=True
            )
            if generation_mode == "Multiple Runs":
                multi_run_download_format = 'CIF'

                st.markdown("""
                    <style>
                    div.stButton > button[kind="primary"] {
                        background-color: #0099ff; color: white; font-size: 16px; font-weight: bold;
                        padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
                    }
                    div.stButton > button[kind="primary"]:active, div.stButton > button[kind="primary"]:focus {
                        background-color: #007acc !important; color: white !important; box-shadow: none !important;
                    }

                    div.stButton > button[kind="secondary"] {
                        background-color: #dc3545; color: white; font-size: 16px; font-weight: bold;
                        padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
                    }
                    div.stButton > button[kind="secondary"]:active, div.stButton > button[kind="secondary"]:focus {
                        background-color: #c82333 !important; color: white !important; box-shadow: none !important;
                    }

                    div.stButton > button[kind="tertiary"] {
                        background-color: #6f42c1; color: white; font-size: 16px; font-weight: bold;
                        padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
                    }
                    div.stButton > button[kind="tertiary"]:active, div.stButton > button[kind="tertiary"]:focus {
                        background-color: #5a2d91 !important; color: white !important; box-shadow: none !important;
                    }

                    div[data-testid="stDataFrameContainer"] table td { font-size: 16px !important; }
                    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
                    </style>
                """, unsafe_allow_html=True)


                if not target_concentrations:
                    st.warning("Create atleast 1 sublattice (with minimum of two elements) first.")
                    run_multi_sqs = st.button(" Generate Multiple SQS Structures", type="tertiary", disabled = True,
                                              help = "Configure atleast 1 sublattice concentration first.")
                elif len(supercell_preview) > 24 and internal_method == "enumeration":
                    st.warning("Please select different SQS method. Enumeration is only possible up to 24 atoms due to its"
                            "computational complexity.")
                    run_multi_sqs = st.button(" Generate Multiple SQS Structures", type="tertiary", disabled = True,
                                              )
                else:
                    run_multi_sqs = st.button(" Generate Multiple SQS Structures", type="tertiary")

                if run_multi_sqs:
                    handle_multi_run_button(
                        working_structure, target_concentrations, transformation_matrix,
                        cutoffs, internal_method, n_steps, num_runs, multi_run_base_seed,
                        use_sublattice_mode, chem_symbols, multi_run_download_format
                    )

            else:

                st.markdown("""
                    <style>
                    div.stButton > button[kind="primary"] {
                        background-color: #0099ff; color: white; font-size: 16px; font-weight: bold;
                        padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
                    }
                    div.stButton > button[kind="primary"]:active, div.stButton > button[kind="primary"]:focus {
                        background-color: #007acc !important; color: white !important; box-shadow: none !important;
                    }

                    div.stButton > button[kind="secondary"] {
                        background-color: #dc3545; color: white; font-size: 16px; font-weight: bold;
                        padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
                    }
                    div.stButton > button[kind="secondary"]:active, div.stButton > button[kind="secondary"]:focus {
                        background-color: #c82333 !important; color: white !important; box-shadow: none !important;
                    }

                    div.stButton > button[kind="tertiary"] {
                        background-color: #6f42c1; color: white; font-size: 16px; font-weight: bold;
                        padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
                    }
                    div.stButton > button[kind="tertiary"]:active, div.stButton > button[kind="tertiary"]:focus {
                        background-color: #5a2d91 !important; color: white !important; box-shadow: none !important;
                    }

                    div[data-testid="stDataFrameContainer"] table td { font-size: 16px !important; }
                    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
                    </style>
                """, unsafe_allow_html=True)

                if not target_concentrations:
                    st.warning("Create atleast 1 sublattice (with minimum of two elements) first.")
                    run_sqs = st.button("Generate SQS Structure", type="tertiary", disabled = True,
                                        help = "Create atleast 1 sublattice (with minimum of two elements) first.")
                else:
                    run_sqs = st.button("Generate SQS Structure", type="tertiary")
                if "sqs_results" not in st.session_state:
                    st.session_state.sqs_results = {}

                config_str = str(target_concentrations)
                current_config_key = f"{selected_sqs_file}_{reduce_to_primitive}_{nx}_{ny}_{nz}_{config_str}_{internal_method}_{n_steps}_{random_seed}"

                if run_sqs:
                    progress_container = st.container()
                    with progress_container:
                        st.subheader("🔄 SQS Generation Progress")
                        if internal_method != "enumeration":
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            chart_placeholder = st.empty()
                        else:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            chart_placeholder = None

                    try:
                        start_time = time.time()
                        if not use_sublattice_mode:
                            sqs_atoms, cluster_space, achievable_concentrations, progress_data = generate_sqs_with_icet_progress(
                                primitive_structure=working_structure,
                                target_concentrations=target_concentrations,
                                transformation_matrix=transformation_matrix,
                                cutoffs=cutoffs, method=internal_method, n_steps=n_steps,
                                random_seed=random_seed, progress_placeholder=progress_bar,
                                chart_placeholder=chart_placeholder, status_placeholder=status_text
                            )
                        else:
                            sqs_atoms, cluster_space, achievable_concentrations, progress_data = generate_sqs_with_icet_progress_sublattice(
                                primitive_structure=working_structure, chemical_symbols=chem_symbols,
                                target_concentrations=target_concentrations,
                                transformation_matrix=transformation_matrix, cutoffs=cutoffs,
                                method=internal_method, n_steps=n_steps, random_seed=random_seed,
                                progress_placeholder=progress_bar, chart_placeholder=chart_placeholder,
                                status_placeholder=status_text
                            )

                        elapsed_time = time.time() - start_time
                        best_energy = min(progress_data['scores']) if progress_data.get('scores') else None

                        sqs_result_with_vacancies = ase_to_pymatgen(sqs_atoms)
                        all_used_elements = []
                        if not use_sublattice_mode:
                            all_used_elements = list(target_concentrations.keys())
                        else:
                            for site_elements in chem_symbols:
                                for elem in site_elements:
                                    if elem not in all_used_elements:
                                        all_used_elements.append(elem)

                        if 'X' in all_used_elements:
                            sqs_result = remove_vacancies_from_structure(sqs_result_with_vacancies)
                        else:
                            sqs_result = sqs_result_with_vacancies

                        cif_writer = CifWriter(sqs_result)
                        cif_content = cif_writer.__str__()
                        st.session_state.sqs_results[current_config_key] = {
                            'structure': sqs_result, 'cif_content': cif_content,
                            'elapsed_time': elapsed_time, 'result_name': f"SQS_{selected_sqs_file.split('.')[0]}",
                            'algorithm': "ICET", 'method': sqs_method,
                            'target_concentrations': target_concentrations,
                            'achievable_concentrations': achievable_concentrations,
                            'best_energy': best_energy, 'progress_data': progress_data
                        }
                        progress_container.empty()
                        #st.success(f"🎉 SQS generation completed successfully!")

                    except Exception as e:
                        st.error(f"Error generating SQS structure: {e}")
                        import traceback
                        st.error(traceback.format_exc())
                colhh1, colhh2, = st.columns(2)
                if current_config_key in st.session_state.sqs_results:
                    result = st.session_state.sqs_results[current_config_key]
                    icet_results_short_sum(result)
                    st.subheader("Generated SQS Structure Information")
                    generated_SQS_information(result)
                    with colhh1:
                        st.write("**SQS Structure Visualization:**")
                        sqs_visualization(result)
                    calculate_and_display_sqs_prdf(result['structure'], cutoff=prdf_cutoff, bin_size=prdf_bin_size)
                    with colhh2:
                        create_sqs_download_section(result, selected_sqs_file)

            st.markdown("---")

            if ("multi_run_results" in st.session_state and
                    st.session_state.multi_run_results and
                    len(st.session_state.multi_run_results) > 0):

                st.subheader("🏆 Previous Multi-Run Results")
                st.info("Results from previous multi-run generations are shown below:")

                display_format = "CIF"
                if st.session_state.multi_run_results:
                    first_result = st.session_state.multi_run_results[0]
                    if 'format' in first_result:
                        display_format = first_result['format']

                display_multi_run_results(download_format=display_format)

        except Exception as e:
            st.error(f"Error loading structure: {e}")
            import traceback
            st.error(traceback.format_exc())
    else:
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
        #st.rerun()

    if "🎲 SQS Transformation" in calc_mode and len(calc_mode) > 1:
        calc_mode = ["🎲 SQS Transformation"]
        #st.rerun()

    st.session_state.previous_calc_mode = calc_mode.copy()
    return calc_mode

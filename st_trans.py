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
from mp_api.client import MPRester
import spglib
from pymatgen.core import Structure
from aflow import search, K
from aflow import search  # ensure your file is not named aflow.py!
import aflow.keywords as AFLOW_K
import requests
import io
import re

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

MP_API_KEY = "UtfGa1BUI3RlWYVwfpMco2jVt8ApHOye"

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
                    "H": "#FFFFFF",
                    "He": "#D9FFFF",
                    "Li": "#CC80FF",
                    "Be": "#C2FF00",
                    "B": "#FFB5B5",
                    "C": "#909090",
                    "N": "#3050F8",
                    "O": "#FF0D0D",
                    "F": "#90E050",
                    "Ne": "#B3E3F5",
                    "Na": "#AB5CF2",
                    "Mg": "#8AFF00",
                    "Al": "#BFA6A6",
                    "Si": "#F0C8A0",
                    "P": "#FF8000",
                    "S": "#FFFF30",
                    "Cl": "#1FF01F",
                    "Ar": "#80D1E3",
                    "K": "#8F40D4",
                    "Ca": "#3DFF00",
                    "Sc": "#E6E6E6",
                    "Ti": "#BFC2C7",
                    "V": "#A6A6AB",
                    "Cr": "#8A99C7",
                    "Mn": "#9C7AC7",
                    "Fe": "#E06633",
                    "Co": "#F090A0",
                    "Ni": "#50D050",
                    "Cu": "#C88033",
                    "Zn": "#7D80B0",
                    "Ga": "#C28F8F",
                    "Ge": "#668F8F",
                    "As": "#BD80E3",
                    "Se": "#FFA100",
                    "Br": "#A62929",
                    "Kr": "#5CB8D1",
                    "Rb": "#702EB0",
                    "Sr": "#00FF00",
                    "Y": "#94FFFF",
                    "Zr": "#94E0E0",
                    "Nb": "#73C2C9",
                    "Mo": "#54B5B5",
                    "Tc": "#3B9E9E",
                    "Ru": "#248F8F",
                    "Rh": "#0A7D8C",
                    "Pd": "#006985",
                    "Ag": "#C0C0C0",
                    "Cd": "#FFD98F",
                    "In": "#A67573",
                    "Sn": "#668080",
                    "Sb": "#9E63B5",
                    "Te": "#D47A00",
                    "I": "#940094",
                    "Xe": "#429EB0",
                    "Cs": "#57178F",
                    "Ba": "#00C900",
                    "La": "#70D4FF",
                    "Ce": "#FFFFC7",
                    "Pr": "#D9FFC7",
                    "Nd": "#C7FFC7",
                    "Pm": "#A3FFC7",
                    "Sm": "#8FFFC7",
                    "Eu": "#61FFC7",
                    "Gd": "#45FFC7",
                    "Tb": "#30FFC7",
                    "Dy": "#1FFFC7",
                    "Ho": "#00FF9C",
                    "Er": "#00E675",
                    "Tm": "#00D452",
                    "Yb": "#00BF38",
                    "Lu": "#00AB24",
                    "Hf": "#4DC2FF",
                    "Ta": "#4DA6FF",
                    "W": "#2194D6",
                    "Re": "#267DAB",
                    "Os": "#266696",
                    "Ir": "#175487",
                    "Pt": "#D0D0E0",
                    "Au": "#FFD123",
                    "Hg": "#B8B8D0",
                    "Tl": "#A6544D",
                    "Pb": "#575961",
                    "Bi": "#9E4FB5",
                    "Po": "#AB5C00",
                    "At": "#754F45",
                    "Rn": "#428296",
                    "Fr": "#420066",
                    "Ra": "#007D00",
                    "Ac": "#70ABFA",
                    "Th": "#00BAFF",
                    "Pa": "#00A1FF",
                    "U": "#008FFF",
                    "Np": "#0080FF",
                    "Pu": "#006BFF",
                    "Am": "#545CF2",
                    "Cm": "#785CE3",
                    "Bk": "#8A4FE3",
                    "Cf": "#A136D4",
                    "Es": "#B31FD4",
                    "Fm": "#B31FBA",
                    "Md": "#B30DA6",
                    "No": "#BD0D87",
                    "Lr": "#C70066",
                    "Rf": "#CC0059",
                    "Db": "#D1004F",
                    "Sg": "#D90045",
                    "Bh": "#E00038",
                    "Hs": "#E6002E",
                    "Mt": "#EB0026"
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
        st.write("#### **Element Distribution:**")

        element_counts = {}
        total_atoms = comp.num_atoms

        for el, amt in comp.items():
            element_counts[el.symbol] = int(amt)

        cols = st.columns(min(len(element_counts), 4))
        for i, (elem, count) in enumerate(sorted(element_counts.items())):
            percentage = count / total_atoms * 100
            with cols[i % len(cols)]:
                if percentage >= 80:
                    color = "#2E4057"  # Dark Blue-Gray for very high concentration
                elif percentage >= 60:
                    color = "#4A6741"  # Dark Forest Green for high concentration
                elif percentage >= 40:
                    color = "#6B73FF"  # Purple-Blue for medium-high concentration
                elif percentage >= 25:
                    color = "#FF8C00"  # Dark Orange for medium concentration
                elif percentage >= 15:
                    color = "#4ECDC4"  # Teal for medium-low concentration
                elif percentage >= 10:
                    color = "#45B7D1"  # Blue for low-medium concentration
                elif percentage >= 5:
                    color = "#96CEB4"  # Green for low concentration
                elif percentage >= 2:
                    color = "#FECA57"  # Yellow for very low concentration
                elif percentage >= 1:
                    color = "#DDA0DD"  # Plum for trace concentration
                else:
                    color = "#D3D3D3"  # Light Gray for minimal concentration

                st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, {color}, {color}CC);
                            padding: 20px; 
                            border-radius: 15px; 
                            text-align: center; 
                            margin: 10px 0;
                            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
                            border: 2px solid rgba(255,255,255,0.2);
                        ">
                            <h1 style="
                                color: white; 
                                font-size: 3em; 
                                margin: 0; 
                                text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
                                font-weight: bold;
                            ">{elem}</h1>
                            <h2 style="
                                color: white; 
                                font-size: 2em; 
                                margin: 10px 0 0 0;
                                text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
                            ">{percentage:.1f}%</h2>
                            <p style="
                                color: white; 
                                font-size: 1.8em; 
                                margin: 5px 0 0 0;
                                opacity: 0.9;
                            ">{count} atoms</p>
                        </div>
                        """, unsafe_allow_html=True)
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
                f"✅ Generation completed! Final step: {final_step+1000}/{n_steps} | Best Score: {best_score:.4f}")
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


    st.title("🎲 Special Quasi-Random Structure (SQS) Generation using ICET Package")
    st.markdown(f"**Article for ICET (please cite this)**: [ÅNGQVIST, Mattias, et al. ICET–A Python library for constructing and sampling alloy cluster expansions. Advanced Theory and Simulations, 2019](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/adts.201900015?casa_token=cVHsP6-qM_cAAAAA%3AkLdF6LOJks6NUpk1gChewQP7Rax_MJTDoNjfm9TO3_vVxV7NbVLJKTwK3ZHXbXMaV7BwuSFteaci_cw)")
    st.markdown(
        """
        <hr style="border: none; height: 6px; background-color: #3399ff; border-radius: 8px; margin: 20px 0;">
        """,
        unsafe_allow_html=True
    )


    # -------------- DATABASE ----------
    show_database_search = st.checkbox("🗃️ Enable database search (MP, AFLOW, COD)",
                                       value=False,
                                       help="🗃️ Enable to search in Materials Project, AFLOW, and COD databases")
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
           </style>
       """, unsafe_allow_html=True)

    def get_space_group_info(number):
        symbol = SPACE_GROUP_SYMBOLS.get(number, f"SG#{number}")
        return symbol

    if show_database_search:
        with st.expander("Search for Structures Online in Databases", icon="🔍", expanded=True):
            cols, cols2, cols3 = st.columns([1.5, 1.5, 3.5])
            with cols:
                db_choices = st.multiselect(
                    "Select Database(s)",
                    options=["Materials Project", "AFLOW", "COD"],
                    default=["Materials Project", "AFLOW", "COD"],
                    help="Choose which databases to search for structures. You can select multiple databases."
                )

                if not db_choices:
                    st.warning("Please select at least one database to search.")

                st.markdown(
                    "**Maximum number of structures to be found in each database (for improving performance):**")
                col_limits = st.columns(3)

                search_limits = {}
                if "Materials Project" in db_choices:
                    with col_limits[0]:
                        search_limits["Materials Project"] = st.number_input(
                            "MP Limit:", min_value=1, max_value=2000, value=300, step=10,
                            help="Maximum results from Materials Project"
                        )
                if "AFLOW" in db_choices:
                    with col_limits[1]:
                        search_limits["AFLOW"] = st.number_input(
                            "AFLOW Limit:", min_value=1, max_value=2000, value=300, step=10,
                            help="Maximum results from AFLOW"
                        )
                if "COD" in db_choices:
                    with col_limits[2]:
                        search_limits["COD"] = st.number_input(
                            "COD Limit:", min_value=1, max_value=2000, value=300, step=10,
                            help="Maximum results from COD"
                        )

            with cols2:
                search_mode = st.radio(
                    "Search by:",
                    options=["Elements", "Structure ID", "Space Group + Elements", "Formula", "Search Mineral"],
                    help="Choose your search strategy"
                )

                if search_mode == "Elements":
                    selected_elements = st.multiselect(
                        "Select elements for search:",
                        options=ELEMENTS,
                        default=["Sr", "Ti", "O"],
                        help="Choose one or more chemical elements"
                    )
                    search_query = " ".join(selected_elements) if selected_elements else ""

                elif search_mode == "Structure ID":
                    structure_ids = st.text_area(
                        "Enter Structure IDs (one per line):",
                        value="mp-5229\ncod_1512124\naflow:010158cb2b41a1a5",
                        help="Enter structure IDs. Examples:\n- Materials Project: mp-5229\n- COD: cod_1512124 (with cod_ prefix)\n- AFLOW: aflow:010158cb2b41a1a5 (AUID format)"
                    )

                elif search_mode == "Space Group + Elements":
                    col_sg1, col_sg2 = st.columns(2)
                    with col_sg1:
                        all_space_groups_help = "Enter space group number (1-230)\n\nAll space groups:\n\n"
                        for num in sorted(SPACE_GROUP_SYMBOLS.keys()):
                            all_space_groups_help += f"• {num}: {SPACE_GROUP_SYMBOLS[num]}\n\n"

                        space_group_number = st.number_input(
                            "Space Group Number:",
                            min_value=1,
                            max_value=230,
                            value=221,
                            help=all_space_groups_help
                        )
                        sg_symbol = get_space_group_info(space_group_number)
                        st.info(f"#:**{sg_symbol}**")

                    selected_elements = st.multiselect(
                        "Select elements for search:",
                        options=ELEMENTS,
                        default=["Sr", "Ti", "O"],
                        help="Choose one or more chemical elements"
                    )

                elif search_mode == "Formula":
                    formula_input = st.text_input(
                        "Enter Chemical Formula:",
                        value="Sr Ti O3",
                        help="Enter chemical formula with spaces between elements. Examples:\n- Sr Ti O3 (strontium titanate)\n- Ca C O3 (calcium carbonate)\n- Al2 O3 (alumina)"
                    )

                elif search_mode == "Search Mineral":
                    mineral_options = []
                    mineral_mapping = {}

                    for space_group, minerals in MINERALS.items():
                        for mineral_name, formula in minerals.items():
                            option_text = f"{mineral_name} - SG #{space_group}"
                            mineral_options.append(option_text)
                            mineral_mapping[option_text] = {
                                'space_group': space_group,
                                'formula': formula,
                                'mineral_name': mineral_name
                            }

                    # Sort mineral options alphabetically
                    mineral_options.sort()

                    selected_mineral = st.selectbox(
                        "Select Mineral Structure:",
                        options=mineral_options,
                        help="Choose a mineral structure type. The exact formula and space group will be automatically set.",
                        index=2
                    )

                    if selected_mineral:
                        mineral_info = mineral_mapping[selected_mineral]

                        # col_mineral1, col_mineral2 = st.columns(2)
                        # with col_mineral1:
                        sg_symbol = get_space_group_info(mineral_info['space_group'])
                        st.info(
                            f"**Structure:** {mineral_info['mineral_name']}, **Space Group:** {mineral_info['space_group']} ({sg_symbol}), "
                            f"**Formula:** {mineral_info['formula']}")

                        space_group_number = mineral_info['space_group']
                        formula_input = mineral_info['formula']

                        st.success(
                            f"**Search will use:** Formula = {formula_input}, Space Group = {space_group_number}")

                show_element_info = st.checkbox("ℹ️ Show information about element groups")
                if show_element_info:
                    st.markdown("""
                    **Element groups note:**
                    **Common Elements (14):** H, C, N, O, F, Na, Mg, Al, Si, P, S, Cl, K, Ca  
                    **Transition Metals (10):** Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn  
                    **Alkali Metals (6):** Li, Na, K, Rb, Cs, Fr  
                    **Alkaline Earth (6):** Be, Mg, Ca, Sr, Ba, Ra  
                    **Noble Gases (6):** He, Ne, Ar, Kr, Xe, Rn  
                    **Halogens (5):** F, Cl, Br, I, At  
                    **Lanthanides (15):** La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu  
                    **Actinides (15):** Ac, Th, Pa, U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm, Md, No, Lr  
                    **Other Elements (51):** All remaining elements
                    """)

            if st.button("Search Selected Databases"):
                if not db_choices:
                    st.error("Please select at least one database to search.")
                else:
                    for db_choice in db_choices:
                        if db_choice == "Materials Project":
                            mp_limit = search_limits.get("Materials Project", 50)
                            with st.spinner(f"Searching **the MP database** (limit: {mp_limit}), please wait. 😊"):
                                try:
                                    with MPRester(MP_API_KEY) as mpr:
                                        docs = None

                                        if search_mode == "Elements":
                                            elements_list = [el.strip() for el in search_query.split() if el.strip()]
                                            if not elements_list:
                                                st.error("Please enter at least one element for the search.")
                                                continue
                                            elements_list_sorted = sorted(set(elements_list))
                                            docs = mpr.materials.summary.search(
                                                elements=elements_list_sorted,
                                                num_elements=len(elements_list_sorted),
                                                fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                            )

                                        elif search_mode == "Structure ID":
                                            mp_ids = [id.strip() for id in structure_ids.split('\n')
                                                      if id.strip() and id.strip().startswith('mp-')]
                                            if not mp_ids:
                                                st.warning(
                                                    "No valid Materials Project IDs found (should start with 'mp-')")
                                                continue
                                            docs = mpr.materials.summary.search(
                                                material_ids=mp_ids,
                                                fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                            )

                                        elif search_mode == "Space Group + Elements":
                                            elements_list = sorted(set(selected_elements))
                                            if not elements_list:
                                                st.warning(
                                                    "Please select elements for Materials Project space group search.")
                                                continue

                                            search_params = {
                                                "elements": elements_list,
                                                "num_elements": len(elements_list),
                                                "fields": ["material_id", "formula_pretty", "symmetry", "nsites",
                                                           "volume"],
                                                "spacegroup_number": space_group_number
                                            }

                                            docs = mpr.materials.summary.search(**search_params)

                                        elif search_mode == "Formula":
                                            if not formula_input.strip():
                                                st.warning(
                                                    "Please enter a chemical formula for Materials Project search.")
                                                continue

                                            # Convert space-separated format to compact format (Sr Ti O3 -> SrTiO3)
                                            clean_formula = formula_input.strip()
                                            if ' ' in clean_formula:
                                                parts = clean_formula.split()
                                                compact_formula = ''.join(parts)
                                            else:
                                                compact_formula = clean_formula

                                            docs = mpr.materials.summary.search(
                                                formula=compact_formula,
                                                fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                            )

                                        elif search_mode == "Search Mineral":
                                            if not selected_mineral:
                                                st.warning(
                                                    "Please select a mineral structure for Materials Project search.")
                                                continue
                                            clean_formula = formula_input.strip()
                                            if ' ' in clean_formula:
                                                parts = clean_formula.split()
                                                compact_formula = ''.join(parts)
                                            else:
                                                compact_formula = clean_formula

                                            # Search by formula and space group
                                            docs = mpr.materials.summary.search(
                                                formula=compact_formula,
                                                spacegroup_number=space_group_number,
                                                fields=["material_id", "formula_pretty", "symmetry", "nsites", "volume"]
                                            )

                                        if docs:
                                            status_placeholder = st.empty()
                                            st.session_state.mp_options = []
                                            st.session_state.full_structures_see = {}
                                            limited_docs = docs[:mp_limit]

                                            for doc in limited_docs:
                                                full_structure = mpr.get_structure_by_material_id(doc.material_id,
                                                                                                  conventional_unit_cell=True)
                                                st.session_state.full_structures_see[doc.material_id] = full_structure
                                                lattice = full_structure.lattice
                                                leng = len(full_structure)
                                                lattice_str = (f"{lattice.a:.3f} {lattice.b:.3f} {lattice.c:.3f} Å, "
                                                               f"{lattice.alpha:.1f}, {lattice.beta:.1f}, {lattice.gamma:.1f} °")
                                                st.session_state.mp_options.append(
                                                    f"{doc.material_id}: {doc.formula_pretty} ({doc.symmetry.symbol} #{doc.symmetry.number}) [{lattice_str}], {float(doc.volume):.1f} Å³, {leng} atoms"
                                                )
                                                status_placeholder.markdown(
                                                    f"- **Structure loaded:** `{full_structure.composition.reduced_formula}` ({doc.material_id})"
                                                )
                                            if len(limited_docs) < len(docs):
                                                st.info(
                                                    f"Showing first {mp_limit} of {len(docs)} total Materials Project results. Increase limit to see more.")
                                            st.success(
                                                f"Found {len(st.session_state.mp_options)} structures in Materials Project.")
                                        else:
                                            st.session_state.mp_options = []
                                            st.warning("No matching structures found in Materials Project.")
                                except Exception as e:
                                    st.error(f"An error occurred with Materials Project: {e}")

                        elif db_choice == "AFLOW":
                            aflow_limit = search_limits.get("AFLOW", 50)
                            with st.spinner(f"Searching **the AFLOW database** (limit: {aflow_limit}), please wait. 😊"):
                                try:
                                    results = []

                                    if search_mode == "Elements":
                                        elements_list = [el.strip() for el in search_query.split() if el.strip()]
                                        if not elements_list:
                                            st.warning("Please enter elements for AFLOW search.")
                                            continue
                                        ordered_elements = sorted(elements_list)
                                        ordered_str = ",".join(ordered_elements)
                                        aflow_nspecies = len(ordered_elements)

                                        results = list(
                                            search(catalog="icsd")
                                            .filter(
                                                (AFLOW_K.species % ordered_str) & (AFLOW_K.nspecies == aflow_nspecies))
                                            .select(
                                                AFLOW_K.auid,
                                                AFLOW_K.compound,
                                                AFLOW_K.geometry,
                                                AFLOW_K.spacegroup_relax,
                                                AFLOW_K.aurl,
                                                AFLOW_K.files,
                                            )
                                        )

                                    elif search_mode == "Structure ID":
                                        aflow_auids = []
                                        for id_line in structure_ids.split('\n'):
                                            id_line = id_line.strip()
                                            if id_line.startswith('aflow:'):
                                                auid = id_line.replace('aflow:', '').strip()
                                                aflow_auids.append(auid)

                                        if not aflow_auids:
                                            st.warning("No valid AFLOW AUIDs found (should start with 'aflow:')")
                                            continue

                                        results = []
                                        for auid in aflow_auids:
                                            try:
                                                result = list(search(catalog="icsd")
                                                              .filter(AFLOW_K.auid == f"aflow:{auid}")
                                                              .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                                      AFLOW_K.spacegroup_relax, AFLOW_K.aurl,
                                                                      AFLOW_K.files))
                                                results.extend(result)
                                            except Exception as e:
                                                st.warning(f"AFLOW search failed for AUID '{auid}': {e}")
                                                continue

                                    elif search_mode == "Space Group + Elements":
                                        if not selected_elements:
                                            st.warning("Please select elements for AFLOW space group search.")
                                            continue
                                        ordered_elements = sorted(selected_elements)
                                        ordered_str = ",".join(ordered_elements)
                                        aflow_nspecies = len(ordered_elements)

                                        try:
                                            results = list(search(catalog="icsd")
                                                           .filter((AFLOW_K.species % ordered_str) &
                                                                   (AFLOW_K.nspecies == aflow_nspecies) &
                                                                   (AFLOW_K.spacegroup_relax == space_group_number))
                                                           .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                                   AFLOW_K.spacegroup_relax, AFLOW_K.aurl,
                                                                   AFLOW_K.files))
                                        except Exception as e:
                                            st.warning(f"AFLOW space group search failed: {e}")
                                            results = []


                                    elif search_mode == "Formula":

                                        if not formula_input.strip():
                                            st.warning("Please enter a chemical formula for AFLOW search.")

                                            continue

                                        def convert_to_aflow_formula(formula_input):

                                            import re

                                            formula_parts = formula_input.strip().split()

                                            elements_dict = {}

                                            for part in formula_parts:

                                                match = re.match(r'([A-Z][a-z]?)(\d*)', part)

                                                if match:
                                                    element = match.group(1)

                                                    count = match.group(2) if match.group(
                                                        2) else "1"  # Add "1" if no number

                                                    elements_dict[element] = count

                                            aflow_parts = []

                                            for element in sorted(elements_dict.keys()):
                                                aflow_parts.append(f"{element}{elements_dict[element]}")

                                            return "".join(aflow_parts)

                                        # Generate 2x multiplied formula
                                        def multiply_formula_by_2(formula_input):

                                            import re

                                            formula_parts = formula_input.strip().split()

                                            elements_dict = {}

                                            for part in formula_parts:

                                                match = re.match(r'([A-Z][a-z]?)(\d*)', part)

                                                if match:
                                                    element = match.group(1)

                                                    count = int(match.group(2)) if match.group(2) else 1

                                                    elements_dict[element] = str(count * 2)  # Multiply by 2

                                            aflow_parts = []

                                            for element in sorted(elements_dict.keys()):
                                                aflow_parts.append(f"{element}{elements_dict[element]}")

                                            return "".join(aflow_parts)

                                        aflow_formula = convert_to_aflow_formula(formula_input)

                                        aflow_formula_2x = multiply_formula_by_2(formula_input)

                                        if aflow_formula_2x != aflow_formula:

                                            results = list(search(catalog="icsd")

                                                           .filter((AFLOW_K.compound == aflow_formula) |

                                                                   (AFLOW_K.compound == aflow_formula_2x))

                                                           .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,

                                                                   AFLOW_K.spacegroup_relax, AFLOW_K.aurl,
                                                                   AFLOW_K.files))

                                            st.info(
                                                f"Searching for both {aflow_formula} and {aflow_formula_2x} formulas simultaneously")

                                        else:
                                            results = list(search(catalog="icsd")
                                                           .filter(AFLOW_K.compound == aflow_formula)
                                                           .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                                   AFLOW_K.spacegroup_relax, AFLOW_K.aurl,
                                                                   AFLOW_K.files))

                                            st.info(f"Searching for formula {aflow_formula}")


                                    elif search_mode == "Search Mineral":
                                        if not selected_mineral:
                                            st.warning("Please select a mineral structure for AFLOW search.")
                                            continue

                                        def convert_to_aflow_formula_mineral(formula_input):
                                            import re
                                            formula_parts = formula_input.strip().split()
                                            elements_dict = {}
                                            for part in formula_parts:

                                                match = re.match(r'([A-Z][a-z]?)(\d*)', part)
                                                if match:
                                                    element = match.group(1)

                                                    count = match.group(2) if match.group(
                                                        2) else "1"  # Always add "1" for single atoms

                                                    elements_dict[element] = count

                                            aflow_parts = []

                                            for element in sorted(elements_dict.keys()):
                                                aflow_parts.append(f"{element}{elements_dict[element]}")

                                            return "".join(aflow_parts)

                                        def multiply_mineral_formula_by_2(formula_input):

                                            import re

                                            formula_parts = formula_input.strip().split()

                                            elements_dict = {}

                                            for part in formula_parts:
                                                match = re.match(r'([A-Z][a-z]?)(\d*)', part)
                                                if match:
                                                    element = match.group(1)
                                                    count = int(match.group(2)) if match.group(2) else 1
                                                    elements_dict[element] = str(count * 2)  # Multiply by 2
                                            aflow_parts = []
                                            for element in sorted(elements_dict.keys()):
                                                aflow_parts.append(f"{element}{elements_dict[element]}")
                                            return "".join(aflow_parts)

                                        aflow_formula = convert_to_aflow_formula_mineral(formula_input)

                                        aflow_formula_2x = multiply_mineral_formula_by_2(formula_input)

                                        # Search for both formulas with space group constraint in a single query

                                        if aflow_formula_2x != aflow_formula:
                                            results = list(search(catalog="icsd")
                                                           .filter(((AFLOW_K.compound == aflow_formula) |
                                                                    (AFLOW_K.compound == aflow_formula_2x)) &
                                                                   (AFLOW_K.spacegroup_relax == space_group_number))
                                                           .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                                   AFLOW_K.spacegroup_relax, AFLOW_K.aurl,
                                                                   AFLOW_K.files))

                                            st.info(
                                                f"Searching {mineral_info['mineral_name']} for both {aflow_formula} and {aflow_formula_2x} with space group {space_group_number}")

                                        else:
                                            results = list(search(catalog="icsd")
                                                           .filter((AFLOW_K.compound == aflow_formula) &
                                                                   (AFLOW_K.spacegroup_relax == space_group_number))
                                                           .select(AFLOW_K.auid, AFLOW_K.compound, AFLOW_K.geometry,
                                                                   AFLOW_K.spacegroup_relax, AFLOW_K.aurl,
                                                                   AFLOW_K.files))

                                            st.info(
                                                f"Searching {mineral_info['mineral_name']} for formula {aflow_formula} with space group {space_group_number}")

                                    if results:
                                        status_placeholder = st.empty()
                                        st.session_state.aflow_options = []
                                        st.session_state.entrys = {}

                                        limited_results = results[:aflow_limit]

                                        for entry in limited_results:
                                            st.session_state.entrys[entry.auid] = entry
                                            st.session_state.aflow_options.append(
                                                f"{entry.auid}: {entry.compound} ({entry.spacegroup_relax}) {entry.geometry}"
                                            )
                                            status_placeholder.markdown(
                                                f"- **Structure loaded:** `{entry.compound}` (aflow_{entry.auid})"
                                            )
                                        if len(limited_results) < len(results):
                                            st.info(
                                                f"Showing first {aflow_limit} of {len(results)} total AFLOW results. Increase limit to see more.")
                                        st.success(f"Found {len(st.session_state.aflow_options)} structures in AFLOW.")
                                    else:
                                        st.session_state.aflow_options = []
                                        st.warning("No matching structures found in AFLOW.")
                                except Exception as e:
                                    st.warning(f"No matching structures found in AFLOW.")
                                    st.session_state.aflow_options = []

                        elif db_choice == "COD":
                            cod_limit = search_limits.get("COD", 50)
                            with st.spinner(f"Searching **the COD database** (limit: {cod_limit}), please wait. 😊"):
                                try:
                                    cod_entries = []

                                    if search_mode == "Elements":
                                        elements = [el.strip() for el in search_query.split() if el.strip()]
                                        if elements:
                                            params = {'format': 'json', 'detail': '1'}
                                            for i, el in enumerate(elements, start=1):
                                                params[f'el{i}'] = el
                                            params['strictmin'] = str(len(elements))
                                            params['strictmax'] = str(len(elements))
                                            cod_entries = get_cod_entries(params)
                                        else:
                                            st.warning("Please enter elements for COD search.")
                                            continue

                                    elif search_mode == "Structure ID":
                                        cod_ids = []
                                        for id_line in structure_ids.split('\n'):
                                            id_line = id_line.strip()
                                            if id_line.startswith('cod_'):
                                                # Extract numeric ID from cod_XXXXX format
                                                numeric_id = id_line.replace('cod_', '').strip()
                                                if numeric_id.isdigit():
                                                    cod_ids.append(numeric_id)

                                        if not cod_ids:
                                            st.warning(
                                                "No valid COD IDs found (should start with 'cod_' followed by numbers)")
                                            continue

                                        cod_entries = []
                                        for cod_id in cod_ids:
                                            try:
                                                params = {'format': 'json', 'detail': '1', 'id': cod_id}
                                                entry = get_cod_entries(params)
                                                if entry:
                                                    if isinstance(entry, list):
                                                        cod_entries.extend(entry)
                                                    else:
                                                        cod_entries.append(entry)
                                            except Exception as e:
                                                st.warning(f"COD search failed for ID {cod_id}: {e}")
                                                continue

                                    elif search_mode == "Space Group + Elements":
                                        elements = selected_elements
                                        if elements:
                                            params = {'format': 'json', 'detail': '1'}
                                            for i, el in enumerate(elements, start=1):
                                                params[f'el{i}'] = el
                                            params['strictmin'] = str(len(elements))
                                            params['strictmax'] = str(len(elements))
                                            params['space_group_number'] = str(space_group_number)

                                            cod_entries = get_cod_entries(params)
                                        else:
                                            st.warning("Please select elements for COD space group search.")
                                            continue

                                    elif search_mode == "Formula":
                                        if not formula_input.strip():
                                            st.warning("Please enter a chemical formula for COD search.")
                                            continue

                                        # alphabet sorting
                                        alphabet_form = sort_formula_alphabetically(formula_input)
                                        print(alphabet_form)
                                        params = {'format': 'json', 'detail': '1', 'formula': alphabet_form}
                                        cod_entries = get_cod_entries(params)

                                    elif search_mode == "Search Mineral":
                                        if not selected_mineral:
                                            st.warning("Please select a mineral structure for COD search.")
                                            continue

                                        # Use both formula and space group for COD search
                                        alphabet_form = sort_formula_alphabetically(formula_input)
                                        params = {
                                            'format': 'json',
                                            'detail': '1',
                                            'formula': alphabet_form,
                                            'space_group_number': str(space_group_number)
                                        }
                                        cod_entries = get_cod_entries(params)

                                    if cod_entries and isinstance(cod_entries, list):
                                        st.session_state.cod_options = []
                                        st.session_state.full_structures_see_cod = {}
                                        status_placeholder = st.empty()
                                        limited_entries = cod_entries[:cod_limit]
                                        errors = []

                                        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                                            future_to_entry = {executor.submit(fetch_and_parse_cod_cif, entry): entry
                                                               for
                                                               entry in limited_entries}

                                            processed_count = 0
                                            for future in concurrent.futures.as_completed(future_to_entry):
                                                processed_count += 1
                                                status_placeholder.markdown(
                                                    f"- **Processing:** {processed_count}/{len(limited_entries)} entries...")
                                                try:
                                                    cod_id, structure, entry_data, error = future.result()
                                                    if error:
                                                        original_entry = future_to_entry[future]
                                                        errors.append(
                                                            f"Entry `{original_entry.get('file', 'N/A')}` failed: {error}")
                                                        continue  # Skip to the next completed future
                                                    if cod_id and structure and entry_data:
                                                        st.session_state.full_structures_see_cod[cod_id] = structure

                                                        spcs = entry_data.get("sg", "Unknown")
                                                        spcs_number = entry_data.get("sgNumber", "Unknown")
                                                        cell_volume = structure.lattice.volume
                                                        option_str = (
                                                            f"{cod_id}: {structure.composition.reduced_formula} ({spcs} #{spcs_number}) [{structure.lattice.a:.3f} {structure.lattice.b:.3f} {structure.lattice.c:.3f} Å, {structure.lattice.alpha:.2f}, "
                                                            f"{structure.lattice.beta:.2f}, {structure.lattice.gamma:.2f}°], {cell_volume:.1f} Å³, {len(structure)} atoms"
                                                        )
                                                        st.session_state.cod_options.append(option_str)

                                                except Exception as e:
                                                    errors.append(
                                                        f"A critical error occurred while processing a result: {e}")
                                        status_placeholder.empty()
                                        if st.session_state.cod_options:
                                            if len(limited_entries) < len(cod_entries):
                                                st.info(
                                                    f"Showing first {cod_limit} of {len(cod_entries)} total COD results. Increase limit to see more.")
                                            st.success(
                                                f"Found and processed {len(st.session_state.cod_options)} structures from COD.")
                                        else:
                                            st.warning("COD: No matching structures could be successfully processed.")
                                        if errors:
                                            st.error(f"Encountered {len(errors)} error(s) during the search.")
                                            with st.container(border=True):
                                                for e in errors:
                                                    st.warning(e)
                                    else:
                                        st.session_state.cod_options = []
                                        st.warning("COD: No matching structures found.")
                                except Exception as e:
                                    st.warning(f"COD search error: {e}")
                                    st.session_state.cod_options = []

            # with cols2:
            #     image = Image.open("images/Rabbit2.png")
            #     st.image(image, use_container_width=True)

            with cols3:
                if any(x in st.session_state for x in ['mp_options', 'aflow_options', 'cod_options']):
                    tabs = []
                    if 'mp_options' in st.session_state and st.session_state.mp_options:
                        tabs.append("Materials Project")
                    if 'aflow_options' in st.session_state and st.session_state.aflow_options:
                        tabs.append("AFLOW")
                    if 'cod_options' in st.session_state and st.session_state.cod_options:
                        tabs.append("COD")

                    if tabs:
                        selected_tab = st.tabs(tabs)

                        tab_index = 0
                        if 'mp_options' in st.session_state and st.session_state.mp_options:
                            with selected_tab[tab_index]:
                                st.subheader("🧬 Structures Found in Materials Project")
                                selected_structure = st.selectbox("Select a structure from MP:",
                                                                  st.session_state.mp_options)
                                selected_id = selected_structure.split(":")[0].strip()
                                composition = selected_structure.split(":", 1)[1].split("(")[0].strip()
                                file_name = f"{selected_id}_{composition}.cif"
                                file_name = re.sub(r'[\\/:"*?<>|]+', '_', file_name)

                                if selected_id in st.session_state.full_structures_see:
                                    selected_entry = st.session_state.full_structures_see[selected_id]

                                    conv_lattice = selected_entry.lattice
                                    cell_volume = selected_entry.lattice.volume
                                    density = str(selected_entry.density).split()[0]
                                    n_atoms = len(selected_entry)
                                    atomic_den = n_atoms / cell_volume

                                    structure_type = identify_structure_type(selected_entry)
                                    st.write(f"**Structure type:** {structure_type}")
                                    analyzer = SpacegroupAnalyzer(selected_entry)
                                    st.write(
                                        f"**Space Group:** {analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})")

                                    st.write(
                                        f"**Material ID:** {selected_id}, **Formula:** {composition}, N. of Atoms {n_atoms}")

                                    st.write(
                                        f"**Conventional Lattice:** a = {conv_lattice.a:.4f} Å, b = {conv_lattice.b:.4f} Å, c = {conv_lattice.c:.4f} Å, α = {conv_lattice.alpha:.1f}°, β = {conv_lattice.beta:.1f}°, γ = {conv_lattice.gamma:.1f}° (Volume {cell_volume:.1f} Å³)")
                                    st.write(f"**Density:** {float(density):.2f} g/cm³ ({atomic_den:.4f} 1/Å³)")

                                    mp_url = f"https://materialsproject.org/materials/{selected_id}"
                                    st.write(f"**Link:** {mp_url}")

                                    col_mpd, col_mpb = st.columns([2, 1])
                                    with col_mpd:
                                        if st.button("Add Selected Structure (MP)", key="add_btn_mp"):
                                            pmg_structure = st.session_state.full_structures_see[selected_id]
                                            #check_structure_size_and_warn(pmg_structure, f"MP structure {selected_id}")
                                            st.session_state.full_structures[file_name] = pmg_structure
                                            cif_writer = CifWriter(pmg_structure)
                                            cif_content = cif_writer.__str__()
                                            cif_file = io.BytesIO(cif_content.encode('utf-8'))
                                            cif_file.name = file_name
                                            if 'uploaded_files' not in st.session_state:
                                                st.session_state.uploaded_files = []
                                            if all(f.name != file_name for f in st.session_state.uploaded_files):
                                                st.session_state.uploaded_files.append(cif_file)
                                            st.success("Structure added from Materials Project!")
                                    with col_mpb:
                                        st.download_button(
                                            label="Download MP CIF",
                                            data=str(
                                                CifWriter(st.session_state.full_structures_see[selected_id],
                                                          symprec=0.01)),
                                            file_name=file_name,
                                            type="primary",
                                            mime="chemical/x-cif"
                                        )
                                st.info(
                                    f"**Note**: If H element is missing in CIF file, it is not shown in the formula either.")
                            tab_index += 1

                        if 'aflow_options' in st.session_state and st.session_state.aflow_options:
                            with selected_tab[tab_index]:
                                st.subheader("🧬 Structures Found in AFLOW")
                                st.warning(
                                    "The AFLOW does not provide atomic occupancies and includes only information about primitive cell in API. For better performance, volume and n. of atoms are purposely omitted from the expander.")
                                selected_structure = st.selectbox("Select a structure from AFLOW:",
                                                                  st.session_state.aflow_options)
                                selected_auid = selected_structure.split(": ")[0].strip()
                                selected_entry = next(
                                    (entry for entry in st.session_state.entrys.values() if
                                     entry.auid == selected_auid),
                                    None)
                                if selected_entry:

                                    cif_files = [f for f in selected_entry.files if
                                                 f.endswith("_sprim.cif") or f.endswith(".cif")]

                                    if cif_files:

                                        cif_filename = cif_files[0]

                                        # Correct the AURL: replace the first ':' with '/'

                                        host_part, path_part = selected_entry.aurl.split(":", 1)

                                        corrected_aurl = f"{host_part}/{path_part}"

                                        file_url = f"http://{corrected_aurl}/{cif_filename}"
                                        response = requests.get(file_url)
                                        cif_content = response.content

                                        structure_from_aflow = Structure.from_str(cif_content.decode('utf-8'),
                                                                                  fmt="cif")
                                        converted_structure = get_full_conventional_structure(structure_from_aflow,
                                                                                              symprec=0.1)

                                        conv_lattice = converted_structure.lattice
                                        cell_volume = converted_structure.lattice.volume
                                        density = str(converted_structure.density).split()[0]
                                        n_atoms = len(converted_structure)
                                        atomic_den = n_atoms / cell_volume

                                        structure_type = identify_structure_type(converted_structure)
                                        st.write(f"**Structure type:** {structure_type}")
                                        analyzer = SpacegroupAnalyzer(structure_from_aflow)
                                        st.write(
                                            f"**Space Group:** {analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})")
                                        st.write(
                                            f"**AUID:** {selected_entry.auid}, **Formula:** {selected_entry.compound}, **N. of Atoms:** {n_atoms}")
                                        st.write(
                                            f"**Conventional Lattice:** a = {conv_lattice.a:.4f} Å, b = {conv_lattice.b:.4f} Å, c = {conv_lattice.c:.4f} Å, α = {conv_lattice.alpha:.1f}°, β = {conv_lattice.beta:.1f}°, "
                                            f"γ = {conv_lattice.gamma:.1f}° (Volume {cell_volume:.1f} Å³)")
                                        st.write(f"**Density:** {float(density):.2f} g/cm³ ({atomic_den:.4f} 1/Å³)")

                                        linnk = f"https://aflowlib.duke.edu/search/ui/material/?id=" + selected_entry.auid
                                        st.write("**Link:**", linnk)

                                        if st.button("Add Selected Structure (AFLOW)", key="add_btn_aflow"):
                                            if 'uploaded_files' not in st.session_state:
                                                st.session_state.uploaded_files = []
                                            cif_file = io.BytesIO(cif_content)
                                            cif_file.name = f"{selected_entry.compound}_{selected_entry.auid}.cif"

                                            st.session_state.full_structures[cif_file.name] = structure_from_aflow

                                            #check_structure_size_and_warn(structure_from_aflow, cif_file.name)
                                            if all(f.name != cif_file.name for f in st.session_state.uploaded_files):
                                                st.session_state.uploaded_files.append(cif_file)
                                            st.success("Structure added from AFLOW!")

                                        st.download_button(
                                            label="Download AFLOW CIF",
                                            data=cif_content,
                                            file_name=f"{selected_entry.compound}_{selected_entry.auid}.cif",
                                            type="primary",
                                            mime="chemical/x-cif"
                                        )
                                        st.info(
                                            f"**Note**: If H element is missing in CIF file, it is not shown in the formula either.")
                                    else:
                                        st.warning("No CIF file found for this AFLOW entry.")
                            tab_index += 1

                        # COD tab
                        if 'cod_options' in st.session_state and st.session_state.cod_options:
                            with selected_tab[tab_index]:
                                st.subheader("🧬 Structures Found in COD")
                                selected_cod_structure = st.selectbox(
                                    "Select a structure from COD:",
                                    st.session_state.cod_options,
                                    key='sidebar_select_cod'
                                )
                                cod_id = selected_cod_structure.split(":")[0].strip()
                                if cod_id in st.session_state.full_structures_see_cod:
                                    selected_entry = st.session_state.full_structures_see_cod[cod_id]
                                    lattice = selected_entry.lattice
                                    cell_volume = selected_entry.lattice.volume
                                    density = str(selected_entry.density).split()[0]
                                    n_atoms = len(selected_entry)
                                    atomic_den = n_atoms / cell_volume

                                    idcodd = cod_id.removeprefix("cod_")

                                    structure_type = identify_structure_type(selected_entry)
                                    st.write(f"**Structure type:** {structure_type}")
                                    analyzer = SpacegroupAnalyzer(selected_entry)
                                    st.write(
                                        f"**Space Group:** {analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})")

                                    st.write(
                                        f"**COD ID:** {idcodd}, **Formula:** {selected_entry.composition.reduced_formula}, **N. of Atoms:** {n_atoms}")
                                    st.write(
                                        f"**Conventional Lattice:** a = {lattice.a:.3f} Å, b = {lattice.b:.3f} Å, c = {lattice.c:.3f} Å, α = {lattice.alpha:.2f}°, β = {lattice.beta:.2f}°, γ = {lattice.gamma:.2f}° (Volume {cell_volume:.1f} Å³)")
                                    st.write(f"**Density:** {float(density):.2f} g/cm³ ({atomic_den:.4f} 1/Å³)")

                                    cod_url = f"https://www.crystallography.net/cod/{cod_id.split('_')[1]}.html"
                                    st.write(f"**Link:** {cod_url}")

                                    file_name = f"{selected_entry.composition.reduced_formula}_COD_{cod_id.split('_')[1]}.cif"

                                    if st.button("Add Selected Structure (COD)", key="sid_add_btn_cod"):
                                        cif_writer = CifWriter(selected_entry, symprec=0.01)
                                        cif_data = str(cif_writer)
                                        st.session_state.full_structures[file_name] = selected_entry
                                        cif_file = io.BytesIO(cif_data.encode('utf-8'))
                                        cif_file.name = file_name
                                        if 'uploaded_files' not in st.session_state:
                                            st.session_state.uploaded_files = []
                                        if all(f.name != file_name for f in st.session_state.uploaded_files):
                                            st.session_state.uploaded_files.append(cif_file)

                                        #check_structure_size_and_warn(selected_entry, file_name)
                                        st.success("Structure added from COD!")

                                    st.download_button(
                                        label="Download COD CIF",
                                        data=str(CifWriter(selected_entry, symprec=0.01)),
                                        file_name=file_name,
                                        mime="chemical/x-cif", type="primary",
                                    )
                                    st.info(
                                        f"**Note**: If H element is missing in CIF file, it is not shown in the formula either.")











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

            st.subheader("1️⃣ Step 1: Select ICET SQS Method")
            colzz, colss = st.columns([1, 1])
            with colzz:

                sqs_method = st.radio(
                    "ICET SQS Method:",
                    ["Supercell-Specific", "Maximum n. of atoms (Not yet implemented)",
                     "Enumeration (For max 24 atoms, Not yet implemented)"],
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
                "Enumeration (For max 24 atoms, Not yet implemented)": "enumeration"
            }
            internal_method = method_map[sqs_method]
            colsz, colsc = st.columns(2)
            with colsc:
                if internal_method != "enumeration":
                    n_steps = st.number_input(
                        f"📌 Number of Monte Carlo **steps**:",
                        min_value=1000,
                        max_value=10000000,
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
                        num_runs = st.number_input(f"📌 Number of runs:", min_value=2, max_value=100, value=5, step=1)
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
                    index=0,
                    key="composition_mode_radio",
                    help="Global: Specify overall composition. Sublattice: Control each atomic position separately."
                )
            with colb2:
                with st.expander("ℹ️ Composition Mode Details", expanded=False):
                    st.markdown("""
                    ##### 🔄 Global Composition
                    - Specify the target composition for the entire structure (e.g., 50% Fe, 50% Ni)
                    - All crystallographic sites can be occupied by any of the selected elements
                    - Elements are distributed randomly throughout the structure according to the specified fractions
                    - **Example:** Fe₀.₅Ni₀.₅ random alloy where Fe and Ni atoms can occupy any position

                    ---

                    ##### 🎯 Sublattice-Specific  
                    - Control which elements can occupy specific crystallographic sites (Wyckoff positions)
                    - Set different compositions for different atomic sublattices
                    - **Example:** In a perovskite ABO₃, control A-site (Ba/Sr) and B-site (Ti/Zr) compositions independently

                    ---
                    **Global** treats all sites equally, while **Sublattice-Specific** allows site-dependent element distributions based on crystallographic positions.

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
                nx = st.number_input("x-axis multiplier", value=1, min_value=1, max_value=10, step=1,
                                     key="nx_global")
            with col_y:
                ny = st.number_input("y-axis multiplier", value=1, min_value=1, max_value=10, step=1,
                                     key="ny_global")
            with col_z:
                nz = st.number_input("z-axis multiplier", value=1, min_value=1, max_value=10, step=1,
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
                if len(element_list) == 0:
                    st.error("You must select at least one element.")
                    st.stop()
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
                element_list = [2,2]
                composition_input = []
                chem_symbols, target_concentrations, otrs = render_site_sublattice_selector(working_structure,
                                                                                            all_sites)

            if composition_mode == "🔄 Global Composition":
                try:
                    total_sites = len(supercell_preview)
                    st.write("**Target vs Achievable Concentrations:**")

                    conc_data = []
                    total_element_counts = {}  # Store for the visual cards

                    for element, target_frac in target_concentrations.items():
                        target_count = target_frac * total_sites
                        achievable_count = int(round(target_count))
                        achievable_frac = achievable_count / total_sites

                        # Store the counts for the visual display
                        total_element_counts[element] = achievable_count

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

                    # Add the colorful element distribution cards
                    if total_element_counts:
                        st.write("#### **Overall Expected Element Distribution in Supercell:**")

                        cols = st.columns(min(len(total_element_counts), 4))
                        for i, (elem, count) in enumerate(sorted(total_element_counts.items())):
                            percentage = (count / total_sites) * 100 if total_sites > 0 else 0
                            with cols[i % len(cols)]:
                                if percentage >= 80:
                                    color = "#2E4057"  # Dark Blue-Gray
                                elif percentage >= 60:
                                    color = "#4A6741"  # Dark Forest Green
                                elif percentage >= 40:
                                    color = "#6B73FF"  # Purple-Blue
                                elif percentage >= 25:
                                    color = "#FF8C00"  # Dark Orange
                                elif percentage >= 15:
                                    color = "#4ECDC4"  # Teal
                                elif percentage >= 10:
                                    color = "#45B7D1"  # Blue
                                elif percentage >= 5:
                                    color = "#96CEB4"  # Green
                                elif percentage >= 2:
                                    color = "#FECA57"  # Yellow
                                elif percentage >= 1:
                                    color = "#DDA0DD"  # Plum
                                else:
                                    color = "#D3D3D3"  # Light Gray

                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, {color}, {color}CC);
                                    padding: 20px;
                                    border-radius: 15px;
                                    text-align: center;
                                    margin: 10px 0;
                                    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
                                    border: 2px solid rgba(255,255,255,0.2);
                                ">
                                    <h1 style="
                                        color: white;
                                        font-size: 3em;
                                        margin: 0;
                                        text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
                                        font-weight: bold;
                                    ">{elem}</h1>
                                    <h2 style="
                                        color: white;
                                        font-size: 2em;
                                        margin: 10px 0 0 0;
                                        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
                                    ">{percentage:.1f}%</h2>
                                    <p style="
                                        color: white;
                                        font-size: 1.8em;
                                        margin: 5px 0 0 0;
                                        opacity: 0.9;
                                    ">{int(count)} atoms</p>
                                </div>
                                """, unsafe_allow_html=True)

                        st.write(f"**Total expected atoms in supercell:** {total_sites}")

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
                elif not len(element_list) > 1:
                    st.warning(f"Select atleast two elements first in 4️⃣ Step 4:")
                    run_multi_sqs = st.button(" Generate Multiple SQS Structures", type="tertiary", disabled = True,
                                              help = "Select atleast two elements first.")
                else:
                    run_multi_sqs = st.button(" Generate Multiple SQS Structures", type="tertiary")

                if run_multi_sqs:
                    time.sleep(1)
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
                    </style>
                """, unsafe_allow_html=True)

                if not target_concentrations:
                    st.warning("Create atleast 1 sublattice (with minimum of two elements) first in 4️⃣ Step 4.")
                    run_sqs = st.button("Generate SQS Structure", type="tertiary", disabled = True,
                                        help = "Create atleast 1 sublattice (with minimum of two elements) first.")
                elif not len(element_list) > 1:
                    st.warning(f"Select atleast two elements first in 4️⃣ Step 4:")
                    run_sqs = st.button("Generate SQS Structure", type="tertiary", disabled = True,
                                        help = "Select atleast two elements.")
                else:
                    run_sqs = st.button("Generate SQS Structure", type="tertiary")
                if "sqs_results" not in st.session_state:
                    st.session_state.sqs_results = {}

                config_str = str(target_concentrations)
                current_config_key = f"{selected_sqs_file}_{reduce_to_primitive}_{nx}_{ny}_{nz}_{config_str}_{internal_method}_{n_steps}_{random_seed}"

                if run_sqs:
                    time.sleep(1)
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

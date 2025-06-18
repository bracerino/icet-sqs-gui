import streamlit as st
st.set_page_config(
    layout="wide",  
    initial_sidebar_state="expanded",
    page_title="ICET Special Quasi-Random Structure (SQS) Generation", 
    page_icon="🧊" 
)


from pymatgen.core import Structure
import io
import os
import re
from pymatgen.io.cif import CifWriter
import numpy as np
import pandas as pd
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from st_trans import render_sqs_module, check_sqs_mode


# import pkg_resources
# installed_packages = sorted([(d.project_name, d.version) for d in pkg_resources.working_set])
# st.subheader("Installed Python Modules")
# for package, version in installed_packages:
#    st.write(f"{package}=={version}")

if 'full_structures' not in st.session_state:
    st.session_state.full_structures = {}

if 'uploaded_files' not in st.session_state or st.session_state['uploaded_files'] is None:
    st.session_state['uploaded_files'] = []

if 'previous_uploaded_files' not in st.session_state:
    st.session_state['previous_uploaded_files'] = []




def remove_fractional_occupancies_safely(structure):
    species = []
    coords = []

    for site in structure:
        if site.is_ordered:
            species.append(site.specie)
        else:
            dominant_sp = max(site.species.items(), key=lambda x: x[1])[0]
            species.append(dominant_sp)
        coords.append(site.frac_coords)

    ordered_structure = Structure(
        lattice=structure.lattice,
        species=species,
        coords=coords,
        coords_are_cartesian=False
    )

    return ordered_structure


# File uploader in the sidebar
st.write("Debug: Checking if sidebar code is reached")
uploaded_files_user_sidebar = st.sidebar.file_uploader(
    "Upload Structure Files (CIF, POSCAR, LMP, XSF, PW, CFG, ...):",
    type=None,
    accept_multiple_files=True,
    key="sidebar_uploader"
)

current_file_names = [file.name for file in uploaded_files_user_sidebar] if uploaded_files_user_sidebar else []
previous_file_names = [file.name for file in st.session_state['previous_uploaded_files']]

# Detect removed files
removed_files = set(previous_file_names) - set(current_file_names)

for removed_file in removed_files:
    if removed_file in st.session_state.full_structures:
        del st.session_state.full_structures[removed_file]
        st.success(f"Removed structure: {removed_file}")

st.session_state['uploaded_files'] = [
    file for file in st.session_state['uploaded_files']
    if file.name in current_file_names
]

import os
from ase.io import read
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.lammps.data import LammpsData
import streamlit as st


def load_structure(file):
    try:
        file_content = file.read()
        file.seek(0)  # Reset file pointer

        with open(file.name, "wb") as f:
            f.write(file_content)
        filename = file.name.lower()

        if filename.endswith(".cif"):
            mg_structure = Structure.from_file(file.name)
        elif filename.endswith(".data"):
            lmp_filename = file.name.replace(".data", ".lmp")
            os.rename(file.name, lmp_filename)
            lammps_data = LammpsData.from_file(lmp_filename, atom_style="atomic")
            mg_structure = lammps_data.structure
        elif filename.endswith(".lmp"):
            lammps_data = LammpsData.from_file(file.name, atom_style="atomic")
            mg_structure = lammps_data.structure
        else:
            atoms = read(file.name)
            mg_structure = AseAtomsAdaptor.get_structure(atoms)

        if os.path.exists(file.name):
            os.remove(file.name)

        return mg_structure

    except Exception as e:
        st.error(f"Failed to parse {file.name}: {e}")
        st.error(
            f"This does not work. Are you sure you tried to upload here the structure files (CIF, POSCAR, LMP, XSF, PW)? For the **experimental XY data**, put them to the other uploader\n"
            f"and please remove this wrongly placed file. 😊")
        raise e


def handle_uploaded_files(uploaded_files_user_sidebar):
    if uploaded_files_user_sidebar:
        for file in uploaded_files_user_sidebar:
            if file.name not in st.session_state.full_structures:
                try:
                    structure = load_structure(file)
                    st.session_state.full_structures[file.name] = structure
                    st.success(f"Successfully loaded structure: {file.name}")

                    # Add to uploaded_files list for tracking
                    if 'uploaded_files' not in st.session_state:
                        st.session_state['uploaded_files'] = []
                    if all(f.name != file.name for f in st.session_state['uploaded_files']):
                        st.session_state['uploaded_files'].append(file)

                except Exception as e:
                    pass


def update_file_upload_section():
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = []
    if 'previous_uploaded_files' not in st.session_state:
        st.session_state['previous_uploaded_files'] = []

    current_file_names = [file.name for file in uploaded_files_user_sidebar] if uploaded_files_user_sidebar else []
    previous_file_names = [file.name for file in st.session_state['previous_uploaded_files']]

    removed_files = set(previous_file_names) - set(current_file_names)
    for removed_file in removed_files:
        if removed_file in st.session_state.full_structures:
            del st.session_state.full_structures[removed_file]
            st.success(f"Removed structure: {removed_file}")
    st.session_state['uploaded_files'] = [
        file for file in st.session_state['uploaded_files']
        if file.name in current_file_names
    ]

    handle_uploaded_files(uploaded_files_user_sidebar)

    st.session_state['previous_uploaded_files'] = uploaded_files_user_sidebar if uploaded_files_user_sidebar else []

    if st.session_state.full_structures:
        st.sidebar.subheader("📋 Currently Loaded Structures")
        for filename in st.session_state.full_structures.keys():
            st.sidebar.text(f"• {filename}")

update_file_upload_section()

st.sidebar.info(f"❤️🫶 **[Donations always appreciated!](https://buymeacoffee.com/bracerino)**")
st.sidebar.info(
    "Try also the main application **[XRDlicious](xrdlicious.com)**, and the generation of **[SQS using ATAT mcsqs](https://atat-sqs.streamlit.app/)**. 🌀 Developed by **[IMPLANT team](https://implant.fs.cvut.cz/)**. 📺 (Quick tutorial (in July)). The **online version** may become **unstable** with high settings or heavy traffic. "
    "To ensure better performance, consider compiling the app **locally** on your computer from **GitHub**."
)

st.sidebar.link_button("GitHub page (for local compilation)", "https://github.com/bracerino/SQS-GUI.git",
                       type="primary")
update_file_upload_section()
st.session_state['previous_uploaded_files'] = uploaded_files_user_sidebar if uploaded_files_user_sidebar else []

# Render the SQS transformation module


# Call the SQS module
render_sqs_module()

st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
# def get_memory_usage():
#    process = psutil.Process(os.getpid())
#    mem_info = process.memory_info()
#    return mem_info.rss / (1024 ** 2)  # in MB


# memory_usage = get_memory_usage()
# st.write(
#    f"🔍 Current memory usage: **{memory_usage:.2f} MB**. We are now using free hosting by Streamlit Community Cloud servis, which has a limit for RAM memory of 2.6 GBs. For more extensive computations, please compile the application locally from the [GitHub](https://github.com/bracerino/xrdlicious).")
st.markdown("""
**The GUI SQS application is open-source and released under the [MIT License](https://github.com/bracerino/SQS-GUI/blob/main/LICENSE).**
""")

st.markdown("""

### Acknowledgments

This project uses several open-source tools and datasets. We gratefully acknowledge their authors: **[ICET](https://icet.materialsmodeling.org/)** Licensed under the [Mozilla Public License Version 2.0](https://gitlab.com/materials-modeling/icet/-/blob/master/LICENSE?ref_type=heads). **[Matminer](https://github.com/hackingmaterials/matminer)** Licensed under the [Modified BSD License](https://github.com/hackingmaterials/matminer/blob/main/LICENSE). **[Pymatgen](https://github.com/materialsproject/pymatgen)** Licensed under the [MIT License](https://github.com/materialsproject/pymatgen/blob/master/LICENSE).
 **[ASE (Atomic Simulation Environment)](https://gitlab.com/ase/ase)** Licensed under the [GNU Lesser General Public License (LGPL)](https://gitlab.com/ase/ase/-/blob/master/COPYING.LESSER). **[Py3DMol](https://github.com/avirshup/py3dmol/tree/master)** Licensed under the [BSD-style License](https://github.com/avirshup/py3dmol/blob/master/LICENSE.txt). **[Materials Project](https://next-gen.materialsproject.org/)** Data from the Materials Project is made available under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). **[AFLOW](http://aflow.org)** Licensed under the [GNU General Public License (GPL)](https://www.gnu.org/licenses/gpl-3.0.html)
 **[Crystallographic Open Database (COD)](https://www.crystallography.net/cod/)** under the CC0 license.
""")

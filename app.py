import streamlit as st
from pymatgen.core import Structure
import io
import os
import re
from pymatgen.io.cif import CifWriter
import numpy as np
import pandas as pd
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

st.set_page_config(layout="wide")

# Initialize session state variables
if 'full_structures' not in st.session_state:
    st.session_state.full_structures = {}

if 'uploaded_files' not in st.session_state or st.session_state['uploaded_files'] is None:
    st.session_state['uploaded_files'] = []

if 'previous_uploaded_files' not in st.session_state:
    st.session_state['previous_uploaded_files'] = []

# Function to load structure from file
def load_structure(file):
    """Load a structure file into a pymatgen Structure object"""
    try:
        file_content = file.read()
        file.seek(0)  # Reset file pointer

        # Create a temporary file to read the structure
        with open(file.name, "wb") as f:
            f.write(file_content)

        # Try to load the structure using pymatgen
        structure = Structure.from_file(file.name)

        # Clean up temporary file
        if os.path.exists(file.name):
            os.remove(file.name)

        return structure
    except Exception as e:
        st.error(f"Failed to parse {file.name}: {e}")
        raise e

# Function to safely remove fractional occupancies
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
st.sidebar.subheader("📁 Upload Your Structure Files")
uploaded_files_user_sidebar = st.sidebar.file_uploader(
    "Upload Structure Files (CIF, POSCAR, LMP, XSF, PW, CFG, ...):",
    type=None,
    accept_multiple_files=True,
    key="sidebar_uploader"
)

# Handle file removal detection and synchronization
current_file_names = [file.name for file in uploaded_files_user_sidebar] if uploaded_files_user_sidebar else []
previous_file_names = [file.name for file in st.session_state['previous_uploaded_files']]

# Detect removed files
removed_files = set(previous_file_names) - set(current_file_names)

# Remove structures for files that were removed from uploader
for removed_file in removed_files:
    if removed_file in st.session_state.full_structures:
        del st.session_state.full_structures[removed_file]
        st.success(f"Removed structure: {removed_file}")

# Update uploaded_files list to match current uploader state
st.session_state['uploaded_files'] = [
    file for file in st.session_state['uploaded_files']
    if file.name in current_file_names
]

# Process newly uploaded files
if uploaded_files_user_sidebar:
    for file in uploaded_files_user_sidebar:
        if file.name not in st.session_state.full_structures:
            try:
                structure = load_structure(file)
                st.session_state.full_structures[file.name] = structure

                # Add to uploaded_files list for tracking
                if all(f.name != file.name for f in st.session_state['uploaded_files']):
                    st.session_state['uploaded_files'].append(file)

                st.success(f"Successfully loaded structure: {file.name}")
            except Exception as e:
                st.error(
                    f"This does not work. Are you sure you tried to upload here the structure files (CIF, POSCAR, LMP, XSF, PW)? For the **experimental XY data**, put them to the other uploader\n"
                    f"and please remove this wrongly placed file. 😊")

# Update previous_uploaded_files for next comparison
st.session_state['previous_uploaded_files'] = uploaded_files_user_sidebar if uploaded_files_user_sidebar else []

# Optional: Display currently loaded structures
if st.session_state.full_structures:
    st.sidebar.subheader("📋 Currently Loaded Structures")
    for filename in st.session_state.full_structures.keys():
        st.sidebar.text(f"• {filename}")

# Render the SQS transformation module
from st_trans import render_sqs_module, check_sqs_mode

# Call the SQS module
render_sqs_module()

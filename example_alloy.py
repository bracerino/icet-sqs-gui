"""One-click example loader for the ICET SQS GUI.

Renders, on the landing page next to the "upload a structure" prompt, a small
crystal-structure selector (bcc / fcc / sc / hcp) plus a button that loads a
ready-made random alloy of that lattice and pre-fills every SQS parameter, so a
new user can see a complete workflow without uploading anything or tuning
settings.

Each option is a real (or, for sc, clearly illustrative) equiatomic solid
solution whose composition divides the 3x3x3 supercell evenly, so the target
concentrations come out exact:

    bcc -> MoNbTa   (refractory HEA)      a = 3.24 A         54 atoms, 18 each
    fcc -> CoCrFeNi (Cantor-family HEA)   a = 3.57 A        108 atoms, 27 each
    hcp -> TiZrHf   (group-4 hcp alloy)   a = 3.15, c = 5.02 54 atoms, 18 each
    sc  -> VNbTa    (illustrative only *) a = 3.20 A         27 atoms,  9 each

    * simple-cubic metallic solid solutions do not really occur (alpha-Po is the
      only sc element); the sc option is provided purely for demonstration.

For every option: sublattice-specific mode, 3x3x3 supercell, equiatomic
concentrations (which follow from the per-element widget defaults, because the
concentration step is 1 / atoms-on-the-sublattice), pair cutoff 5.0 A, triplet
cutoff 4.0 A, and 3 runs of 50 000 Monte Carlo steps each - which the generated
console script executes 3-at-a-time in parallel.

Unlike SimplySQS, the search itself is *not* started automatically: ICET runs it
in the browser session, so the user presses "Generate SQS Structure" when ready.

Everything is done by pre-seeding the Streamlit ``session_state`` keys that the
workflow widgets read from, then letting the natural post-callback rerun rebuild
the UI with those values in place.
"""

import streamlit as st
from pymatgen.core import Structure, Lattice

# --- Example definitions ----------------------------------------------------
EXAMPLE_SUPERCELL = (3, 3, 3)
EXAMPLE_PAIR_CUTOFF = 5.0
EXAMPLE_TRIPLET_CUTOFF = 4.0
EXAMPLE_N_STEPS = 50000
EXAMPLE_RUNS = 3
EXAMPLE_PARALLEL = 3

# Ordered so the selector lists them bcc, fcc, sc, hcp.
EXAMPLES = {
    "bcc": {
        "label": "BCC · MoNbTa (refractory HEA)",
        "name": "example_bcc_MoNbTa.cif",
        "elements": ["Mo", "Nb", "Ta"],
        "a": 3.24,
    },
    "fcc": {
        "label": "FCC · CoCrFeNi (Cantor-type HEA)",
        "name": "example_fcc_CoCrFeNi.cif",
        "elements": ["Co", "Cr", "Fe", "Ni"],
        "a": 3.57,
    },
    "sc": {
        "label": "SC · VNbTa (illustrative)",
        "name": "example_sc_VNbTa.cif",
        "elements": ["V", "Nb", "Ta"],
        "a": 3.20,
    },
    "hcp": {
        "label": "HCP · TiZrHf (group-4 hcp)",
        "name": "example_hcp_TiZrHf.cif",
        "elements": ["Ti", "Zr", "Hf"],
        "a": 3.15,
        "c": 5.02,
    },
}


def create_example_structure(kind):
    """Build the small conventional base cell for the requested lattice.

    Sublattice-specific mode replaces every site with the selected elements, so
    the base occupancy only has to define the crystallographic *positions*; the
    alloy's first element is used as a placeholder occupant.
    """
    info = EXAMPLES[kind]
    base = info["elements"][0]
    a = info["a"]

    if kind == "sc":                                  # simple cubic, 1 atom
        return Structure(Lattice.cubic(a), [base], [[0.0, 0.0, 0.0]])
    if kind == "bcc":                                 # body-centred cubic, 2 atoms
        return Structure(Lattice.cubic(a), [base] * 2,
                         [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    if kind == "fcc":                                 # face-centred cubic, 4 atoms
        return Structure(Lattice.cubic(a), [base] * 4,
                         [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0],
                          [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    if kind == "hcp":                                 # hexagonal close packed, 2 atoms
        return Structure(Lattice.hexagonal(a, info["c"]), [base] * 2,
                         [[1 / 3, 2 / 3, 1 / 4], [2 / 3, 1 / 3, 3 / 4]])
    raise ValueError(f"Unknown structure kind: {kind}")


def load_example_alloy():
    """Inject the selected example structure and pre-set all SQS parameters.

    Used as an ``on_click`` callback (Streamlit reruns automatically afterwards,
    so we must not call ``st.rerun`` here). On that rerun the workflow widgets
    are created for the first time and pick up the values seeded below. The
    crystal-structure choice is read from the selector's session-state key.
    """
    kind = st.session_state.get("example_alloy_kind", "bcc")
    info = EXAMPLES[kind]

    if not st.session_state.get("full_structures"):
        st.session_state.full_structures = {}
    st.session_state.full_structures[info["name"]] = create_example_structure(kind)

    nx, ny, nz = EXAMPLE_SUPERCELL
    reduce_to_primitive = False

    # The sublattice widgets are keyed on the structure name and the
    # primitive-cell flag, see render_site_sublattice_selector's `stable_key`.
    stable_key = f"icet_sqs_{info['name']}_{reduce_to_primitive}"

    preset = {
        # structure selection (sidebar)
        "sqs_structure_selector": info["name"],
        "sqs_reduce_primitive": reduce_to_primitive,
        # Step 1 - method parameters
        "sqs_cutoff_pair": EXAMPLE_PAIR_CUTOFF,
        "sqs_cutoff_triplet": EXAMPLE_TRIPLET_CUTOFF,
        "sqs_n_steps": EXAMPLE_N_STEPS,
        # a small multi-run batch, executed in parallel by the console script
        "generation_mode_selector": "Multiple Runs",
        "sqs_num_runs": EXAMPLE_RUNS,
        "sqs_parallel_runs": EXAMPLE_PARALLEL,
        # Step 2 - composition mode
        "composition_mode_radio": "🎯 Sublattice-Specific",
        # Step 3 - supercell
        "nx_global": nx,
        "ny_global": ny,
        "nz_global": nz,
        # Step 4 - elements on the single Wyckoff sublattice "A".
        # Equiatomic concentrations follow from the per-element widget defaults,
        # which snap to 1 / (atoms on the sublattice) and divide evenly here.
        f"{stable_key}_sublattice_A_elements": list(info["elements"]),
    }
    for key, value in preset.items():
        st.session_state[key] = value

    st.session_state["example_alloy_loaded"] = info["name"]


def render_example_selector():
    """Selector for the example crystal-structure type (bcc / fcc / sc / hcp)."""
    st.selectbox(
        "Example crystal structure:",
        options=list(EXAMPLES.keys()),
        format_func=lambda k: EXAMPLES[k]["label"],
        key="example_alloy_kind",
    )


def render_example_alloy_button():
    """Button that loads the currently-selected example alloy in one click."""
    kind = st.session_state.get("example_alloy_kind", "bcc")
    info = EXAMPLES[kind]
    formula = "".join(info["elements"])
    # Small spacer so the button lines up with the selectbox input (which has a label).
    st.markdown("<div style='height:1.7em'></div>", unsafe_allow_html=True)
    st.button(
        f"🎲 Load example alloy — {kind.upper()} · {formula} (3×3×3)",
        type="primary",
        width='stretch',
        key="load_example_alloy_btn",
        on_click=load_example_alloy,
    )

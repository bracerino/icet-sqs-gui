#!/usr/bin/env python3
# ===== ICET-SQS-DOCSTRING-BEGIN =====
"""Standalone ICET SQS search with console progress, structure export and plots.

This is the ICET counterpart of the `monitor.sh` script produced by the ATAT SQS
GUI: a single self-contained file that runs the whole search outside Streamlit,
reports what it is doing on the console, and then writes the structures plus the
graphical analysis to disk.

What it does
------------
1. Builds the supercell and the ICET ``ClusterSpace`` from the configuration
   below (or from the command line).
2. Runs the Monte Carlo SQS search, one or several times with different seeds,
   printing a live progress line and recording every reported MC step.
3. Saves each SQS structure in the requested formats and copies the best run to
   ``POSCAR_best_overall`` / ``best_sqs.cif``.
4. Writes the graphical analysis:
   - ``objective_plots/``       best score vs MC step, per run, overlaid, zoomed
   - ``cluster_vector_plots/``  SQS vs target cluster vector and the mismatch
   - ``prdf_plots/``            partial RDF of the best structure
5. Writes ``sqs_progress.csv``, ``cluster_vector_run*.csv`` and
   ``sqs_summary.txt`` next to the plots. Everything goes into ``output_dir``,
   which defaults to the folder the script is run from.

Usage
-----
    python run_sqs_icet.py --structure POSCAR --supercell 3 3 3 \
        --elements Fe:0.5,Ni:0.5 --steps 20000 --runs 4

    python run_sqs_icet.py --config my_config.json

    python run_sqs_icet.py            # runs the CONFIG block below as-is

Ctrl+C stops the search after the current run and still writes every structure,
CSV and plot collected so far.

Requires: icet, ase, pymatgen, numpy, matplotlib (matminer only for the PRDF
plot, which is skipped when it is missing).
"""
# ===== ICET-SQS-DOCSTRING-END =====

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime

# --------------------------------------------------------------------------- #
#  CONFIGURATION                                                               #
#  ICET-SQS replaces everything between the two markers below when it       #
#  generates a ready-to-run copy of this script - keep them intact.            #
# --------------------------------------------------------------------------- #

# ===== ICET-SQS-CONFIG-BEGIN =====
CONFIG = {
    # --- structure -------------------------------------------------------- #
    # Either point at a file, or paste a POSCAR into "structure_poscar".
    "structure_file": "",
    "structure_poscar": """Cu
1.0
   0.000000000    1.807500000    1.807500000
   1.807500000    0.000000000    1.807500000
   1.807500000    1.807500000    0.000000000
Cu
1
Direct
   0.000000000    0.000000000    0.000000000
""",
    "structure_name": "Cu",
    "reduce_to_primitive": False,

    # --- search ----------------------------------------------------------- #
    "supercell": [3, 3, 3],
    # [pair, triplet] in Angstrom. Keep the pair cutoff below half the shortest
    # supercell vector or clusters wrap onto their own periodic images.
    "cutoffs": [5.0, 4.0],
    # "monte_carlo" (simulated annealing on the chosen supercell) or
    # "enumeration" (exhaustive, guarantees the optimum, but only tractable for
    # small cells - use --estimate first to see how big the problem is).
    "method": "monte_carlo",
    "n_steps": 10000,
    "n_runs": 1,
    # How many of those runs to execute at the same time, each in its own
    # process. 1 keeps them sequential. Do not exceed your core count.
    "parallel_runs": 1,
    "base_seed": 42,                # run i uses base_seed + i; 0 means random

    # --- composition ------------------------------------------------------ #
    # Global mode: sublattice_mode = False and a flat {element: fraction} dict.
    # Sublattice mode: sublattice_mode = True, chemical_symbols is one list of
    # allowed species per site of the (primitive) input structure, and
    # target_concentrations is keyed by sublattice letter, e.g.
    #     {"A": {"Fe": 0.5, "Ni": 0.5}}
    "sublattice_mode": False,
    "chemical_symbols": None,
    "target_concentrations": {"Cu": 0.5, "Ni": 0.5},

    # --- output ----------------------------------------------------------- #
    "output_dir": ".",
    "output_formats": ["POSCAR", "CIF"],   # POSCAR, CIF, LAMMPS, XYZ
    "prdf_cutoff": 10.0,
    "prdf_bin_size": 0.1,

    # --- reporting -------------------------------------------------------- #
    "log_every_seconds": 5.0,
    # Enumeration only: seconds allowed for counting the candidates up front so
    # the progress line can count down. Counting roughly doubles the work, so it
    # is abandoned past this budget and the run just reports what it has scored.
    "enumeration_count_timeout": 30.0,
    # Enumeration only: seconds the --estimate probe is allowed to run.
    "enumeration_probe_seconds": 5.0,
    # Enumeration only: refuse to start if the closed-form upper bound on the
    # number of candidates exceeds this. Raise it if you really mean it.
    "enumeration_max_candidates": 5e7,
    # Soft budget for the whole job: once it is exceeded no further run is
    # started. A run already in progress always finishes its n_steps, because
    # ICET's annealing cannot be interrupted without losing the structure.
    # Ignored when parallel_runs > 1, where every run is launched up front.
    "time_limit_minutes": 0,
}
# ===== ICET-SQS-CONFIG-END =====

# --------------------------------------------------------------------------- #

MC_STEP_PATTERN = re.compile(
    r"MC step (\d+)/(\d+) \((\d+) accepted trials, temperature ([\d.eE+-]+)\), "
    r"best score: ([\d.eE+-]+)"
)

PALETTE = ["#2E86C1", "#E67E22", "#27AE60", "#8E44AD", "#C0392B",
           "#16A085", "#D4AC0D", "#2C3E50", "#CB4335", "#5D6D7E"]
INK = "#1E3D7B"
MATCH_COLOR = "#2E86C1"
TARGET_COLOR = "#4A5568"
POS_COLOR = "#E67E22"
NEG_COLOR = "#2E86C1"

ZOOM_FRACTION = 0.20
MIN_ZOOM_POINTS = 3

VACANCY_SYMBOL = "X"


def rule(char="="):
    print(char * 74)


def banner(text):
    print("")
    rule()
    print(f"  {text}")
    rule()


# --------------------------------------------------------------------------- #
#  Configuration handling                                                      #
# --------------------------------------------------------------------------- #

def parse_composition(text):
    """'Fe:0.5,Ni:0.5' -> {'Fe': 0.5, 'Ni': 0.5}."""
    concentrations = OrderedDict()
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"expected ELEMENT:FRACTION, got {chunk!r}")
        element, fraction = chunk.split(":", 1)
        concentrations[element.strip()] = float(fraction)
    if not concentrations:
        raise ValueError("no elements given")
    return concentrations


def build_config(argv=None):
    parser = argparse.ArgumentParser(
        description="Run an ICET SQS search, then save the structures and the plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="JSON file overriding the CONFIG block")
    parser.add_argument("--structure", help="input structure file (CIF, POSCAR, ...)")
    parser.add_argument("--primitive", action="store_true",
                        help="reduce the input to its primitive cell first")
    parser.add_argument("--supercell", nargs=3, type=int, metavar=("NX", "NY", "NZ"))
    parser.add_argument("--elements", help="global composition, e.g. 'Fe:0.5,Ni:0.5'")
    parser.add_argument("--cutoffs", nargs="+", type=float,
                        metavar="R", help="cluster cutoffs: pair [triplet [quadruplet]]")
    parser.add_argument("--method", choices=["monte_carlo", "enumeration"],
                        help="simulated annealing, or exhaustive enumeration")
    parser.add_argument("--estimate", action="store_true",
                        help="only size up an exhaustive enumeration (how many candidate "
                             "structures, how long, and whether Monte Carlo is the better "
                             "choice) and exit without searching")
    parser.add_argument("--count-timeout", type=float, metavar="SECONDS",
                        help="enumeration: budget for counting the candidates up front so "
                             "the progress line can count down (0 disables it)")
    parser.add_argument("--steps", type=int, help="Monte Carlo steps per run")
    parser.add_argument("--runs", type=int, help="number of independent runs")
    parser.add_argument("--parallel", type=int, metavar="N",
                        help="run N of those searches at the same time, each in its own "
                             "process (default 1 = sequential)")
    parser.add_argument("--seed", type=int, help="base random seed (0 = random)")
    parser.add_argument("--output-dir", help="directory for structures, CSVs and plots")
    parser.add_argument("--formats", help="comma separated: POSCAR,CIF,LAMMPS,XYZ")
    parser.add_argument("--prdf-cutoff", type=float)
    parser.add_argument("--prdf-bin-size", type=float)
    parser.add_argument("--log-every", type=float, metavar="SECONDS",
                        help="minimum delay between two console progress lines")
    parser.add_argument("--time-limit", type=float, metavar="MINUTES",
                        help="stop starting new runs once this many minutes elapsed")
    parser.add_argument("--no-plots", action="store_true", help="skip the plots")
    args = parser.parse_args(argv)

    config = dict(CONFIG)
    if args.config:
        with open(args.config) as handle:
            config.update(json.load(handle))

    if args.structure:
        config["structure_file"] = args.structure
        config["structure_poscar"] = ""
        config["structure_name"] = os.path.splitext(os.path.basename(args.structure))[0]
    if args.primitive:
        config["reduce_to_primitive"] = True
    if args.supercell:
        config["supercell"] = list(args.supercell)
    if args.elements:
        config["target_concentrations"] = parse_composition(args.elements)
        config["sublattice_mode"] = False
        config["chemical_symbols"] = None
    if args.cutoffs:
        config["cutoffs"] = list(args.cutoffs)
    if args.method:
        config["method"] = args.method
    if args.estimate:
        config["method"] = "enumeration"
    if args.count_timeout is not None:
        config["enumeration_count_timeout"] = args.count_timeout
    if args.steps:
        config["n_steps"] = args.steps
    if args.runs:
        config["n_runs"] = args.runs
    if args.parallel:
        config["parallel_runs"] = args.parallel
    if args.seed is not None:
        config["base_seed"] = args.seed
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.formats:
        config["output_formats"] = [f.strip().upper() for f in args.formats.split(",") if f.strip()]
    if args.prdf_cutoff:
        config["prdf_cutoff"] = args.prdf_cutoff
    if args.prdf_bin_size:
        config["prdf_bin_size"] = args.prdf_bin_size
    if args.log_every is not None:
        config["log_every_seconds"] = args.log_every
    if args.time_limit is not None:
        config["time_limit_minutes"] = args.time_limit
    config["make_plots"] = not args.no_plots
    config["estimate_only"] = bool(args.estimate)
    return config


# --------------------------------------------------------------------------- #
#  Structure input                                                             #
# --------------------------------------------------------------------------- #

def load_primitive_atoms(config):
    """Return the ASE Atoms the ClusterSpace is built on."""
    from ase.io import read as ase_read

    structure_file = config.get("structure_file") or ""
    poscar_text = config.get("structure_poscar") or ""

    if structure_file:
        if not os.path.exists(structure_file):
            raise SystemExit(f"ERROR: structure file not found: {structure_file}")
        atoms = ase_read(structure_file)
        source = structure_file
    elif poscar_text.strip():
        import io
        atoms = ase_read(io.StringIO(poscar_text), format="vasp")
        source = "embedded POSCAR"
    else:
        raise SystemExit("ERROR: no structure given (set structure_file, "
                         "structure_poscar, or pass --structure)")

    if config.get("reduce_to_primitive"):
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        structure = AseAtomsAdaptor.get_structure(atoms)
        primitive = SpacegroupAnalyzer(structure).get_primitive_standard_structure()
        atoms = AseAtomsAdaptor.get_atoms(primitive)
        source += " (reduced to primitive)"

    return atoms, source


def build_chemical_symbols(config, atoms):
    """One list of allowed species per site of the primitive cell."""
    if config.get("sublattice_mode"):
        chemical_symbols = config.get("chemical_symbols")
        if not chemical_symbols:
            raise SystemExit("ERROR: sublattice_mode needs chemical_symbols")
        if len(chemical_symbols) != len(atoms):
            raise SystemExit(
                f"ERROR: chemical_symbols has {len(chemical_symbols)} entries but the "
                f"structure has {len(atoms)} sites")
        return [sorted(species) for species in chemical_symbols]

    elements = sorted(config["target_concentrations"].keys())
    return [list(elements) for _ in range(len(atoms))]


def achievable_global_concentrations(target_concentrations, total_sites):
    """Round the target fractions onto whole atoms, same rule as the GUI."""
    counts = {}
    remaining = total_sites
    ordered = sorted(target_concentrations.items(), key=lambda item: item[1], reverse=True)
    for index, (element, fraction) in enumerate(ordered):
        if index == len(ordered) - 1:
            counts[element] = remaining
        else:
            count = int(round(fraction * total_sites))
            counts[element] = count
            remaining -= count
    return {element: count / total_sites for element, count in counts.items()}, counts


# --------------------------------------------------------------------------- #
#  Progress capture                                                            #
# --------------------------------------------------------------------------- #

class ProgressLogHandler(logging.Handler):
    """Records ICET's MC-step lines and echoes a throttled progress line."""

    def __init__(self, run_index, records, log_every_seconds):
        super().__init__(level=logging.INFO)
        self.run_index = run_index
        self.records = records
        self.log_every_seconds = log_every_seconds
        self.started = time.time()
        self.last_print = 0.0
        self.last_row = None
        self.printed_step = None

    def emit(self, record):
        try:
            match = MC_STEP_PATTERN.search(record.getMessage())
        except Exception:  # pragma: no cover - logging must never explode
            return
        if not match:
            return

        row = {
            "run": self.run_index,
            "step": int(match.group(1)),
            "total_steps": int(match.group(2)),
            "accepted_trials": int(match.group(3)),
            "temperature": float(match.group(4)),
            "best_score": float(match.group(5)),
            "elapsed_seconds": round(time.time() - self.started, 3),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.records.append(row)
        self.last_row = row

        now = time.time()
        if now - self.last_print >= self.log_every_seconds:
            self.last_print = now
            self.print_row(row)

    def print_row(self, row):
        self.printed_step = row["step"]
        percent = 100.0 * row["step"] / row["total_steps"] if row["total_steps"] else 0.0
        print(f"   run {row['run']:>2} | step {row['step']:>9}/{row['total_steps']:<9}"
              f" ({percent:5.1f} %) | T = {row['temperature']:>9.4f}"
              f" | accepted {row['accepted_trials']:>8}"
              f" | best score {row['best_score']:.6f}"
              f" | {row['elapsed_seconds']:7.1f} s",
              flush=True)


# --------------------------------------------------------------------------- #
#  The search                                                                  #
# --------------------------------------------------------------------------- #

def run_single_sqs(cluster_space, supercell, concentrations, n_steps, seed,
                   run_index, records, log_every_seconds):
    """One annealing run; returns (atoms, seconds, best_score)."""
    from icet.tools.structure_generation import generate_sqs_from_supercells

    # ICET's own console handler stays at WARNING (its default), so the only
    # progress on screen is the formatted line below; the raw INFO records go to
    # icet_sqs.log through the file handler installed in main().
    icet_logger = logging.getLogger("icet.target_cluster_vector_annealing")
    handler = ProgressLogHandler(run_index, records, log_every_seconds)
    icet_logger.addHandler(handler)

    started = time.time()
    try:
        atoms = generate_sqs_from_supercells(
            cluster_space=cluster_space,
            supercells=[supercell],
            target_concentrations=concentrations,
            n_steps=n_steps,
            random_seed=seed if seed else None,
        )
    finally:
        icet_logger.removeHandler(handler)

    elapsed = time.time() - started
    if handler.last_row is not None and handler.last_row["step"] != handler.printed_step:
        handler.print_row(handler.last_row)

    run_scores = [row["best_score"] for row in records if row["run"] == run_index]
    best_score = min(run_scores) if run_scores else None
    return atoms, elapsed, best_score


# --------------------------------------------------------------------------- #
#  Parallel runs                                                               #
# --------------------------------------------------------------------------- #

class QueueProgressHandler(logging.Handler):
    """Ships a worker's MC-step records to the parent process."""

    def __init__(self, run_index, queue):
        super().__init__(level=logging.INFO)
        self.run_index = run_index
        self.queue = queue
        self.started = time.time()
        self.rows = []

    def emit(self, record):
        try:
            match = MC_STEP_PATTERN.search(record.getMessage())
        except Exception:  # pragma: no cover - logging must never explode
            return
        if not match:
            return
        row = {
            "run": self.run_index,
            "step": int(match.group(1)),
            "total_steps": int(match.group(2)),
            "accepted_trials": int(match.group(3)),
            "temperature": float(match.group(4)),
            "best_score": float(match.group(5)),
            "elapsed_seconds": round(time.time() - self.started, 3),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.rows.append(row)
        try:
            self.queue.put(row)
        except Exception:
            pass  # a full or closed queue must not kill the search


def _parallel_worker(job):
    """One SQS run, executed in its own process.

    The cluster space is rebuilt here rather than pickled: it is cheap to
    construct and avoids depending on icet's objects being picklable.
    """
    import logging as worker_logging

    from icet import ClusterSpace
    from icet.tools.structure_generation import generate_sqs_from_supercells

    (run, atoms, supercell, cutoffs, chemical_symbols,
     concentrations, n_steps, seed, queue) = job

    cluster_space = ClusterSpace(atoms, cutoffs, chemical_symbols)

    icet_logger = worker_logging.getLogger("icet.target_cluster_vector_annealing")
    handler = QueueProgressHandler(run, queue)
    icet_logger.addHandler(handler)

    started = time.time()
    try:
        sqs_atoms = generate_sqs_from_supercells(
            cluster_space=cluster_space,
            supercells=[supercell],
            target_concentrations=concentrations,
            n_steps=n_steps,
            random_seed=seed if seed else None,
        )
    finally:
        icet_logger.removeHandler(handler)

    elapsed = time.time() - started
    scores = [row["best_score"] for row in handler.rows]
    return run, sqs_atoms, elapsed, (min(scores) if scores else None), handler.rows


def _print_parallel_status(state, n_runs, elapsed):
    """One line summarising every worker, in the style of ATAT's monitor.

    Finished runs keep showing their final objective value, so the runs stay
    directly comparable at a glance while the rest are still going.
    """
    parts = []
    for run in range(1, n_runs + 1):
        row = state.get(run)
        if row is None:
            parts.append(f"R{run} ⏳ ------")          # queued, not started yet
        elif row.get("failed"):
            parts.append(f"R{run} ❌ ------")
        elif row.get("done"):
            score = row.get("best_score")
            parts.append(f"R{run} ✅ " + ("  n/a " if score is None else f"{score:.4f}"))
        else:
            percent = (100.0 * row["step"] / row["total_steps"]
                       if row.get("total_steps") else 0.0)
            parts.append(f"R{run} {percent:3.0f}% {row['best_score']:.4f}")
    print(f"   [{elapsed:7.1f} s] " + " | ".join(parts), flush=True)


def run_parallel_sqs(atoms, supercell, cutoffs, chemical_symbols, concentrations,
                     n_steps, base_seed, n_runs, parallel_runs, log_every_seconds,
                     on_result):
    """Run `n_runs` searches `parallel_runs` at a time.

    `on_result(run, sqs_atoms, elapsed, best_score, rows)` is called as soon as
    each run lands, so its structure and plots are written while the remaining
    workers are still going.

    Returns the list of MC-step records gathered from every worker.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    manager = multiprocessing.Manager()
    queue = manager.Queue()

    jobs = []
    for run in range(1, n_runs + 1):
        seed = base_seed + (run - 1) if base_seed else 0
        jobs.append((run, atoms, supercell, cutoffs, chemical_symbols,
                     concentrations, n_steps, seed, queue))

    print(f"   {n_runs} runs, {parallel_runs} at a time "
          f"(seeds {base_seed}..{base_seed + n_runs - 1})" if base_seed
          else f"   {n_runs} runs, {parallel_runs} at a time (random seeds)")

    records = []
    state = {}
    started = time.time()
    last_print = 0.0

    with ProcessPoolExecutor(max_workers=parallel_runs) as executor:
        futures = {executor.submit(_parallel_worker, job): job[0] for job in jobs}

        pending = set(futures)
        while pending:
            done, pending = _wait_briefly(pending)

            while True:
                try:
                    row = queue.get_nowait()
                except Exception:
                    break
                state[row["run"]] = row

            now = time.time()
            if now - last_print >= log_every_seconds:
                last_print = now
                _print_parallel_status(state, n_runs, now - started)

            for future in done:
                run = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"   ❌ run {run} failed: {exc}", flush=True)
                    state[run] = {"done": True, "failed": True}
                    continue
                run, sqs_atoms, elapsed, best_score, rows = result
                records.extend(rows)
                state[run] = {"done": True, "best_score": best_score}
                on_result(run, sqs_atoms, elapsed, best_score, rows)

    _print_parallel_status(state, n_runs, time.time() - started)
    return records


def _wait_briefly(pending, timeout=0.5):
    """concurrent.futures.wait, split out so the polling loop reads cleanly."""
    from concurrent.futures import FIRST_COMPLETED, wait

    done, still_pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
    return done, still_pending


# --------------------------------------------------------------------------- #
#  Enumeration                                                                 #
# --------------------------------------------------------------------------- #

def enumeration_size(cluster_space, supercell):
    """How many primitive cells the chosen supercell holds.

    ICET sizes its enumeration in *primitive cells*, not atoms, so a supercell
    of a 4-atom conventional fcc cell repeated 2x2x2 is 32 primitive cells, not
    32 atoms and not 8 cells.
    """
    n_primitive = len(cluster_space.primitive_structure)
    return max(1, int(round(len(supercell) / n_primitive)))


def _concentration_restrictions(cluster_space, concentrations):
    """Per-sublattice concentrations -> enumerate_structures' restriction dict.

    Same translation generate_sqs_by_enumeration does internally: the caller's
    per-sublattice fractions become exact whole-cell fractions.
    """
    from icet.tools.structure_generation import _validate_concentrations

    # Global mode hands in a flat {element: fraction} dict; normalise it to the
    # per-sublattice form first, otherwise every lookup below misses.
    concentrations = _validate_concentrations(concentrations, cluster_space)

    restrictions = {}
    primitive = cluster_space.primitive_structure
    for sublattice in cluster_space.get_sublattices(primitive):
        weight = len(sublattice.indices) / len(primitive)
        sublattice_conc = concentrations.get(
            sublattice.symbol, {sublattice.chemical_symbols[0]: 1.0})
        for species, fraction in sublattice_conc.items():
            value = fraction * weight
            if species in restrictions:
                low, high = restrictions[species]
                restrictions[species] = (low + value, high + value)
            else:
                restrictions[species] = (value, value)
    return restrictions


def arrangement_count(cluster_space, supercell, concentrations):
    """Ways to arrange the atoms in the chosen supercell, ignoring symmetry.

    The plain multinomial per sublattice. This is the size of the problem for
    *one* cell shape; ICET additionally enumerates every inequivalent shape of
    the same size, and then removes symmetry-equivalent decorations.
    """
    from math import factorial

    from icet.tools.structure_generation import _validate_concentrations

    concentrations = _validate_concentrations(concentrations, cluster_space)

    total = 1
    for sublattice in cluster_space.get_sublattices(supercell):
        if len(sublattice.chemical_symbols) < 2:
            continue
        sublattice_conc = concentrations.get(sublattice.symbol)
        if not sublattice_conc:
            continue
        n_sites = len(sublattice.indices)
        ways = factorial(n_sites)
        for fraction in sublattice_conc.values():
            ways //= factorial(int(round(fraction * n_sites)))
        total *= ways
    return total


def hnf_cell_count(n_cells):
    """Number of distinct Hermite normal forms of determinant `n_cells`.

    Every derivative superstructure of that size sits in one of these cells, so
    hnf_cell_count * arrangement_count is a rigorous (if generous, by roughly
    the order of the point group) upper bound on what ICET has to enumerate.
    Closed form, so it costs nothing — unlike icet's enumerate_supercells, which
    builds every cell.
    """
    total = 0
    for a in range(1, n_cells + 1):
        if n_cells % a:
            continue
        rest = n_cells // a
        for b in range(1, rest + 1):
            if rest % b:
                continue
            c = rest // b
            total += b * c * c
    return total


def _enumerated_structures(cluster_space, n_cells, restrictions, pbc=None):
    """The raw stream of candidate structures ICET would score."""
    from icet.tools import enumerate_structures

    primitive = cluster_space.primitive_structure
    primitive.set_pbc(pbc or (True, True, True))
    return enumerate_structures(primitive, [n_cells], cluster_space.chemical_symbols,
                                concentration_restrictions=restrictions)


def count_enumerated_structures(cluster_space, n_cells, restrictions,
                                time_budget=None, pbc=None):
    """Count the candidates, optionally giving up after `time_budget` seconds.

    Returns (count, elapsed, finished). Enumeration is the expensive half of the
    search (scoring is comparatively cheap), so counting first roughly doubles
    the work — hence the budget.
    """
    started = time.time()
    count = 0
    for _ in _enumerated_structures(cluster_space, n_cells, restrictions, pbc):
        count += 1
        if time_budget and (time.time() - started) > time_budget:
            return count, time.time() - started, False
    return count, time.time() - started, True


def enumeration_scale(cluster_space, supercell, concentrations):
    """The instant, closed-form size numbers for an enumeration.

    No enumeration is performed, so this is safe to call on any configuration,
    however large.
    """
    n_cells = enumeration_size(cluster_space, supercell)
    arrangements = arrangement_count(cluster_space, supercell, concentrations)
    shapes = hnf_cell_count(n_cells)
    return {
        "n_atoms": len(supercell),
        "n_cells": n_cells,
        "arrangements": arrangements,
        "hnf_shapes": shapes,
        "upper_bound": shapes * arrangements,
    }


def print_enumeration_scale(scale):
    print(f"  Supercell                    : {scale['n_atoms']} atoms "
          f"= {scale['n_cells']} primitive cells")
    print(f"  Arrangements in that cell    : {scale['arrangements']:,}")
    print(f"  Distinct cell shapes (HNFs)  : {scale['hnf_shapes']:,}")
    print(f"  Upper bound on candidates    : {scale['upper_bound']:,}"
          f"   (before symmetry reduction)")


def estimate_enumeration(cluster_space, supercell, concentrations,
                         probe_seconds=5.0, pbc=None, max_candidates=5e7):
    """Measure how big an exhaustive enumeration would be, and advise on it.

    Runs the real enumerate-and-score loop for at most `probe_seconds`. If it
    finishes, the numbers reported are exact. If it does not, they are a lower
    bound plus the measured rate, which is all that can be said honestly without
    paying for the whole thing.

    Hopeless cases are settled from the closed-form numbers alone and never
    touch the enumerator: it can spend a very long time inside itself before
    yielding its first structure, so the in-loop time budget would not save us.
    """
    from icet.tools.structure_generation import (
        _get_sqs_cluster_vector, _validate_concentrations, compare_cluster_vectors)

    scale = enumeration_scale(cluster_space, supercell, concentrations)
    n_cells = scale["n_cells"]

    if scale["upper_bound"] > max_candidates:
        result = dict(scale)
        result.update({
            "counted": 0, "elapsed": 0.0, "finished": False, "rate": 0.0,
            "total_seconds": None, "probed": False,
            "verdict": "too-large",
            "advice": (
                f"Enumeration is hopeless here: this cell admits up to "
                f"{scale['upper_bound']:,} candidate structures, far beyond the "
                f"{max_candidates:,.0f} that could be walked in any reasonable "
                f"time. Use the Monte Carlo method (Supercell-Specific), or "
                f"shrink the supercell."),
        })
        return result
    restrictions = _concentration_restrictions(cluster_space, concentrations)

    validated = _validate_concentrations(concentrations, cluster_space)
    target_vector = _get_sqs_cluster_vector(cluster_space, validated)
    as_list = cluster_space.as_list

    started = time.time()
    counted = 0
    finished = True
    for structure in _enumerated_structures(cluster_space, n_cells, restrictions, pbc):
        cluster_vector = cluster_space.get_cluster_vector(structure)
        compare_cluster_vectors(cv_1=cluster_vector, cv_2=target_vector, as_list=as_list)
        counted += 1
        if probe_seconds and (time.time() - started) > probe_seconds:
            finished = False
            break
    elapsed = time.time() - started

    rate = counted / elapsed if elapsed > 0 else 0.0
    result = dict(scale)
    result.update({
        "probed": True,
        "counted": counted,
        "elapsed": elapsed,
        "finished": finished,
        "rate": rate,
        "total_seconds": elapsed if finished else None,
    })

    if finished:
        if elapsed <= 60:
            result["verdict"] = "recommended"
            result["advice"] = (
                f"Enumeration is a good choice here: it walked all "
                f"{counted:,} candidate structures in {elapsed:.1f} s and is "
                f"exhaustive, so the result is provably the best structure of "
                f"this size.")
        elif elapsed <= 900:
            result["verdict"] = "feasible"
            result["advice"] = (
                f"Enumeration is feasible but slow: {counted:,} structures take "
                f"about {elapsed / 60:.1f} min. It guarantees the optimum; Monte "
                f"Carlo would get very close in seconds.")
        else:
            result["verdict"] = "slow"
            result["advice"] = (
                f"Enumeration would take about {elapsed / 3600:.1f} h for "
                f"{counted:,} structures. Monte Carlo is the practical choice "
                f"unless you specifically need the guaranteed optimum.")
    else:
        result["verdict"] = "too-large"
        result["advice"] = (
            f"Enumeration is not practical here: after {elapsed:.1f} s it had "
            f"produced {counted:,} structures at {rate:,.0f}/s and was still "
            f"going, with at most {scale['upper_bound']:,} to get through. "
            f"Use the Monte Carlo method (Supercell-Specific) instead.")

    return result


def print_enumeration_estimate(estimate):
    """Console rendering of estimate_enumeration's result."""
    banner("Enumeration size estimate")
    print_enumeration_scale(estimate)
    print("")
    if not estimate.get("probed", True):
        print("  Not measured: the closed-form numbers already settle it.")
    elif estimate["finished"]:
        print(f"  Candidates actually enumerated: {estimate['counted']:,}")
        print(f"  Time for the full enumeration : {estimate['elapsed']:.1f} s")
    else:
        print(f"  Enumerated in {estimate['elapsed']:.1f} s so far : "
              f"{estimate['counted']:,} (not finished)")
        print(f"  Measured rate                 : {estimate['rate']:,.0f} structures/s")
    print("")
    print(f"  Verdict: {estimate['verdict'].upper()}")
    for line in _wrap(estimate["advice"], 68):
        print(f"    {line}")
    rule()


def _wrap(text, width):
    """Minimal word wrap, so the advice paragraph stays readable."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def run_enumeration(cluster_space, supercell, concentrations, log_every_seconds=5.0,
                    count_timeout=30.0, pbc=None):
    """Exhaustive enumeration with a console progress readout.

    When the candidates can be counted quickly (within `count_timeout`) the
    progress line counts down; otherwise it reports how many have been scored so
    far, because the total is not knowable without doing the enumeration twice.

    Returns (best_structure, elapsed, best_score, n_scored).
    """
    from icet.tools.structure_generation import (
        _get_sqs_cluster_vector, _validate_concentrations, compare_cluster_vectors)

    n_cells = enumeration_size(cluster_space, supercell)
    restrictions = _concentration_restrictions(cluster_space, concentrations)

    print(f"   enumerating derivative superstructures of {n_cells} primitive cells "
          f"({len(supercell)} atoms)")

    total = None
    if count_timeout:
        print(f"   counting the candidates first (up to {count_timeout:g} s)...", flush=True)
        counted, count_elapsed, finished = count_enumerated_structures(
            cluster_space, n_cells, restrictions, time_budget=count_timeout, pbc=pbc)
        if finished:
            total = counted
            print(f"   {total:,} candidates to score (counted in {count_elapsed:.1f} s)",
                  flush=True)
        else:
            print(f"   still counting after {count_elapsed:.1f} s ({counted:,}+ candidates) "
                  f"- continuing without a countdown", flush=True)

    validated = _validate_concentrations(concentrations, cluster_space)
    target_vector = _get_sqs_cluster_vector(cluster_space, validated)
    as_list = cluster_space.as_list

    started = time.time()
    last_print = 0.0
    best_score = None
    best_structure = None
    scored = 0

    for structure in _enumerated_structures(cluster_space, n_cells, restrictions, pbc):
        cluster_vector = cluster_space.get_cluster_vector(structure)
        score = compare_cluster_vectors(cv_1=cluster_vector, cv_2=target_vector,
                                        as_list=as_list)
        scored += 1
        if best_score is None or score < best_score:
            best_score = score
            best_structure = structure

        now = time.time()
        if now - last_print >= log_every_seconds:
            last_print = now
            _print_enumeration_progress(scored, total, best_score, now - started)

    elapsed = time.time() - started
    _print_enumeration_progress(scored, total, best_score, elapsed)

    if best_structure is None:
        raise RuntimeError(
            "Enumeration produced no structure at all - the requested concentrations "
            "are probably not realisable in a cell of this size.")

    return best_structure, elapsed, best_score, scored


def _print_enumeration_progress(scored, total, best_score, elapsed):
    score_text = "n/a" if best_score is None else f"{best_score:.6f}"
    if total:
        remaining = max(0, total - scored)
        percent = 100.0 * scored / total
        rate = scored / elapsed if elapsed > 0 else 0.0
        # The first few samples are dominated by start-up, so the extrapolation
        # from them is meaningless - show it only once the rate has settled.
        if scored >= 10 and rate > 0:
            eta_text = f"~{remaining / rate:7.1f} s to go"
        else:
            eta_text = "estimating rate..."
        print(f"   scored {scored:>9,}/{total:<9,} ({percent:5.1f} %)"
              f" | {remaining:>9,} left"
              f" | best score {score_text}"
              f" | {elapsed:7.1f} s elapsed, {eta_text}", flush=True)
    else:
        rate = scored / elapsed if elapsed > 0 else 0.0
        print(f"   scored {scored:>9,} structures (total unknown)"
              f" | best score {score_text}"
              f" | {elapsed:7.1f} s elapsed, {rate:,.0f}/s", flush=True)


# --------------------------------------------------------------------------- #
#  Cluster vector analysis                                                     #
# --------------------------------------------------------------------------- #

def cluster_vector_analysis(cluster_space, atoms, target_vector):
    """Per-orbit SQS vs target comparison, skipping the trivial zerolet."""
    import numpy as np

    sqs_vector = np.asarray(cluster_space.get_cluster_vector(atoms), dtype=float)
    target_vector = np.asarray(target_vector, dtype=float)
    orbits = cluster_space.as_list

    clusters = []
    for index, orbit in enumerate(orbits):
        if index >= len(sqs_vector) or index >= len(target_vector):
            break
        if orbit.get("order", 0) == 0:
            continue  # always exactly 1.0 for every structure
        clusters.append({
            "index": index,
            "order": int(orbit.get("order", 0)),
            "radius": float(orbit.get("radius", 0.0)),
            "multiplicity": int(orbit.get("multiplicity", 1)),
            "sqs": float(sqs_vector[index]),
            "target": float(target_vector[index]),
            "diff": float(sqs_vector[index] - target_vector[index]),
        })
    return clusters


def sublattice_symbol_overlap(cluster_space, supercell):
    """Species that ICET would place on more than one *active* sublattice.

    mchammer cannot tell two atoms of the same species apart, so it refuses to
    run when one symbol is allowed on two different active sublattices
    ("Symbols {...} found on multiple active sublattices"). Catching it here
    turns that into an explanation before any time is spent on the search.

    Returns {symbol: [sublattice letters]} for the offending species only.
    """
    active = [sublattice for sublattice in cluster_space.get_sublattices(supercell)
              if len(sublattice.chemical_symbols) > 1]

    where = defaultdict(list)
    for sublattice in active:
        for symbol in sublattice.chemical_symbols:
            where[symbol].append(sublattice.symbol)

    return {symbol: letters for symbol, letters in where.items() if len(letters) > 1}


def orbit_weights(clusters):
    """Weight every orbit by 1 / radius, so near-neighbour shells count for more.

    An unweighted RMSE treats a 6 A pair exactly like a nearest-neighbour pair,
    which is not how an SQS is judged: the short-range order is what matters.
    ICET's own objective encodes that by rewarding a perfectly matched run of
    near shells, and `compare_cluster_vectors` accepts per-orbit weights for the
    same reason (it just defaults them all to 1). Weighting by 1 / radius gives
    the match score the same priority, so it no longer disagrees with the score
    purely because the two look at different distances.

    Zero-radius orbits (the point terms, which carry the composition) take the
    largest weight in the set.
    """
    radii = [cluster["radius"] for cluster in clusters if cluster["radius"] > 1e-9]
    if not radii:
        return [1.0] * len(clusters)
    closest = min(radii)
    return [1.0 / max(cluster["radius"], closest) for cluster in clusters]


def match_stats(clusters, weighted=True):
    """RMSE / mean / worst mismatch plus a 0-100 % match score.

    Cluster vector components live in [-1, 1], so the RMSE of (SQS - target) is
    already on that scale and 100 * (1 - RMSE) reads directly as "how well does
    this SQS reproduce the random-alloy target". With `weighted` the RMSE is
    taken over orbit_weights(), emphasising the near shells.
    """
    diffs = [cluster["diff"] for cluster in clusters]
    if not diffs:
        return 0.0, 0.0, 0.0, 0, 0.0

    count = len(diffs)
    weights = orbit_weights(clusters) if weighted else [1.0] * count
    total_weight = sum(weights) or float(count)

    rmse = (sum(w * d * d for w, d in zip(weights, diffs)) / total_weight) ** 0.5
    mean_abs = sum(w * abs(d) for w, d in zip(weights, diffs)) / total_weight
    worst = max(abs(value) for value in diffs)
    exact = sum(1 for value in diffs if abs(value) < 1e-6)
    score = max(0.0, min(100.0, 100.0 * (1.0 - rmse)))
    return rmse, mean_abs, worst, exact, score


def score_breakdown(clusters, optimality_weight=1.0, tol=1e-5):
    """Split ICET's objective into the two terms it is actually made of.

    ICET scores a candidate as

        score = sum |cv - target|  -  optimality_weight * longest_optimal_radius

    where `longest_optimal_radius` is the radius of the furthest *pair* shell
    such that every pair shell up to it matches the target exactly. The second
    term is measured in Angstrom and is therefore often much larger than the
    first, which is why a structure can have a better (more negative) score than
    another while having a larger average mismatch: the score deliberately
    prizes getting the near-neighbour shells exactly right over spreading the
    error evenly.

    Returns (sum_abs_mismatch, longest_optimal_radius, n_perfect_pair_shells).
    """
    total_mismatch = sum(abs(cluster["diff"]) for cluster in clusters)

    longest_optimal_radius = 0.0
    perfect_shells = 0
    for cluster in clusters:
        if cluster["order"] != 2:
            continue
        if abs(cluster["diff"]) < tol:
            longest_optimal_radius = cluster["radius"]
            perfect_shells += 1
        else:
            break

    return total_mismatch, longest_optimal_radius, perfect_shells


def write_cluster_vector_csv(path, clusters):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Orbit", "Order", "Radius", "Multiplicity",
                         "SQS", "Target", "SQS_minus_target"])
        for cluster in clusters:
            writer.writerow([cluster["index"], cluster["order"],
                             f"{cluster['radius']:.6f}", cluster["multiplicity"],
                             f"{cluster['sqs']:.8f}", f"{cluster['target']:.8f}",
                             f"{cluster['diff']:.8f}"])


# --------------------------------------------------------------------------- #
#  Structure export                                                            #
# --------------------------------------------------------------------------- #

def strip_vacancies(structure):
    """Drop the placeholder 'X' sites the GUI uses to model vacancies."""
    from pymatgen.core import Structure

    kept = [(site.specie.symbol, site.frac_coords) for site in structure
            if site.specie.symbol != VACANCY_SYMBOL]
    if len(kept) == len(structure):
        return structure, 0
    removed = len(structure) - len(kept)
    if not kept:
        return structure, 0
    return Structure(structure.lattice,
                     [symbol for symbol, _ in kept],
                     [coords for _, coords in kept]), removed


def save_structure(structure, directory, basename, formats):
    """Write one structure in every requested format; returns the paths."""
    from pymatgen.io.cif import CifWriter
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.io import write as ase_write

    written = []
    for fmt in formats:
        fmt = fmt.upper()
        try:
            if fmt == "POSCAR":
                path = os.path.join(directory, f"{basename}_POSCAR")
                structure.to(filename=path, fmt="poscar")
            elif fmt == "CIF":
                path = os.path.join(directory, f"{basename}.cif")
                with open(path, "w") as handle:
                    handle.write(str(CifWriter(structure)))
            elif fmt == "LAMMPS":
                path = os.path.join(directory, f"{basename}.lmp")
                ase_write(path, AseAtomsAdaptor.get_atoms(structure),
                          format="lammps-data", masses=True)
            elif fmt in ("XYZ", "EXTXYZ"):
                path = os.path.join(directory, f"{basename}.xyz")
                ase_write(path, AseAtomsAdaptor.get_atoms(structure), format="extxyz")
            else:
                print(f"   ⚠️  unknown output format {fmt!r} - skipped")
                continue
        except Exception as exc:
            print(f"   ⚠️  could not write {fmt}: {exc}")
            continue
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
#  Plotting                                                                    #
# --------------------------------------------------------------------------- #

def setup_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"   ⚠️  matplotlib not available ({exc}) - skipping plots")
        return None

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titlepad": 12,
        "axes.labelsize": 12,
        "axes.labelpad": 8,
        "axes.linewidth": 1.1,
        "axes.edgecolor": "#4A5568",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#C8CFDA",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.55,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.frameon": False,
        "legend.fontsize": 10.5,
        "lines.antialiased": True,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })
    return plt


def zoom_start(all_x):
    """Beginning of the last ZOOM_FRACTION of the search."""
    if not all_x:
        return None
    first, last = min(all_x), max(all_x)
    if last <= first:
        return None
    return last - ZOOM_FRACTION * (last - first)


def clip(xs, ys, start):
    pairs = [(x, y) for x, y in zip(xs, ys) if x >= start - 1e-9]
    if len(pairs) < MIN_ZOOM_POINTS:
        return [], []
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def plot_objective(plt, records, run_indices, plot_dir, written):
    """Best score vs MC step: one figure per run, one overlay, plus zooms."""
    os.makedirs(plot_dir, exist_ok=True)

    def draw(entries, title, filename, legend=False):
        fig, ax = plt.subplots(figsize=(8.2, 5.0) if legend else (6.8, 4.4))
        for label, xs, ys, color in entries:
            ax.plot(xs, ys, linewidth=2.0 if legend else 2.2, color=color,
                    zorder=3, label=label)
            if not legend and len(xs) <= 60:
                ax.plot(xs, ys, "o", markersize=5.5, color=color,
                        markeredgecolor="white", markeredgewidth=1.1, zorder=4)
            if not legend:
                ax.fill_between(xs, ys, max(ys), color=color, alpha=0.10,
                                linewidth=0, zorder=1)
        if legend:
            ax.legend(loc="best", ncol=1)
        ax.set_title(title, color=INK)
        ax.set_xlabel("MC step")
        ax.set_ylabel("Best score (objective function)")
        ax.margins(x=0.03, y=0.10)
        ax.set_axisbelow(True)
        fig.tight_layout()
        out_path = os.path.join(plot_dir, filename)
        fig.savefig(out_path)
        plt.close(fig)
        written.append(out_path)

    collected = []
    for run in run_indices:
        rows = [row for row in records if row["run"] == run]
        if not rows:
            print(f"   run {run}: no MC steps recorded - skipped")
            continue
        xs = [row["step"] for row in rows]
        ys = [row["best_score"] for row in rows]
        color = PALETTE[(run - 1) % len(PALETTE)]
        draw([(None, xs, ys, color)],
             f"ICET run {run}  |  best {min(ys):.6f}",
             f"run{run}_objective.png")
        collected.append((run, xs, ys, color))

    if len(collected) > 1:
        draw([(f"Run {run}  (best {min(ys):.6f})", xs, ys, color)
              for run, xs, ys, color in collected],
             f"ICET best score - all {len(collected)} runs",
             "all_runs_objective.png", legend=True)

    every_x = [x for _, xs, _, _ in collected for x in xs]
    start = zoom_start(every_x)
    if start is None:
        print("   too little data for the zoomed views - skipped")
        return

    zoom_entries = []
    for run, xs, ys, color in collected:
        zx, zy = clip(xs, ys, start)
        if not zx:
            continue
        draw([(None, zx, zy, color)],
             f"ICET run {run} - last {int(ZOOM_FRACTION * 100)} % of the search",
             f"run{run}_objective_zoom.png")
        zoom_entries.append((f"Run {run}  (best {min(zy):.6f})", zx, zy, color))

    if len(zoom_entries) > 1:
        draw(zoom_entries,
             f"ICET best score - all runs, last {int(ZOOM_FRACTION * 100)} % of the search",
             "all_runs_objective_zoom.png", legend=True)


def cluster_ticks(clusters):
    """Label groups of orbits sharing (order, radius) instead of every tick."""
    groups = []
    for position, cluster in enumerate(clusters):
        key = (cluster["order"], round(cluster["radius"], 4))
        if groups and groups[-1][0] == key:
            groups[-1][2] = position
        else:
            groups.append([key, position, position])

    if len(groups) <= 10:
        positions = [(first + last) / 2.0 + 1 for key, first, last in groups]
        labels = ["%d-pt\n%.3f" % key for key, first, last in groups]
        separators = [first + 0.5 for key, first, last in groups[1:]]
        return positions, labels, separators

    stride = int(len(clusters) / 10) + 1
    positions = [i + 1 for i in range(0, len(clusters), stride)]
    labels = [str(i + 1) for i in range(0, len(clusters), stride)]
    return positions, labels, []


def plot_cluster_vector(plt, clusters, title, out_path, best_score=None):
    """Target vs SQS cluster vector on top, the mismatch as bars below."""
    idx = list(range(1, len(clusters) + 1))
    sqs = [cluster["sqs"] for cluster in clusters]
    target = [cluster["target"] for cluster in clusters]
    diff = [cluster["diff"] for cluster in clusters]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.4, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.25], "hspace": 0.10})

    if len(clusters) <= 12:
        target_size, sqs_size, stem_width = 9.0, 7.5, 1.2
    elif len(clusters) <= 30:
        target_size, sqs_size, stem_width = 6.5, 5.5, 1.0
    else:
        target_size, sqs_size, stem_width = 5.0, 4.2, 0.9

    for x, sqs_value, target_value in zip(idx, sqs, target):
        ax_top.plot([x, x], [target_value, sqs_value], color="#B7BFCC",
                    linewidth=stem_width, zorder=1)
    ax_top.axhline(0.0, color="#9AA3B2", linewidth=0.9, linestyle=":", zorder=0)
    ax_top.plot(idx, target, "s", markersize=target_size, markerfacecolor="none",
                markeredgecolor=TARGET_COLOR, markeredgewidth=1.5, zorder=3,
                label="Target (perfectly random)")
    ax_top.plot(idx, sqs, "o", markersize=sqs_size, color=MATCH_COLOR,
                markeredgecolor="white", markeredgewidth=1.0, zorder=4, label="SQS")
    ax_top.set_ylabel("Cluster vector component")
    ax_top.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, borderaxespad=0.0)
    ax_top.set_axisbelow(True)
    low, high = min(sqs + target), max(sqs + target)
    pad = max((high - low) * 0.18, 1e-3)
    ax_top.set_ylim(low - pad, high + pad)

    rmse, mean_abs, worst, exact, score = match_stats(clusters)
    if best_score is not None:
        title = f"{title}  |  best score {best_score:.6f}"
    ax_top.set_title(f"{title}  |  match score {score:.2f} %", color=INK, pad=34)

    colors = [POS_COLOR if value >= 0 else NEG_COLOR for value in diff]
    ax_bot.bar(idx, diff, width=0.6, color=colors, edgecolor="white",
               linewidth=0.8, zorder=3)
    ax_bot.axhline(0.0, color="#4A5568", linewidth=1.0, zorder=2)
    span = max([abs(value) for value in diff] + [1e-6]) * 1.35
    ax_bot.set_ylim(-span, span)
    ax_bot.set_ylabel("SQS - target")
    ax_bot.set_xlabel("Cluster (points, radius)")
    positions, labels, separators = cluster_ticks(clusters)
    ax_bot.set_xticks(positions)
    ax_bot.set_xticklabels(labels)
    ax_bot.set_axisbelow(True)
    for boundary in separators:
        ax_top.axvline(boundary, color="#DDE2EA", linewidth=0.9, zorder=0)
        ax_bot.axvline(boundary, color="#DDE2EA", linewidth=0.9, zorder=0)

    # tight_layout cannot handle the legend anchored outside the axes; the
    # explicit margins plus savefig(bbox="tight") give the same result.
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.19, hspace=0.10)
    fig.text(0.5, 0.012,
             f"RMSE = {rmse:.6f}     mean |mismatch| = {mean_abs:.6f}"
             f"     max |mismatch| = {worst:.6f}     exact matches: {exact}/{len(diff)}",
             ha="center", va="bottom", fontsize=10, color="#4A5568")
    fig.savefig(out_path)
    plt.close(fig)


def plot_cluster_vector_overlay(plt, results, out_path):
    """Mismatch of every run in one figure."""
    reference = results[0]["clusters"]
    idx = list(range(1, len(reference) + 1))
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.axhline(0.0, color="#4A5568", linewidth=1.0, zorder=2)
    for position, result in enumerate(results):
        clusters = result["clusters"]
        if len(clusters) != len(reference):
            continue
        color = PALETTE[position % len(PALETTE)]
        label = f"Run {result['run']}"
        if result.get("best_score") is not None:
            label += f"  (best score {result['best_score']:.6f})"
        label += f"  -  match {match_stats(clusters)[4]:.2f} %"
        diff = [cluster["diff"] for cluster in clusters]
        ax.plot(idx, diff, linewidth=1.8, color=color, zorder=3, label=label)
        ax.plot(idx, diff, "o", markersize=5.5, color=color,
                markeredgecolor="white", markeredgewidth=1.0, zorder=4)
    ax.set_ylabel("SQS - target")
    ax.set_xlabel("Cluster (points, radius)")
    positions, labels, separators = cluster_ticks(reference)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    for boundary in separators:
        ax.axvline(boundary, color="#DDE2EA", linewidth=0.9, zorder=0)
    ax.set_title(f"Cluster vector mismatch - all {len(results)} runs", color=INK)
    ax.legend(loc="best")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_prdf(plt, structure, cutoff, bin_size, plot_dir, written):
    """Partial RDF of the best structure - the same analysis the GUI shows."""
    try:
        from matminer.featurizers.structure import PartialRadialDistributionFunction
    except Exception as exc:
        print(f"   ⚠️  matminer not available ({exc}) - skipping the PRDF plot")
        return

    try:
        featurizer = PartialRadialDistributionFunction(cutoff=cutoff, bin_size=bin_size)
        featurizer.fit([structure])
        values = featurizer.featurize(structure)
        labels = featurizer.feature_labels()
    except Exception as exc:
        print(f"   ⚠️  PRDF calculation failed ({exc}) - skipping the PRDF plot")
        return

    prdf = defaultdict(list)
    distances = defaultdict(list)
    for value, label in zip(values, labels):
        head, _, tail = label.partition(" PRDF r=")
        if not tail:
            continue
        pair = tuple(head.split("-"))
        low, _, high = tail.partition("-")
        try:
            prdf[pair].append(value)
            distances[pair].append((float(low) + float(high)) / 2.0)
        except ValueError:
            continue

    if not prdf:
        print("   ⚠️  no PRDF data produced - skipping the PRDF plot")
        return

    os.makedirs(plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for position, pair in enumerate(sorted(prdf)):
        ax.plot(distances[pair], prdf[pair], linewidth=1.8,
                color=PALETTE[position % len(PALETTE)], label=f"{pair[0]}-{pair[1]}")
    ax.set_xlabel("Distance (Å)")
    ax.set_ylabel("PRDF intensity")
    ax.set_ylim(bottom=0.0)
    ax.set_title("PRDF of the best SQS - all element pairs", color=INK)
    ax.legend(loc="best", ncol=2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out_path = os.path.join(plot_dir, "prdf_best_sqs.png")
    fig.savefig(out_path)
    plt.close(fig)
    written.append(out_path)

    csv_path = os.path.join(plot_dir, "prdf_best_sqs.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Pair", "Distance_A", "PRDF"])
        for pair in sorted(prdf):
            for distance, value in zip(distances[pair], prdf[pair]):
                writer.writerow([f"{pair[0]}-{pair[1]}", f"{distance:.4f}", f"{value:.6f}"])
    written.append(csv_path)


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main(argv=None):
    config = build_config(argv)

    import numpy as np
    from ase.build import make_supercell
    from icet import ClusterSpace
    from icet.tools.structure_generation import (
        _get_sqs_cluster_vector, _validate_concentrations)
    from pymatgen.io.ase import AseAtomsAdaptor

    banner("ICET SQS search")

    atoms, source = load_primitive_atoms(config)
    nx, ny, nz = config["supercell"]
    transformation = np.diag([nx, ny, nz])
    supercell = make_supercell(atoms, transformation)
    total_sites = len(supercell)

    chemical_symbols = build_chemical_symbols(config, atoms)
    cutoffs = list(config["cutoffs"])

    if config.get("sublattice_mode"):
        concentrations = config["target_concentrations"]
    else:
        concentrations, counts = achievable_global_concentrations(
            config["target_concentrations"], total_sites)

    print(f"  Structure          : {config.get('structure_name') or source}")
    print(f"  Source             : {source}")
    print(f"  Primitive sites    : {len(atoms)}")
    print(f"  Supercell          : {nx}x{ny}x{nz}  ({total_sites} atoms)")
    print(f"  Cluster cutoffs    : {', '.join(f'{c:g}' for c in cutoffs)} Å")
    method = config.get("method", "monte_carlo")
    print(f"  Method             : "
          f"{'exhaustive enumeration' if method == 'enumeration' else 'Monte Carlo (annealing)'}")
    if method == "enumeration":
        # Enumeration is deterministic: seeds and repeated runs change nothing.
        print(f"  Runs               : 1 (enumeration is exhaustive and deterministic)")
    else:
        print(f"  Monte Carlo steps  : {config['n_steps']:,} per run")
        _parallel = max(1, min(int(config.get("parallel_runs", 1) or 1),
                               int(config["n_runs"])))
        print(f"  Runs               : {config['n_runs']}"
              + (f"  ({_parallel} in parallel)" if _parallel > 1 else "  (sequential)"))
        print(f"  Base random seed   : {config['base_seed'] or 'random'}")
    print(f"  Composition mode   : {'sublattice-specific' if config.get('sublattice_mode') else 'global'}")
    if config.get("sublattice_mode"):
        for sublattice, sublattice_conc in concentrations.items():
            pretty = ", ".join(f"{element}: {fraction:.4f}"
                               for element, fraction in sorted(sublattice_conc.items()))
            print(f"    sublattice {sublattice}   : {pretty}")
    else:
        for element in sorted(concentrations):
            print(f"    {element:<18}: {concentrations[element]:.4f}"
                  f"  ({counts[element]} atoms)")
    output_dir = os.path.abspath(config["output_dir"])
    print(f"  Output directory   : {output_dir}")

    banner("Building the cluster space")
    cluster_space = ClusterSpace(atoms, cutoffs, chemical_symbols)
    print(cluster_space)
    orbit_count = len(cluster_space.as_list)
    print(f"\n  Orbits in the cluster space: {orbit_count}")

    overlap = sublattice_symbol_overlap(cluster_space, supercell)
    if overlap:
        banner("❌ This composition cannot be searched")
        print("  ICET builds one sublattice per set of allowed species, and mchammer cannot")
        print("  tell two atoms of the same species apart. A species may therefore appear on")
        print("  only ONE active sublattice, but here:")
        print("")
        for symbol in sorted(overlap):
            print(f"    {symbol:<4} is allowed on sublattices {', '.join(overlap[symbol])}")
        print("")
        for sublattice in cluster_space.get_sublattices(supercell):
            if len(sublattice.chemical_symbols) > 1:
                print(f"    sublattice {sublattice.symbol}: "
                      f"{', '.join(sublattice.chemical_symbols)}")
        print("")
        print("  Two ways out:")
        print("    - give those Wyckoff positions the SAME set of elements, so they become one")
        print("      sublattice sharing one concentration, or")
        print("    - make the element sets disjoint, so no species is shared between them.")
        rule()
        return 1

    validated = _validate_concentrations(concentrations, cluster_space)
    target_vector = _get_sqs_cluster_vector(cluster_space, validated)

    # Nothing is written until here, so a rejected configuration leaves no litter.
    os.makedirs(output_dir, exist_ok=True)
    structures_dir = os.path.join(output_dir, "structures")
    os.makedirs(structures_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "icet_sqs.log")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(name)s: %(levelname)s  %(message)s"))
    logging.getLogger("icet").addHandler(file_handler)

    if method == "enumeration":
        if config.get("estimate_only"):
            # --estimate actually runs the enumerator for a few seconds, which is
            # the only honest way to say how long the whole thing would take.
            estimate = estimate_enumeration(
                cluster_space, supercell, concentrations,
                probe_seconds=float(config.get("enumeration_probe_seconds", 5.0)),
                max_candidates=float(config.get("enumeration_max_candidates", 5e7)))
            print_enumeration_estimate(estimate)
            print("  (--estimate given: stopping here without searching.)")
            return 0

        # A normal run only uses the closed-form numbers, so it does not pay for
        # an extra pass over the enumerator just to describe itself.
        scale = enumeration_scale(cluster_space, supercell, concentrations)
        banner("Enumeration size")
        print_enumeration_scale(scale)
        cap = float(config.get("enumeration_max_candidates", 5e7))
        if scale["upper_bound"] > cap:
            print("")
            print(f"  This is above the safety cap of {cap:,.0f} candidates, so the")
            print("  enumeration would not finish in any reasonable time.")
            print("  Run with --estimate to measure it, use --method monte_carlo,")
            print("  or shrink the supercell. Raise enumeration_max_candidates in")
            print("  the CONFIG block to override.")
            rule()
            return 1

    banner("Running the search" if method == "enumeration"
           else f"Running {config['n_runs']} SQS search(es)")
    records = []
    results = []
    job_started = time.time()
    time_limit_seconds = float(config.get("time_limit_minutes") or 0) * 60.0
    interrupted = False

    n_runs = 1 if method == "enumeration" else int(config["n_runs"])
    parallel_runs = max(1, min(int(config.get("parallel_runs", 1) or 1), n_runs))
    plt = setup_matplotlib() if config.get("make_plots", True) else None
    written = []
    cv_dir = os.path.join(output_dir, "cluster_vector_plots")
    obj_dir = os.path.join(output_dir, "objective_plots")

    def finalize_run(run, sqs_atoms, elapsed, best_score, rows):
        """Save one run's structure, CSV and plots the moment it lands."""
        clusters = cluster_vector_analysis(cluster_space, sqs_atoms, target_vector)
        rmse, mean_abs, worst, exact, score = match_stats(clusters)
        total_mismatch, optimal_radius, perfect_shells = score_breakdown(clusters)

        structure = AseAtomsAdaptor.get_structure(sqs_atoms)
        structure, removed = strip_vacancies(structure)

        paths = save_structure(structure, structures_dir, f"sqs_run{run}",
                               config["output_formats"])
        write_cluster_vector_csv(
            os.path.join(output_dir, f"cluster_vector_run{run}.csv"), clusters)

        score_text = "n/a" if best_score is None else f"{best_score:.6f}"
        print(f"   ✅ run {run} finished in {elapsed:.1f} s"
              f" | best score {score_text}"
              f" | match score {score:.2f} %", flush=True)
        print(f"      score = sum|mismatch| {total_mismatch:.6f}"
              f" - perfect pair shells out to {optimal_radius:.4f} A"
              f" ({perfect_shells} shells)")
        print(f"      cluster-vector RMSE {rmse:.6f}, max |mismatch| {worst:.6f},"
              f" exact matches {exact}/{len(clusters)}")
        print(f"      formula: {structure.composition.formula}"
              + (f"  ({removed} vacancies removed)" if removed else ""))
        for path in paths:
            print(f"      → {os.path.relpath(path, output_dir)}")

        result = {
            "run": run,
            "structure": structure,
            "elapsed": elapsed,
            "best_score": best_score,
            "clusters": clusters,
            "rmse": rmse,
            "match_score": score,
            "total_mismatch": total_mismatch,
            "optimal_radius": optimal_radius,
            "perfect_shells": perfect_shells,
            "paths": paths,
        }
        results.append(result)

        # This run's own figures, so a long job produces output as it goes
        # rather than only at the very end.
        if plt is not None:
            os.makedirs(cv_dir, exist_ok=True)
            cv_path = os.path.join(cv_dir, f"run{run}_cluster_vector.png")
            plot_cluster_vector(plt, clusters, f"Cluster vector matching - run {run}",
                                cv_path, best_score)
            written.append(cv_path)
            print(f"      → {os.path.relpath(cv_path, output_dir)}")
            if rows:
                plot_objective(plt, rows, [run], obj_dir, written)
                print(f"      → {os.path.relpath(obj_dir, output_dir)}/run{run}_objective.png")
        return result

    if method != "enumeration" and parallel_runs > 1:
        print(f"\n▶ {n_runs} runs, {parallel_runs} in parallel")
        try:
            records = run_parallel_sqs(
                atoms=atoms, supercell=supercell, cutoffs=cutoffs,
                chemical_symbols=chemical_symbols, concentrations=concentrations,
                n_steps=int(config["n_steps"]), base_seed=int(config["base_seed"]),
                n_runs=n_runs, parallel_runs=parallel_runs,
                log_every_seconds=float(config["log_every_seconds"]) or 5.0,
                on_result=finalize_run)
        except KeyboardInterrupt:
            print("\n🛑 Interrupted - keeping the runs finished so far.")
            interrupted = True
        results.sort(key=lambda result: result["run"])
    else:
        for run in range(1, n_runs + 1):
            if time_limit_seconds and (time.time() - job_started) > time_limit_seconds:
                print(f"\n⏱️  Time limit of {config['time_limit_minutes']} minutes reached "
                      f"- not starting run {run}.")
                break

            seed = config["base_seed"] + (run - 1) if config["base_seed"] else 0
            if method == "enumeration":
                print("\n▶ Exhaustive enumeration")
            else:
                print(f"\n▶ Run {run}/{n_runs}  (seed: {seed or 'random'})")
            run_records = []
            try:
                if method == "enumeration":
                    sqs_atoms, elapsed, best_score, _ = run_enumeration(
                        cluster_space, supercell, concentrations,
                        log_every_seconds=float(config["log_every_seconds"]) or 5.0,
                        count_timeout=float(config.get("enumeration_count_timeout", 30.0)))
                else:
                    sqs_atoms, elapsed, best_score = run_single_sqs(
                        cluster_space, supercell, concentrations, int(config["n_steps"]),
                        seed, run, run_records, float(config["log_every_seconds"]))
            except KeyboardInterrupt:
                print("\n🛑 Interrupted - keeping the runs finished so far.")
                interrupted = True
                break
            except Exception as exc:
                print(f"   ❌ run {run} failed: {exc}")
                continue

            records.extend(run_records)
            finalize_run(run, sqs_atoms, elapsed, best_score, run_records)

    if not results:
        print("\n❌ No run produced a structure - nothing to save.")
        return 1

    # ----- progress CSV ---------------------------------------------------- #
    # Enumeration reports no MC steps, so there is nothing to tabulate.
    progress_path = os.path.join(output_dir, "sqs_progress.csv")
    if not records:
        progress_path = None
    if progress_path:
        with open(progress_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "run", "step", "total_steps", "accepted_trials", "temperature",
                "best_score", "elapsed_seconds", "timestamp"])
            writer.writeheader()
            writer.writerows(records)

    # ----- best run -------------------------------------------------------- #
    def sort_key(result):
        return (result["best_score"] if result["best_score"] is not None
                else float("inf"), result["rmse"])

    best = min(results, key=sort_key)

    banner("Best structure")
    best_score_text = ("n/a" if best["best_score"] is None
                       else f"{best['best_score']:.6f}")
    print(f"  Run {best['run']}  |  best score {best_score_text}"
          f"  |  match score {best['match_score']:.2f} %")

    for path in best["paths"]:
        extension = os.path.basename(path)
        if extension.endswith("_POSCAR"):
            destination = os.path.join(output_dir, "POSCAR_best_overall")
        elif extension.endswith(".cif"):
            destination = os.path.join(output_dir, "best_sqs.cif")
        elif extension.endswith(".lmp"):
            destination = os.path.join(output_dir, "best_sqs.lmp")
        elif extension.endswith(".xyz"):
            destination = os.path.join(output_dir, "best_sqs.xyz")
        else:
            continue
        shutil.copyfile(path, destination)
        print(f"  🏆 {os.path.relpath(destination, output_dir)}")

    # ----- plots ----------------------------------------------------------- #
    # Each run already wrote its own figures as it finished; what is left are
    # the comparisons across runs and the PRDF of the winner.
    if plt is not None:
        if records and len(results) > 1:
            banner("📈 Objective function - all runs")
            plot_objective(plt, records, [r["run"] for r in results], obj_dir, written)

        if len(results) > 1:
            banner("📊 Cluster vector matching - all runs")
            os.makedirs(cv_dir, exist_ok=True)
            out_path = os.path.join(cv_dir, "all_runs_cluster_vector.png")
            plot_cluster_vector_overlay(plt, results, out_path)
            written.append(out_path)

        banner("📉 PRDF of the best structure")
        plot_prdf(plt, best["structure"], float(config["prdf_cutoff"]),
                  float(config["prdf_bin_size"]),
                  os.path.join(output_dir, "prdf_plots"), written)

        banner("Figures written")
        for path in sorted(set(written)):
            print(f"   ✅ {os.path.relpath(path, output_dir)}")

    # ----- summary --------------------------------------------------------- #
    summary_lines = [
        "ICET SQS search summary",
        f"generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"structure        : {config.get('structure_name') or source}",
        f"supercell        : {nx}x{ny}x{nz} ({total_sites} atoms)",
        f"cutoffs          : {', '.join(f'{c:g}' for c in cutoffs)} A",
        (f"method           : exhaustive enumeration" if method == "enumeration"
         else f"MC steps per run : {config['n_steps']}"),
        f"runs completed   : {len(results)}" + (" (interrupted)" if interrupted else ""),
        f"orbits           : {orbit_count}",
        "",
        f"{'run':>4}  {'best score':>12}  {'sum|dev|':>10}  {'perfect':>9}"
        f"  {'match %':>8}  {'seconds':>9}",
    ]
    for result in results:
        score_text = "n/a" if result["best_score"] is None else f"{result['best_score']:.6f}"
        summary_lines.append(
            f"{result['run']:>4}  {score_text:>12}  {result['total_mismatch']:>10.6f}"
            f"  {result['optimal_radius']:>9.4f}  {result['match_score']:>8.2f}"
            f"  {result['elapsed']:>9.1f}")
    summary_lines += [
        "",
        "score   = sum|deviation| minus the radius (A) out to which every pair shell",
        "          matches the target exactly ('perfect'); lower is better.",
        "match % = 100 (1 - RMSE) with each orbit weighted by 1/radius, so the near",
        "          shells dominate, the same priority the score has.",
    ]
    summary_lines += [
        "",
        f"best run         : {best['run']}",
        f"best formula     : {best['structure'].composition.formula}",
    ]
    summary_text = "\n".join(summary_lines)

    summary_path = os.path.join(output_dir, "sqs_summary.txt")
    with open(summary_path, "w") as handle:
        handle.write(summary_text + "\n")

    banner("Summary")
    print(summary_text)
    print("")
    print(f"  Everything written to: {output_dir}")
    rule()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n🛑 Aborted by the user.")
        sys.exit(130)

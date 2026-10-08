"""Locations of the repository's data and output folders.

All paths are resolved relative to the repository root, so the figure and simulation scripts
can be run from any working directory.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
SWEEPS = DATA / "sweeps"                    # frequency-response sweeps and mean-field eigenvalues
SYNCHRONY = DATA / "synchrony"              # pairwise spike cross-correlations
SPONTANEOUS = DATA / "spontaneous"          # spontaneous (undriven) LFP recordings
DRIVEN = DATA / "driven"                    # driven example simulations with full time series
MEAN_FIELD = DATA / "mean_field"            # stored mean-field / spiking-network comparisons

OUTPUT = ROOT / "output"                    # figures are written here


def output_path(name):
    """Path in the output folder (created if needed) for a file called `name`."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    return OUTPUT / name

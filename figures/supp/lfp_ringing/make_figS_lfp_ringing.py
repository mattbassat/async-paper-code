"""Supplementary figures: ringing of the inhibitory-current LFP after the offset of a sinusoidal
drive, without and with a synaptic delay."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data as D
import panels as P
import ringdown as RD
from asyncresp.analysis import aggregate_impulse_results, sampled_resonance_peaks
from asyncresp.io import load_pickle
from asyncresp.paths import OUTPUT, SWEEPS

K_SIGMA = 2.0
EXAMPLE_SEED = 300
DELAY_DRIVE_HZ = 45.0      # empirical resonance of the +5 mV network with 2 ms delay

CASES = {
    "nodelay": dict(roles={"OD": -15.0, "boundary": -5.0, "UD": 5.0}, xlim=(-16, 16),
                    spectra=[-15.0, -5.0, 5.0, 15.0]),
    "delay": dict(roles={"OD": -15.0, "boundary": -10.0, "UD": 5.0}, xlim=(-16, 7.5),
                  spectra=[-15.0, -10.0, 5.0]),
}


def resonance_lines(case):
    """Dashed lines of panel c: {Delta_EL: frequency (Hz)}."""
    if case == "nodelay":
        sweep = aggregate_impulse_results(load_pickle(SWEEPS / "impulse_results_r2b0a1_final.pkl"),
                                          "delta_EL")
        peaks = sampled_resonance_peaks(sweep, "delta_EL")
        return {5.0: peaks[5.0], 15.0: peaks[15.0]}
    return {5.0: DELAY_DRIVE_HZ}


def make(case):
    P.style()
    c = CASES[case]
    roles = c["roles"]
    run = D.load_driven(case)
    delta_EL, mean, sem = RD.significant_cycles_summary(run, K_SIGMA)
    examples = {role: D.example_run(run, d, EXAMPLE_SEED) for role, d in roles.items()}
    spectra = D.spontaneous_spectra(case, c["spectra"])

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.6, wspace=0.32,
                          left=0.08, right=0.98, top=0.84, bottom=0.09)
    P.panel_a_examples(fig, gs[0, :], examples, roles, K_SIGMA)
    P.panel_b_cycles(fig.add_subplot(gs[1, 0]), delta_EL, mean, sem, roles, c["xlim"])
    P.panel_c_spectra(fig.add_subplot(gs[1, 1]), spectra, resonance_lines(case))
    P.delta_el_key(fig, sorted(spectra))

    out = OUTPUT / "supp"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figS_lfp_ringing_{case}.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for case in CASES:
        make(case)

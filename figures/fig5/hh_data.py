"""Data for Fig. 5: the frequency-response sweep of the Hodgkin-Huxley network and the
spontaneous LFP spectra of four of its networks."""
import numpy as np
import pandas as pd

from asyncresp.analysis import rocket_colors, welch_psd  # noqa: F401
from asyncresp.io import load_pickle
from asyncresp.paths import SPONTANEOUS, SWEEPS

SWEEP_FILE = SWEEPS / "impulse_results_hh_grid_final.pkl"
SPONT_DIR = SPONTANEOUS / "hh"

SPECTRA_DEL = [-20.0, -10.0, 0.0, 10.0]      # the four networks of Fig. 5c (Delta_EL, mV)
SEEDS = list(range(300, 310))
WINDOW = (2000.0, 10000.0)                   # spectral analysis window (ms), as in Fig. 1e
SLOW_F = 0.5                                 # the slow input of Fig. 5b (Hz)
LOW_FMAX = 10.0                              # a resonance is only counted above 10 Hz


def sweep():
    """The frequency-response sweep as a DataFrame, with Delta_EL (mV) and input frequency f
    (Hz)."""
    df = pd.DataFrame(load_pickle(SWEEP_FILE))
    df["delta_EL"] = df.El_exc - df.EL_inh
    df["f"] = (1000 / df.period).round(6)
    return df


def response_curves(df):
    """Mean and s.d. over seeds of the E response amplitude, per Delta_EL and frequency."""
    return df.groupby(["delta_EL", "f"]).amp_exc.agg(["mean", "std"]).reset_index()


def peak_frequencies(df):
    """Frequency of the maximum of the mean evoked E response of each network."""
    g = df[df.f >= SLOW_F].groupby(["delta_EL", "f"]).amp_exc.mean().reset_index()
    return {float(d): float(sub.f.values[int(np.argmax(sub.amp_exc.values))])
            for d, sub in g.groupby("delta_EL")}


def spont_path(delta_el, seed):
    tag = f"{delta_el:+.1f}".replace("+", "p").replace("-", "m")
    return SPONT_DIR / f"hh_spont_{tag}_{seed}.pkl"


def spont_spectrum(delta_el, seeds=SEEDS):
    """Welch PSD of the spontaneous LFP proxy of one network, averaged over seeds."""
    psds, freqs = [], None
    for s in seeds:
        o = load_pickle(spont_path(delta_el, s))
        freqs, p = welch_psd(o["lfp_inh"], o["dt_lfp_ms"], WINDOW)
        psds.append(p)
    return freqs, np.mean(psds, axis=0)

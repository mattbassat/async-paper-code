"""Loading of the stored simulations used by the LFP-ringing supplementary figures."""
import numpy as np

from asyncresp.analysis import welch_psd
from asyncresp.io import load_pickle
from asyncresp.paths import SPONTANEOUS, SUPP

DATA = SUPP / "lfp_ringing"
DRIVEN_FILES = {"nodelay": "driven_r2b0_nodelay_f50.pkl", "delay": "driven_r4b0_delay2_f45.pkl"}
SPONT_DELAY_FILE = "spont_r4b0_delay2.pkl"
SEEDS = list(range(300, 310))
ANALYSIS_WINDOW = (2000.0, 10000.0)     # ms of the spontaneous runs used for the spectra


def load_driven(case):
    """Driven-run record of `case` ('nodelay' or 'delay')."""
    return load_pickle(DATA / DRIVEN_FILES[case])


def example_run(run, delta_EL, seed):
    """A single driven run as a dict of the fields needed by the ringdown fit."""
    i = int(np.flatnonzero(run["delta_EL"] == delta_EL)[0])
    j = int(np.flatnonzero(run["seeds"] == seed)[0])
    keys = ("period_ms", "oscillation_onset", "oscillation_offset", "total_ms", "t_state")
    return dict({k: run[k] for k in keys}, lfp_inh=run["lfp_inh"][i, j])


def _mean_psd(traces, dt_ms):
    psds = []
    for x in traces:
        f, p = welch_psd(np.asarray(x).astype(float), dt_ms, window=ANALYSIS_WINDOW)
        psds.append(p)
    return f, np.mean(psds, axis=0)


def spontaneous_spectra(case, delta_ELs):
    """Seed-averaged Welch spectra {Delta_EL: (freqs, psd)} of the spontaneous lfp_inh."""
    out = {}
    if case == "nodelay":
        for d in delta_ELs:
            tag = f"{d:+.0f}".replace("+", "p").replace("-", "m")
            recs = [load_pickle(SPONTANEOUS / "adex_r2b0" / f"r2b0_spontWS_{tag}_0_{s}.pkl")
                    for s in SEEDS]
            out[d] = _mean_psd([r["lfp_inh"] for r in recs], recs[0]["dt_lfp_ms"])
    else:
        rec = load_pickle(DATA / SPONT_DELAY_FILE)
        for d in delta_ELs:
            i = int(np.flatnonzero(rec["delta_EL"] == d)[0])
            out[d] = _mean_psd(rec["lfp_inh"][i], rec["dt_lfp_ms"])
    return out

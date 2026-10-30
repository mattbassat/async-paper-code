"""Frequency-response sweeps of the AdEx network (Fig. 1c-f): loading, averaging over seeds, and
the empirical resonance frequency of each network."""
from asyncresp.analysis import aggregate_impulse_results, sampled_resonance_peaks  # noqa: F401
from asyncresp.io import load_pickle
from asyncresp.paths import SWEEPS

# sweep name -> (stored files, swept parameter)
SWEEP_FILES = {
    "delta_EL": (["impulse_results_r2b0a1_final.pkl"], "delta_EL"),
    "delta_gL": (["impulse_results_r2b0a1gLexc10_delta_gL_final.pkl"], "delta_gL"),
    "b": (["impulse_results_changing_b_r4_final.pkl"], "b_val"),
    "Qee": ([f"impulse_results_r2b0a1_Qee_{ph}.pkl" for ph in ("slow", "slow_extra", "rest", "rest_extra")],
            "Qee"),
    "Qei": ([f"impulse_results_r2b0a1_Qei_{ph}.pkl" for ph in ("slow", "slow_extra", "rest")], "Qei"),
}


def load_sweep(name):
    """Seed-averaged frequency responses of one sweep, and the name of its parameter column."""
    files, param_name = SWEEP_FILES[name]
    runs = []
    for fname in files:
        runs.extend(load_pickle(SWEEPS / fname))
    return aggregate_impulse_results(runs, param_name), param_name

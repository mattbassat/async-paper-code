"""Slow-input responses of the spiking network and the matching mean-field quantities."""
import numpy as np
import pandas as pd

from asyncresp.paths import SWEEPS

SLOW_F = 0.5                               # slow-input frequency (Hz)

# The three parameter sets: stored sweep files, column title, and the colour / marker that
# identifies each set in panels a and b.
SETS = {
    "r2b0": dict(imp="impulse_results_r2b0a1_final.pkl", eig="eigen_df_r2b0_dense.pkl",
                 label=r"$\nu_{drive}$=2 Hz, $b$=0 pA", short="2 Hz, 0 pA",
                 colour="#2a78d6", marker="o"),
    "r4b0": dict(imp="impulse_results_r4b0a1_final.pkl", eig="eigen_df_r4b0_dense.pkl",
                 label=r"$\nu_{drive}$=4 Hz, $b$=0 pA", short="4 Hz, 0 pA",
                 colour="#eb6834", marker="s"),
    "r4b60": dict(imp="impulse_results_r4b60a1_final.pkl", eig="eigen_df_r4b60_dense.pkl",
                  label=r"$\nu_{drive}$=4 Hz, $b$=60 pA", short="4 Hz, 60 pA",
                  colour="#1baf7a", marker="^"),
}


def load_set(key):
    """Seed-averaged SNN responses and mean-field table of one parameter set."""
    s = SETS[key]
    imp = pd.concat([pd.DataFrame(pd.read_pickle(SWEEPS / s["imp"])),
                     pd.DataFrame(pd.read_pickle(SWEEPS / s["imp"].replace("_final.pkl",
                                                                          "_extra.pkl")))],
                    ignore_index=True)
    imp["delta_EL"] = (imp.EL_exc - imp.EL_inh).round(3)
    imp["f"] = (1000 / imp.period).round(6)
    agg = imp.groupby(["delta_EL", "f"]).agg(
        amp_e=("amp_exc", "mean"), amp_e_sd=("amp_exc", "std"),
        amp_i=("amp_inh", "mean"), amp_i_sd=("amp_inh", "std")).reset_index()

    e = pd.read_pickle(SWEEPS / s["eig"]).sort_values("delta_EL").reset_index(drop=True)
    A, B, C, D, T = e.dfedot_dfe, e.dfedot_dfi, e.dfidot_dfe, e.dfidot_dfi, e.T_stationary
    e["det"] = A * D - B * C
    e["zeta"] = -(A + D) / (2 * np.sqrt(e["det"]))
    e["He0"] = -D / (T * e["det"]) - 1
    e["Hi0"] = C / (T * e["det"])
    return agg, e


def slow_table(agg, e):
    """SNN responses at the slow-input frequency, with the mean-field damping ratio of the
    nearest Delta_EL of the dense grid."""
    s = agg[np.isclose(agg.f, SLOW_F)].sort_values("delta_EL").copy()
    s["zeta"] = [e.zeta.iloc[int(np.argmin(np.abs(e.delta_EL.values - d)))] for d in s.delta_EL]
    return s

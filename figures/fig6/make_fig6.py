"""Fig. 6: the state-dependent mean-field time constant T = tau_V."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supp"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import supp_common as SC
from asyncresp.io import load_pickle
from asyncresp.paths import OUTPUT, SWEEPS
from asyncresp.style import INK2

SETS = [("r2b0", r"$\nu_{drive}$ = 2 Hz, $b$ = 0 pA"),
        ("r4b0", r"$\nu_{drive}$ = 4 Hz, $b$ = 0 pA"),
        ("r4b60", r"$\nu_{drive}$ = 4 Hz, $b$ = 60 pA")]
XLIM, XTICKS = (-15.5, 15.5), [-15, -5, 5, 15]
T_COL = "#24448e"


def make():
    SC.style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.1), sharey=True,
                             gridspec_kw=dict(left=0.08, right=0.99, top=0.95, bottom=0.2,
                                              wspace=0.10))
    for ax, (tag, title) in zip(axes, SETS):
        df = load_pickle(SWEEPS / f"eigen_df_{tag}_dense.pkl").sort_values("delta_EL")
        ei = SC.eigen_from_df(df)
        SC._shade(ax, *XLIM, ei["xcrit"])
        ax.plot(df.delta_EL, 1e3 * df.T_stationary, color=T_COL, lw=1.4, zorder=3)
        ax.set_xlim(XLIM); ax.set_xticks(XTICKS); ax.set_ylim(0, 12)
        ax.set_xlabel(r"$\Delta_{E_L}$ (mV)")
        ax.text(0.5 * (XLIM[0] + ei["xcrit"]), 0.6, "OD", ha="center", va="bottom", fontsize=6.5,
                weight="bold", color=INK2)
        ax.text(0.5 * (ei["xcrit"] + XLIM[1]), 0.6, "UD", ha="center", va="bottom", fontsize=6.5,
                weight="bold", color=SC.UD_INK)
    axes[0].set_ylabel(r"$T = \tau_V$ (ms)")
    out = OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig6.png", dpi=300)
    fig.savefig(out / "fig6.pdf")
    return fig


if __name__ == "__main__":
    make()

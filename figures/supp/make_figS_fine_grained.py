"""Supplementary figure: fine-grained SNN responsiveness near the responsiveness peaks."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import supp_common as SC
from asyncresp.analysis import aggregate_impulse_results
from asyncresp.io import load_pickle
from asyncresp.paths import OUTPUT, SUPP
from asyncresp.style import E_COL, I_COL

DATA = SUPP / "fine_grained"
TAGS = ["r2b0a1", "r4b0a1", "r4b60a1"]         # figure columns
XTICKS = {"r2b0a1": [-10, -8, -6, -4, -2, 0], "r4b0a1": [-14, -12, -10, -8, -6],
          "r4b60a1": [-10, -8, -6, -4, -2, 0]}
COINCIDE_TOL = 0.15                            # mV
LW, LW_MAX, ALPHA = 1.1, 0.9, 0.3


def max_lines(ax, x_exc, x_inh):
    """Dashed lines at the Delta_EL of the maximal mean E and I responses."""
    if abs(x_exc - x_inh) < COINCIDE_TOL:
        ax.axvline(x_exc, color=E_COL, lw=LW_MAX, ls=(0, (3, 3)), zorder=1)
        ax.axvline(x_exc, color=I_COL, lw=LW_MAX, ls=(3, (3, 3)), zorder=1)
    else:
        ax.axvline(x_exc, color=E_COL, lw=LW_MAX, ls=(0, (3, 2)), zorder=1)
        ax.axvline(x_inh, color=I_COL, lw=LW_MAX, ls=(0, (3, 2)), zorder=1)


def panel(ax, tag):
    agg = aggregate_impulse_results(load_pickle(DATA / f"impulse_results_fg_slow_{tag}_final.pkl"),
                                    "delta_EL")
    s = agg[np.isclose(agg.frequency, SC.SLOW_F)].sort_values("delta_EL")
    x = s.delta_EL.values
    for key, col in (("exc", E_COL), ("inh", I_COL)):
        m, sd = s[f"amp_{key}_mean"].values, s[f"amp_{key}_std"].values
        ax.fill_between(x, m - sd, m + sd, color=col, alpha=ALPHA, lw=0, zorder=2)
        ax.plot(x, m, color=col, lw=LW, zorder=3)
    max_lines(ax, x[np.argmax(s.amp_exc_mean.values)], x[np.argmax(s.amp_inh_mean.values)])
    pad = 0.04 * (x[-1] - x[0])
    ax.set_xlim(x[0] - pad, x[-1] + pad)
    ax.set_xticks(XTICKS[tag])
    ax.set_xlabel(r"$\Delta_{E_L}$ (mV)")


def make():
    SC.style()
    fig = plt.figure(figsize=(7.2, 2.3))
    gs = fig.add_gridspec(1, 3, left=0.075, right=0.985, top=0.86, bottom=0.18, wspace=0.22)
    for c, tag in enumerate(TAGS):
        ax = fig.add_subplot(gs[0, c])
        panel(ax, tag)
        if c == 0:
            ax.set_ylabel("Evoked amplitude (Hz)")
    fig.legend(handles=[Patch(facecolor=E_COL, label="Excitatory"),
                        Patch(facecolor=I_COL, label="Inhibitory")],
               loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=2, frameon=False,
               fontsize=7, handlelength=1.4, columnspacing=1.6)
    out = OUTPUT / "supp"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figS_fine_grained.png", dpi=300)
    fig.savefig(out / "figS_fine_grained.pdf")
    return fig


if __name__ == "__main__":
    make()

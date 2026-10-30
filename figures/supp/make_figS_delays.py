"""Supplementary figure: responsiveness with uniform synaptic delays."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import supp_common as SC
from asyncresp.analysis import aggregate_impulse_results
from asyncresp.io import load_pickle
from asyncresp.paths import OUTPUT, SUPP

DELAYS = [(1, 10), (2, 5)]                     # (delay in ms, largest asynchronous Delta_EL)
GRID = [-15, -10, -5, 0, 5, 10, 15]            # Fig. 1c colour grid
FMT = lambda v: "0" if v == 0 else f"{v:+g}".replace("-", "−")


def make():
    SC.style()
    colors = SC.rocket(GRID)
    fig = plt.figure(figsize=(5.4, 4.6))
    gs = fig.add_gridspec(2, 2, left=0.12, right=0.98, top=0.86, bottom=0.09, wspace=0.12,
                          hspace=0.45)
    for c, (delay, dmax) in enumerate(DELAYS):
        runs = load_pickle(SUPP / "delays" / f"impulse_results_r4b0a1delay{delay}_extended.pkl")
        # the delay-1 file holds the Delta_EL = 10 mV runs below 100 Hz twice (identical
        # copies); keep one copy of each run
        runs = pd.DataFrame(runs).drop_duplicates().to_dict("records")
        agg = aggregate_impulse_results(runs, "delta_EL")
        agg = agg[agg.delta_EL <= dmax]
        ax_r = fig.add_subplot(gs[0, c])
        SC.response_panel(ax_r, agg, "delta_EL", colors, ymax=7.0, xrange=(0.5, 200))
        ax_s = fig.add_subplot(gs[1, c])
        s = agg[np.isclose(agg.frequency, SC.SLOW_F)].sort_values("delta_EL")
        for v, m, sd in zip(s.delta_EL, s.amp_exc_mean, s.amp_exc_std):
            ax_s.errorbar([v], [m], yerr=[sd], fmt="o", ms=3.0, color=colors[v], mec="white",
                          mew=0.4, ecolor=colors[v], elinewidth=0.7, capsize=2.4, capthick=0.7)
        ax_s.set_xlim(-17.5, 12.5); ax_s.set_xticks([-15, -10, -5, 0, 5, 10])
        ax_s.set_ylim(0, 1.5)
        ax_s.set_xlabel(r"$\Delta_{E_L}$ (mV)")
        ax_s.set_ylabel(f"E response to slow input\n($f = {SC.SLOW_F:g}$ Hz) (Hz)")
        if c:
            for ax in (ax_r, ax_s):
                ax.set_ylabel(""); ax.set_yticklabels([])
        else:
            first = (ax_r, ax_s)
    shown = [v for v in GRID if v <= DELAYS[0][1]]
    fig.legend(handles=[Patch(facecolor=colors[v], edgecolor="none", label=FMT(v)) for v in shown],
               title=r"$\Delta_{E_L}$ (mV)", ncol=len(shown), loc="upper center",
               bbox_to_anchor=(0.55, 0.998), frameon=True, fancybox=False, edgecolor="0.8",
               handlelength=1.0, handleheight=0.9, columnspacing=0.9, handletextpad=0.4,
               title_fontsize=7, fontsize=6.5, borderpad=0.45)
    for ax, s in zip(first, ("a.", "b.")):
        p = ax.get_position()
        fig.text(0.01, p.y1 + 0.02, s, fontsize=9, fontweight="bold", va="bottom", ha="left")
    out = OUTPUT / "supp"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figS_delays.png", dpi=300)
    fig.savefig(out / "figS_delays.pdf")
    return fig


if __name__ == "__main__":
    make()

"""Supplementary figure: higher afferent drive (nu_drive = 8 Hz, b = 0 pA, A = 1 Hz)."""
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
from asyncresp.paths import SUPP, OUTPUT

DATA = SUPP / "high_drive"
FILES = ["impulse_results_r8b0a1_final.pkl", "impulse_results_r8b0a1suppr8_final.pkl"]
XLIM, XTICKS = (-27.5, 17.5), [-25, -15, -5, 5, 15]
XLAB = r"$\Delta_{E_L}$ (mV)"


def make():
    SC.style()
    runs = []
    for f in FILES:
        runs.extend(load_pickle(DATA / f))
    agg = aggregate_impulse_results(runs, "delta_EL")
    ei = SC.eigen_from_df(load_pickle(DATA / "eigen_df_r8b0.pkl"))
    colors = SC.rocket(agg.delta_EL.unique())
    eigf = {v: float(np.interp(v, ei["x"], np.nan_to_num(ei["fd"], nan=-1))) for v in colors}
    eigf = {v: (f if f > 0 else np.nan) for v, f in eigf.items()}

    fig = plt.figure(figsize=(7.2, 2.85))
    gs = fig.add_gridspec(1, 3, left=0.075, right=0.985, top=0.75, bottom=0.14, wspace=0.48,
                          width_ratios=[1, 1.15, 1])
    axt = fig.add_subplot(gs[0, 0])
    SC.eigen_panel(axt, ei, XLIM, XLAB, XTICKS, show_legend=False)
    SC.eigen_key(fig, 0.0, 0.995)

    ax_r = fig.add_subplot(gs[0, 1])
    SC.response_panel(ax_r, agg, "delta_EL", colors, ymax=2.5, eigfreqs=eigf)
    fmt = lambda v: "0" if v == 0 else f"{v:+g}".replace("-", "\u2212")   # noqa: E731

    ax_s = fig.add_subplot(gs[0, 2])
    SC.slow_input_panel(ax_s, agg, "delta_EL", ei, XLIM, XTICKS, XLAB, ymax=2.5, colors=colors, crit_y=0.6)

    # Delta_EL key: one figure-level row centred over panels b and c, above the panel letters
    x_mid = 0.5 * (ax_r.get_position().x0 + ax_s.get_position().x1)
    fig.legend(handles=[Patch(facecolor=colors[v], edgecolor="none", label=fmt(v))
                        for v in sorted(colors)],
               title=r"$\Delta_{E_L}$ (mV)", ncol=len(colors), loc="upper center",
               bbox_to_anchor=(x_mid, 0.998), frameon=True, fancybox=False, edgecolor="0.8",
               handlelength=1.0, handleheight=0.9, columnspacing=0.9, handletextpad=0.4,
               title_fontsize=7, fontsize=6.5, borderpad=0.45)

    for ax, s in ((axt, "a."), (ax_r, "b."), (ax_s, "c.")):     # letters on one line
        x = ax.get_position().x0 - (0.065 if ax is not ax_r else 0.06)
        fig.text(x, 0.775, s, fontsize=9, fontweight="bold", va="bottom", ha="left")
    out = OUTPUT / "supp"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figS_high_drive.png", dpi=300)
    fig.savefig(out / "figS_high_drive.pdf")
    return fig


if __name__ == "__main__":
    make()

"""Supplementary figure: baseline (spontaneous) activity metrics of the three parameter sets."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import supp_common as SC
from asyncresp.paths import OUTPUT, SUPP
from asyncresp.style import E_COL, I_COL, INK, INK2

NAME = "figS_baseline"
DATA = SUPP / "baseline"
SETS = [("r2b0", 2.0, 0.0), ("r4b0", 4.0, 0.0), ("r4b60", 4.0, 60.0)]
EL_EXC = -70.0
XLIM, XTICKS = (-17.5, 17.5), [-15, -10, -5, 0, 5, 10, 15]
XLAB = r"$\Delta_{E_L}$ (mV)"
DEL_GRID = np.linspace(-17.5, 17.5, 141)
COND_COL, AFF_COL, BETA_COL = "#2a62d6", "#8d8d8d", "#e8892b"

# rows: (table, [(column, colour)], y scale, y label, y limits, y ticks, height of the
# Delta_EL^crit label as a fraction of the y range)
ROWS = [
    ("metrics", [("average_exc_rate", E_COL), ("average_inh_rate", I_COL)], 1.0,
     "Baseline firing rate (Hz)", (-1, 40), np.arange(0, 41, 10), 0.80),
    ("metrics", [("muVe_mV", E_COL), ("muVi_mV", I_COL)], 1.0,
     "Mean membrane\npotential (mV)", (-62, -50), np.arange(-62, -49, 2), 0.80),
    ("metrics", [("g_ratio", COND_COL)], 1.0,
     r"Conductance ratio $\mu_G / g_L$", (-1, 20), np.arange(0, 21, 5), 0.80),
    ("currents", [("aff_fraction", AFF_COL), ("abs_inh_exc_ratio", INK)], 1.0,
     "Current fraction", (0, 1), np.arange(0, 1.01, 0.2), 0.30),
    ("currents", [("balance_index", BETA_COL)], 1.0,
     r"Balance index $\beta$", (0, 0.30), np.arange(0, 0.301, 0.05), 0.30),
]


def load():
    rd = lambda f: pd.read_csv(DATA / f, float_precision="round_trip")
    return {"metrics": rd("baseline_metrics.csv"), "currents": rd("current_ratios.csv")}


def shade(ax, xcrit, ylim, crit_y):
    """OD (left) / UD (right) background, dotted boundary and its labels."""
    lo, hi = XLIM
    SC._shade(ax, lo, hi, xcrit)
    ax.lines[-1].set_linestyle(":")
    y0, y1 = ylim
    ax.text(xcrit - 0.012 * (hi - lo), y0 + crit_y * (y1 - y0), r"$\Delta_{E_L}^{\mathrm{crit.}}$",
            ha="right", va="center", fontsize=6.3, color=INK, zorder=6,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
    for x0, txt, col in ((0.5 * (lo + xcrit), "OD", INK2), (0.5 * (xcrit + hi), "UD", SC.UD_INK)):
        ax.text(x0, y0 + 0.955 * (y1 - y0), txt, ha="center", va="top", fontsize=6.5,
                weight="bold", color=col)


def points(ax, df, col, color, scale):
    g = df.groupby("delta_EL")[col].agg(["mean", "std"]).reset_index()
    ax.errorbar(g.delta_EL, g["mean"] * scale, yerr=g["std"] * scale, fmt="o", ms=3.0,
                color=color, mec="white", mew=0.4, ecolor=color, elinewidth=0.7, capsize=2.2,
                capthick=0.7, zorder=4, ls="none")


def make():
    data = load()
    SC.style()
    fig = plt.figure(figsize=(7.2, 9.4))
    gs = fig.add_gridspec(len(ROWS), 3, left=0.105, right=0.985, top=0.945, bottom=0.05,
                          wspace=0.12, hspace=0.42)
    first = []
    for c, (tag, nu, b) in enumerate(SETS):
        xcrit = SC.stored_crit(tag)                  # as in Fig. 3a and the text
        for r, (table, cols, scale, ylabel, ylim, yticks, crit_y) in enumerate(ROWS):
            ax = fig.add_subplot(gs[r, c])
            df = data[table][data[table].param_set == tag]
            shade(ax, xcrit, ylim, crit_y)
            for col, color in cols:
                points(ax, df, col, color, scale)
            ax.set_xlim(*XLIM); ax.set_xticks(XTICKS)
            ax.set_ylim(*ylim); ax.set_yticks(yticks)
            ax.set_xlabel(XLAB, labelpad=1.5)
            if c == 0:
                ax.set_ylabel(ylabel)
                first.append(ax)
            else:
                ax.set_yticklabels([])
    fig.legend(handles=[Patch(facecolor=E_COL, label="Excitatory"),
                        Patch(facecolor=I_COL, label="Inhibitory")],
               ncol=2, loc="upper center", bbox_to_anchor=(0.545, 0.995), frameon=True,
               fancybox=False, edgecolor="0.8", fontsize=6.8, handlelength=1.6,
               columnspacing=1.4, handletextpad=0.5, borderpad=0.45)
    fig.canvas.draw()
    for ax, s in zip(first, "abcde"):
        p = ax.get_position()
        fig.text(0.008, p.y1 + 0.006, f"{s}.", fontsize=9, fontweight="bold", va="bottom",
                 ha="left")
    out = OUTPUT / "supp"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{NAME}.png", dpi=300)
    fig.savefig(out / f"{NAME}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    make()

"""Supplementary figure: mean-field stability with a static instead of a state-dependent T."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "fig1"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import supp_common as SC
from asyncresp.paths import OUTPUT
from sweeps import load_sweep

T_STATIC = [(5e-3, "#6a3d9a"), (7.5e-3, "#1f9e89"), (10e-3, "#d9a400")]   # (T in s, colour)
DEL = np.linspace(-17.5, 17.5, 141)
XLIM, XTICKS = (-17.5, 17.5), [-15, -5, 5, 15]


def make():
    SC.style()
    agg, col = load_sweep("delta_EL")
    colors = SC.rocket(agg[col].unique())
    sweeps = [(None, "black", "state-dependent T")] + [(T, c, f"T = {T * 1e3:g} ms") for T, c in T_STATIC]
    eis = {lab: SC.mean_field_sweep(DEL, 2.0, -70.0, -70.0 - DEL, 0.0, T_fixed=T)
           for T, c, lab in sweeps}

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 3, left=0.085, right=0.985, top=0.87, bottom=0.085, hspace=0.55,
                          wspace=0.16)
    ax_a = fig.add_subplot(gs[0, :2])
    pa = ax_a.get_position()                      # shorten on the right: wider gap to panel b
    ax_a.set_position([pa.x0, pa.y0, pa.width - 0.045, pa.height])
    xc = eis["state-dependent T"]["xcrit"]       # the boundary does not depend on a static T
    SC._shade(ax_a, *XLIM, xc)
    ax_a.axhline(0, color=SC.INK2, lw=0.5)
    for T, c, lab in sweeps:
        ei = eis[lab]
        ax_a.plot(DEL, np.abs(np.imag(ei["lam1"])), color=c, lw=1.3)
        ax_a.plot(DEL, np.real(ei["lam1"]), color=c, lw=1.1, ls=(0, (3, 1.5)))
    ax_a.set_xlim(XLIM); ax_a.set_xticks(XTICKS)
    ax_a.set_xlabel(r"$\Delta_{E_L}$ (mV)"); ax_a.set_ylabel(r"Eigenvalue $\lambda_1$ (s$^{-1}$)")
    # keys above panel a: line style (Im / Re) on the left, timescale colours to its right
    p = ax_a.get_position()
    leg_ri = fig.legend([Line2D([], [], color=SC.INK, lw=1.3),
                         Line2D([], [], color=SC.INK, lw=1.1, ls=(0, (3, 1.5)))],
                        [r"$\mathrm{Im}\{\lambda_1\}$", r"$\mathrm{Re}\{\lambda_1\}$"],
                        loc="lower left", bbox_to_anchor=(p.x0, p.y1 + 0.005), ncol=1,
                        frameon=False, fontsize=6.3, handlelength=1.8, labelspacing=0.2)
    fig.canvas.draw()
    bb = leg_ri.get_window_extent().transformed(fig.transFigure.inverted())
    order = sweeps[1:] + sweeps[:1]                       # 5, 7.5, 10 ms, then state-dependent
    fig.legend([Line2D([], [], color=c, lw=1.3) for T, c, lab in order],
               [f"{T * 1e3:g}" if T is not None else "state-dependent" for T, c, lab in order],
               title=r"Mean-field timescale $T$ (ms)", loc="lower left",
               bbox_to_anchor=(bb.x1 + 0.02, p.y1 + 0.005), ncol=4, frameon=True,
               fancybox=False, edgecolor="0.8", fontsize=6.3, title_fontsize=6.8,
               handlelength=1.6, columnspacing=1.0, borderpad=0.4)

    panels = [(gs[0, 2], sweeps[0])] + [(gs[1, j], sw) for j, sw in enumerate(sweeps[1:])]
    for k, (cell, (T, c, lab)) in enumerate(panels):
        ax = fig.add_subplot(cell)
        SC.response_panel(ax, agg, col, colors, ymax=2.5,
                          eigfreqs=SC.eigenfreqs_at(eis[lab], colors))
        ax.set_title(lab if T is None else lab, fontsize=7, loc="left", pad=3, color=c)
        if k == 0:
            ax_b = ax
            ax.set_ylabel("")
        if k == 1:
            ax_c = ax
        if k >= 2:
            ax.set_ylabel(""); ax.set_yticklabels([])
    for ax, s in ((ax_a, "a."), (ax_b, "b."), (ax_c, "c.")):
        p = ax.get_position()
        fig.text(p.x0 - (0.06 if ax is ax_b else 0.075), p.y1 + 0.02, s, fontsize=9,
                 fontweight="bold", va="bottom", ha="left")
    out = OUTPUT / "supp"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figS_static_T.png", dpi=300)
    fig.savefig(out / "figS_static_T.pdf")
    return fig


if __name__ == "__main__":
    make()

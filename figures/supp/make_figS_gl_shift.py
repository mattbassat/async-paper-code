"""Supplementary figure: the Delta_gL sweep of Fig. 1d repeated at g_L,exc = 8 and 12 nS."""
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
from asyncresp import mean_field as MF
from asyncresp.analysis import aggregate_impulse_results
from asyncresp.io import load_pickle
from asyncresp.paths import OUTPUT, SUPP, SWEEPS

FILES = {8: SUPP / "gl_shift" / "impulse_results_r2b0a1gLexc8_delta_gL_final.pkl",
         10: SWEEPS / "impulse_results_r2b0a1gLexc10_delta_gL_final.pkl",
         12: SUPP / "gl_shift" / "impulse_results_r2b0a1gLexc12_delta_gL_final.pkl"}
XLIM, XTICKS = (-5.6, 5.6), [-5, -3, -1, 1, 3, 5]
XLAB = r"$\Delta_{g_L}$ (nS)"
GRID = np.linspace(-5.6, 5.6, 225)
CRIT = r"$\Delta_{g_L}^{\mathrm{crit.}}$"
RATE_MAX = 50.0     # Hz; above this the mean field has left the asynchronous branch


def mean_field(gle):
    """Mean-field quantities along Delta_gL. At g_L,exc = 8 nS the fixed point jumps to a
    saturated (>100 Hz) branch for Delta_gL above ~4 nS, which is not shown."""
    ei = SC.mean_field_sweep(GRID, 2.0, -70.0, -70.0, 0.0, gLe=gle * 1e-9, gLi=(gle + GRID) * 1e-9)
    n = len(GRID)
    fp = MF.fixed_point(np.full(n, 2.0), np.full(n, -70.0), np.full(n, -70.0), np.zeros(n),
                        gLe=np.full(n, gle * 1e-9), gLi=(gle + GRID) * 1e-9)
    off = np.ravel(fp["fe"]) > RATE_MAX
    for k in ("lam1", "lam2"):
        ei[k] = np.where(off, complex(np.nan, np.nan), ei[k])
    for k in ("B", "fd"):
        ei[k] = np.where(off, np.nan, ei[k])
    return ei


FMT = lambda v: "0" if v == 0 else f"{v:+g}".replace("-", "−")


def make():
    SC.style()
    fig = plt.figure(figsize=(7.2, 6.3))
    gs = fig.add_gridspec(3, 3, left=0.085, right=0.985, top=0.905, bottom=0.07, wspace=0.12,
                          hspace=0.55, height_ratios=[0.85, 1, 1])
    first, ylim = {}, None
    for c, gle in enumerate((8, 10, 12)):
        agg = aggregate_impulse_results(load_pickle(FILES[gle]), "delta_gL")
        ei = mean_field(gle)
        colors = SC.rocket(agg["delta_gL"].unique())

        ax_e = fig.add_subplot(gs[0, c])
        SC.eigen_panel(ax_e, ei, XLIM, XLAB, XTICKS, show_legend=False)
        ax_e.set_title(rf"$g_{{L_\mathrm{{exc}}}}$ = {gle} nS", fontsize=7, loc="left", pad=3)
        ax_r = fig.add_subplot(gs[1, c])
        SC.response_panel(ax_r, agg, "delta_gL", colors, ymax=3.0,
                          eigfreqs=SC.eigenfreqs_at(ei, colors))
        ax_s = fig.add_subplot(gs[2, c])
        SC.slow_input_panel(ax_s, agg, "delta_gL", ei, XLIM, XTICKS, XLAB, ymax=3.0,
                            colors=colors, crit_label=CRIT, crit_y=0.3)
        if c:
            for ax in (ax_e, ax_r, ax_s):
                ax.set_ylabel(""); ax.set_yticklabels([])
        else:
            first.update(e=ax_e, r=ax_r, s=ax_s)
        lo, hi = ax_e.get_ylim()                      # common eigenvalue scale across columns
        ylim = (lo, hi) if ylim is None else (min(ylim[0], lo), max(ylim[1], hi))
    for ax in fig.axes[0::3][:3]:
        ax.set_ylim(ylim)
    fig.legend(handles=[Patch(facecolor=colors[v], edgecolor="none", label=FMT(v))
                        for v in sorted(colors)],
               title=r"$\Delta_{g_L}$ (nS)", ncol=len(colors), loc="upper center",
               bbox_to_anchor=(0.53, 0.998), frameon=True, fancybox=False, edgecolor="0.8",
               handlelength=1.0, handleheight=0.9, columnspacing=0.9, handletextpad=0.4,
               title_fontsize=7, fontsize=6.5, borderpad=0.45)
    SC.eigen_key(fig, 0.03, 0.998)
    fig.canvas.draw()
    for key, s in (("e", "a."), ("r", "b."), ("s", "c.")):
        p = first[key].get_position()
        fig.text(0.008, p.y1 + 0.03, s, fontsize=9, fontweight="bold", va="bottom", ha="left")
    out = OUTPUT / "supp"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figS_gl_shift.png", dpi=300)
    fig.savefig(out / "figS_gl_shift.pdf")
    return fig


if __name__ == "__main__":
    make()

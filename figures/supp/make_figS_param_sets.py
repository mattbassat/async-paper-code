"""Supplementary figures: mean-field stability and SNN responses of the three parameter sets."""
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
from asyncresp.paths import OUTPUT, SUPP, SWEEPS

SETS = [("r2b0", 2.0, 0.0, r"$\nu_{drive}$ = 2 Hz, $b$ = 0 pA"),
        ("r4b0", 4.0, 0.0, r"$\nu_{drive}$ = 4 Hz, $b$ = 0 pA"),
        ("r4b60", 4.0, 60.0, r"$\nu_{drive}$ = 4 Hz, $b$ = 60 pA")]
VERSIONS = {   # name: (A, EL_exc, file pattern)
    "figS_param_sets_A1": (1, -70.0, lambda t: SWEEPS / f"impulse_results_{t}a1_final.pkl"),
    "figS_param_sets_A2": (2, -70.0, lambda t: SUPP / "amplitude_2hz" / f"impulse_results_{t}a2_final.pkl"),
    "figS_param_sets_EL65": (1, -65.0, lambda t: SUPP / "el_exc" / f"impulse_results_{t}a1EL65_final.pkl"),
    "figS_param_sets_EL75": (1, -75.0, lambda t: SUPP / "el_exc" / f"impulse_results_{t}a1EL75_final.pkl"),
}
XLIM, XTICKS = (-17.5, 17.5), [-15, -5, 5, 15]
XLAB = r"$\Delta_{E_L}$ (mV)"
DEL_GRID = np.linspace(-17.5, 17.5, 141)
FMT = lambda v: "0" if v == 0 else f"{v:+g}".replace("-", "−")


def make(name):
    A, ELe, path = VERSIONS[name]
    SC.style()
    fig = plt.figure(figsize=(7.2, 6.3))
    gs = fig.add_gridspec(3, 3, left=0.085, right=0.985, top=0.905, bottom=0.07, wspace=0.12,
                          hspace=0.55, height_ratios=[0.85, 1, 1])
    first = {}
    for c, (tag, nu, b, title) in enumerate(SETS):
        agg = aggregate_impulse_results(load_pickle(path(tag)), "delta_EL")
        ei = SC.mean_field_sweep(DEL_GRID, nu, ELe, ELe - DEL_GRID, b)
        if ELe == -70.0:                              # same boundary as Fig. 3a and the text
            ei["xcrit"] = SC.stored_crit(tag)
        colors = SC.rocket(agg.delta_EL.unique())

        ax_e = fig.add_subplot(gs[0, c])
        SC.eigen_panel(ax_e, ei, XLIM, XLAB, XTICKS, show_legend=False)
        ax_e.set_title(title, fontsize=7, loc="left", pad=3)
        ax_r = fig.add_subplot(gs[1, c])
        SC.response_panel(ax_r, agg, "delta_EL", colors, ymax=2.5 * A,
                          eigfreqs=SC.eigenfreqs_at(ei, colors))
        ax_s = fig.add_subplot(gs[2, c])
        SC.slow_input_panel(ax_s, agg, "delta_EL", ei, XLIM, XTICKS, XLAB, ymax=2.5 * A,
                            colors=colors, crit_y=0.6)
        if c:
            for ax in (ax_e, ax_r, ax_s):
                ax.set_ylabel(""); ax.set_yticklabels([])
        else:
            first.update(e=ax_e, r=ax_r, s=ax_s)
        if c == 0:
            ylim = ax_e.get_ylim()
        else:                                         # common eigenvalue scale across columns
            ylim = (min(ylim[0], ax_e.get_ylim()[0]), max(ylim[1], ax_e.get_ylim()[1]))
    for ax in fig.axes[0::3][:3]:
        ax.set_ylim(ylim)
    fig.legend(handles=[Patch(facecolor=colors[v], edgecolor="none", label=FMT(v))
                        for v in sorted(colors)],
               title=r"$\Delta_{E_L}$ (mV)", ncol=len(colors), loc="upper center",
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
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    for name in VERSIONS:
        make(name)

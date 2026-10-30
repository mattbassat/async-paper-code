"""Supplementary figure: alternative routes along the OD / UD continuum."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "fig1"))                 # the Fig. 1d sweep loader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

import supp_common as SC
from asyncresp.paths import OUTPUT
from sweeps import load_sweep

EL = -70.0

# sweep, column of the stored data, x label, dense mean-field grid, keyword factory for
# mean_field.fixed_point, Fig. 1d palette direction (True: high values are the OD side),
# axis limits and ticks
SWEEPS = [
    dict(name="delta_gL", x=r"$\Delta g_L$ (nS)", grid=np.linspace(-5, 5, 201),
         kw=lambda v: dict(gLe=10e-9, gLi=(10 + v) * 1e-9), reverse=False, nu=2.0,
         crit=r"$\Delta g_L^{\mathrm{crit.}}$",
         xlim=(-5.5, 5.5), xticks=[-5, 0, 5], key=r"$\Delta g_L$"),
    dict(name="b", x=r"$b$ (pA)", grid=np.linspace(0, 160, 161),
         kw=lambda v: dict(b=v), reverse=True, nu=4.0,
         crit=r"$b^{\mathrm{crit.}}$",
         xlim=(-8, 160), xticks=[0, 50, 100, 150], key=r"$b$"),
    dict(name="Qee", x=r"$Q_{ee}$ (nS)", grid=np.linspace(1.25, 1.60, 141),
         kw=lambda v: dict(Qee=v * 1e-9), reverse=False, nu=2.0,
         crit=r"$Q_{ee}^{\mathrm{crit.}}$",
         xlim=(1.23, 1.62), xticks=[1.3, 1.4, 1.5, 1.6], key=r"$Q_{ee}$"),
    dict(name="Qei", x=r"$Q_{ei}$ (nS)", grid=np.linspace(1.40, 2.00, 121),
         kw=lambda v: dict(Qei=v * 1e-9), reverse=True, nu=2.0,
         crit=r"$Q_{ei}^{\mathrm{crit.}}$",
         xlim=(1.37, 2.03), xticks=[1.4, 1.6, 1.8, 2.0], key=r"$Q_{ei}$"),
]


def mean_field_sweep(sw):
    """Eigenvalues, ringing frequency, boundary and Eq. 4 along the dense grid of one sweep."""
    xs = sw["grid"]
    kw = sw["kw"](xs)
    b = kw.pop("b", 0.0)
    return SC.mean_field_sweep(xs, sw["nu"], EL, EL, b, **kw)


def colour_key(fig, ax, colors, label):
    """Discrete colour strip above a response panel, labelled at its two ends."""
    vals = sorted(colors)
    p = ax.get_position()
    cax = fig.add_axes([p.x0 + 0.15 * p.width, p.y1 + 0.012, 0.7 * p.width, 0.010])
    cmap = ListedColormap([colors[v] for v in vals])
    cax.imshow(np.arange(len(vals))[None, :], aspect="auto", cmap=cmap,
               norm=BoundaryNorm(np.arange(len(vals) + 1) - 0.5, len(vals)))
    cax.set_xticks([0, len(vals) - 1])
    cax.set_xticklabels([f"{vals[0]:g}".replace("-", "\u2212"), f"{vals[-1]:g}"], fontsize=5.8)
    cax.tick_params(length=0, pad=1.0, labeltop=True, labelbottom=False)
    cax.set_yticks([])
    for s in cax.spines.values():
        s.set_visible(False)
    cax.text(-0.04, 0.5, label, transform=cax.transAxes, ha="right", va="center", fontsize=6.3)


def make():
    SC.style()
    fig = plt.figure(figsize=(7.2, 6.3))
    gs = fig.add_gridspec(3, 4, left=0.085, right=0.975, top=0.93, bottom=0.075,
                          height_ratios=[0.85, 1.0, 1.0], hspace=0.62, wspace=0.30)
    axes = {}
    for c, sw in enumerate(SWEEPS):
        mf = mean_field_sweep(sw)
        agg, col = load_sweep(sw["name"])
        od_left = not sw["reverse"]                # Fig. 1d convention: dark (OD) side first
        colors = SC.rocket(agg[col].unique(), reverse=sw["reverse"])

        ax_e = fig.add_subplot(gs[0, c])
        SC.eigen_panel(ax_e, mf, sw["xlim"], sw["x"], sw["xticks"], od_left=od_left,
                       show_legend=False)
        ax_r = fig.add_subplot(gs[1, c])
        SC.response_panel(ax_r, agg, col, colors, ymax=3.0,
                          eigfreqs=SC.eigenfreqs_at(mf, colors))
        ax_s = fig.add_subplot(gs[2, c])
        SC.slow_input_panel(ax_s, agg, col, mf, sw["xlim"], sw["xticks"], sw["x"], ymax=2.5,
                            colors=colors, od_left=od_left, crit_label=sw["crit"], crit_y=0.3)
        if c:
            for ax in (ax_e, ax_r, ax_s):
                ax.set_ylabel("")
            for ax in (ax_r, ax_s):                    # shared scales: labels on the first column
                ax.set_yticklabels([])
        axes[c] = (ax_e, ax_r, ax_s, colors, sw)
    fig.canvas.draw()
    for c, (ax_e, ax_r, ax_s, colors, sw) in axes.items():
        colour_key(fig, ax_r, colors, sw["key"])
    SC.eigen_key(fig, 0.03, 0.998)
    for letter, idx in (("a.", 0), ("b.", 1), ("c.", 2)):
        p = axes[0][idx].get_position()
        fig.text(0.008, p.y1 + 0.012, letter, fontsize=9, fontweight="bold", va="bottom",
                 ha="left")
    out = OUTPUT / "supp"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figS_alt_params.png", dpi=300)
    fig.savefig(out / "figS_alt_params.pdf")
    return fig


if __name__ == "__main__":
    make()

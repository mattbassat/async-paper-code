"""Fig. 2a: schematic of the two-population mean field and its equations.

The two rate variables (nu_e, nu_i) are drawn as populations with their four recurrent
projections and the external drive onto both.  Adaptation w is drawn inside the excitatory
population, since it is a state variable of the RS cells: driven by nu_e and feeding back
negatively on it.
"""
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from asyncresp.style import E_COL, I_COL, INK, INK2

E_FILL, I_FILL = "#cfe4c6", "#f6cfcb"
W_COL = "#8d8d8d"


def _arrow(ax, p0, p1, colour, rad=0.0, lw=1.6, ms=11, ls="-", zorder=4):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", color=colour,
                                 arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=zorder,
                                 capstyle="round"))


def _self_loop(ax, centre, r, colour, side=1, lw=1.6, ms=11):
    """A recurrent self-projection: an arc leaving and re-entering the same circle, on the
    outer side (side = -1: left, +1: right) of the population."""
    a0, a1 = (215, 145) if side < 0 else (-35, 35)
    p0 = (centre[0] + r * np.cos(np.radians(a0)), centre[1] + r * np.sin(np.radians(a0)))
    p1 = (centre[0] + r * np.cos(np.radians(a1)), centre[1] + r * np.sin(np.radians(a1)))
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={-1.45 if side < 0 else 1.45}",
                                 color=colour,
                                 arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
                                 shrinkA=0, shrinkB=0, zorder=3))


def draw_schematic(ax):
    ax.set_xlim(-2.32, 10.35); ax.set_ylim(-2.05, 1.72)
    ax.set_aspect("equal"); ax.axis("off")

    ce, ci, R = (0.0, 0.0), (3.5, 0.0), 1.15
    for c, fill, edge, lab, dy in ((ce, E_FILL, E_COL, r"$\nu_e$", 0.30),
                                   (ci, I_FILL, I_COL, r"$\nu_i$", 0.0)):
        ax.add_patch(Circle(c, R, facecolor=fill, edgecolor=edge, linewidth=1.6, zorder=2))
        ax.text(c[0], c[1] + dy, lab, fontsize=13, color=edge, ha="center", va="center",
                zorder=5)

    _self_loop(ax, ce, R, E_COL, side=-1)                # E -> E
    _self_loop(ax, ci, R, I_COL, side=+1)                # I -> I
    _arrow(ax, (ce[0] + 0.92 * R, +0.42 * R), (ci[0] - 0.92 * R, +0.42 * R), E_COL, rad=-0.30)
    _arrow(ax, (ci[0] - 0.92 * R, -0.42 * R), (ce[0] + 0.92 * R, -0.42 * R), I_COL, rad=-0.30)

    # adaptation w inside the excitatory population, with its negative feedback loop
    wc = (ce[0], ce[1] - 0.58)
    ax.add_patch(FancyBboxPatch((wc[0] - 0.26, wc[1] - 0.20), 0.52, 0.40,
                                boxstyle="round,pad=0.06", facecolor="white",
                                edgecolor=W_COL, linewidth=1.0, zorder=4))
    ax.text(wc[0], wc[1], r"$w$", fontsize=11, color=INK2, ha="center", va="center", zorder=5)
    _arrow(ax, (wc[0] - 0.46, wc[1] + 0.46), (wc[0] - 0.36, wc[1] - 0.02), W_COL, rad=0.55,
           lw=1.1, ms=8, zorder=5)
    _arrow(ax, (wc[0] + 0.36, wc[1] - 0.02), (wc[0] + 0.46, wc[1] + 0.46), W_COL, rad=0.55,
           lw=1.1, ms=8, zorder=5)
    ax.text(wc[0] + 0.70, wc[1] + 0.30, "$-$", fontsize=11, color=INK2, ha="left", va="center",
            zorder=5)

    # external drive: one arrow from each side of the label, aimed at each population's centre
    src = (0.5 * (ce[0] + ci[0]), -1.62)
    ax.text(src[0], src[1], r"$\nu_{drive}$", fontsize=10, color=INK, ha="center", va="center")
    for c, rad in ((ce, -0.28), (ci, 0.28)):
        side = np.sign(c[0] - src[0])
        p0 = np.array([src[0] + side * 0.52, src[1]])
        u = np.array([c[0], c[1]]) - p0
        p1 = np.array([c[0], c[1]]) - (R + 0.04) * u / np.hypot(*u)
        ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", color=INK,
                                     arrowstyle="-|>", mutation_scale=9, linewidth=1.3,
                                     shrinkA=3, shrinkB=0, zorder=4))

    # the mean-field equations
    x = 6.55
    ax.text(x - 0.40, 0.02, "{", fontsize=52, color=INK, ha="center", va="center",
            family="serif")
    for k, eq in enumerate((r"$T\,\dot{\nu}_e = F_e - \nu_e$",
                            r"$T\,\dot{\nu}_i = F_i - \nu_i$",
                            r"$\dot{w} = -\,w\,/\,\tau_w + b\,\nu_e$")):
        ax.text(x, 0.78 - 0.78 * k, eq, fontsize=10.5, color=INK, ha="left", va="center")

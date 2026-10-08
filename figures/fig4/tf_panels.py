"""Transfer-function panels of Fig. 4.

Each panel shows the mean-field transfer function F (output firing rate of the excitatory or
inhibitory population) over the plane of its two inputs: the total excitatory input
nu_e + nu_drive (recurrent plus afferent) and the inhibitory input nu_i.  Colours are on a
logarithmic scale; white lines are iso-rate contours (10 to 150 Hz).  The red star is the
network's fixed point (nu_e* + nu_drive, nu_i*) and the red dotted line marks nu_i*.  The inset
shows the one-dimensional cut of F along that line (nu_i = nu_i*), i.e. the gain curve seen by
a change in excitatory input at the operating point.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

from asyncresp import mean_field as MF

EL_EXC = -70.0                       # mV
RATE_MAX = 30.0                      # extent of both input axes (Hz)
GRID = np.linspace(0.0, RATE_MAX, 200)
CUT_MAX = 10.0                       # extent of the inset cut (Hz)
CUT_YMAX = 40.0
NORM = LogNorm(vmin=0.5, vmax=200.0)
CMAP = plt.get_cmap("viridis")
STAR = dict(marker="*", color="red", mec="white", zorder=8)


def operating_points(drive, delta_els, b=0.0):
    """Mean-field fixed points (nu_e*, nu_i*) for each Delta_EL (EL_inh = EL_exc - Delta_EL)."""
    el_inh = EL_EXC - np.asarray(delta_els, float)
    fp = MF.fixed_point(drive, EL_EXC, el_inh, b)
    return [dict(delta_el=d, fe=fe, fi=fi, w=w, el_inh=eli)
            for d, fe, fi, w, eli in zip(delta_els, fp["fe"], fp["fi"], fp["w"], el_inh)]


def rate_surface(pop, x_in, nu_i, op):
    """F_e or F_i at total excitatory input x_in = nu_e + nu_drive and inhibitory input nu_i.
    The excitatory population carries its stationary adaptation current; FS cells do not adapt.
    At zero input (no synaptic fluctuations) the rate is set to 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if pop == "e":
            F = MF.TF(MF.PRS, x_in, nu_i, op["w"], EL_EXC)[0]
        else:
            F = MF.TF(MF.PFS, x_in, nu_i, 0.0, op["el_inh"])[0]
    return np.nan_to_num(F, nan=0.0, posinf=0.0)


def draw_tf_panel(ax, pop, op, drive, show_xlabel=True, show_ylabel=True):
    """Heatmap of F_pop with fixed point, nu_i* line and inset cut.  Returns the mesh."""
    X, Y = np.meshgrid(GRID, GRID)
    Z = rate_surface(pop, X, Y, op)
    mesh = ax.pcolormesh(X, Y, np.clip(Z, NORM.vmin, NORM.vmax), cmap=CMAP, norm=NORM,
                         shading="auto", rasterized=True)
    ax.contour(X, Y, Z, levels=np.linspace(10, 150, 8), colors="white", linewidths=0.45,
               alpha=0.55)

    x_star, y_star = op["fe"] + drive, op["fi"]
    ax.plot([0, CUT_MAX], [y_star, y_star], color="red", ls=":", lw=1.3, alpha=0.9)
    ax.plot(x_star, y_star, ms=8, mew=0.7, **STAR)

    ax.set_xlim(0, RATE_MAX)
    ax.set_ylim(0, RATE_MAX)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, RATE_MAX + 1, 10))
    ax.set_yticks(np.arange(0, RATE_MAX + 1, 10))
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    if show_xlabel:
        ax.set_xlabel(r"$\nu_e + \nu_{\mathrm{drive}}$ (Hz)", labelpad=1.5)
    if show_ylabel:
        ax.set_ylabel(r"$\nu_i$ (Hz)", labelpad=1.5)
    ax.text(0.93, 0.07, rf"$F_{pop}$", transform=ax.transAxes, ha="right", va="bottom",
            color="white", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.45, ec="none"))

    _draw_cut_inset(ax, pop, op, drive, x_star)
    return mesh


def _draw_cut_inset(ax, pop, op, drive, x_star):
    """1-D cut of F at nu_i = nu_i*, coloured with the heatmap's colour scale."""
    ins = ax.inset_axes([0.5, 0.56, 0.43, 0.36])
    x = np.linspace(0.0, CUT_MAX, 300)
    y = rate_surface(pop, x, np.full_like(x, op["fi"]), op)
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    seg = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=CMAP, norm=NORM,
                         linewidth=1.3)
    seg.set_array(np.clip(0.5 * (y[:-1] + y[1:]), NORM.vmin, NORM.vmax))
    ins.add_collection(seg)
    y_star = rate_surface(pop, np.array([x_star]), np.array([op["fi"]]), op)[0]
    ins.plot(x_star, y_star, ms=5.5, mew=0.5, **STAR)

    ins.set_xlim(0, CUT_MAX)
    ins.set_ylim(-0.05 * CUT_YMAX, CUT_YMAX)
    ins.set_facecolor("white")
    ins.tick_params(labelsize=5, length=1.5, width=0.4, pad=1)
    ins.set_xticks([0, 5, 10])
    ins.set_yticks([0, 20, 40])
    ins.set_xlabel(r"$\nu_e + \nu_{\mathrm{drive}}$ (Hz)", fontsize=5, labelpad=0.5)
    ins.set_ylabel(rf"$F_{pop}$ (Hz)", fontsize=5, labelpad=1.0)
    for s in ins.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.4)

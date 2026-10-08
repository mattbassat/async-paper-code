"""Fig. 2b: population rates of the spiking network (SNN) and of the mean field under the same
drive.

One network (nu_drive = 4 Hz, b = 60 pA, ΔEL = 0) is spontaneous, then receives two cycles of a
2 Hz sinusoidal modulation of the external drive, then is spontaneous again.  The stored SNN
rates are drawn as thin traces; the first-order mean field, started from the SNN's initial
rates and given exactly the same drive, is integrated here and drawn as thick traces.
"""
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from asyncresp import mean_field as MF
from asyncresp.io import load_pickle
from asyncresp.paths import MEAN_FIELD
from asyncresp.style import E_COL, I_COL, INK2, DRIVE_COL

# protocol of the stored simulation (rates in Hz, potentials in mV, times in ms)
PROTOCOL = dict(b_val=60, rate=4, EL_inh=-70, EL_exc=-70, A=1.5, period=500.0,
                oscillation_onset=1500, oscillation_end=2500, tot_time=4000)
ON, OFF = PROTOCOL["oscillation_onset"], PROTOCOL["oscillation_end"]
PAD = ON - 500.0                                # spontaneous stretch shown either side
WINDOW = (ON - PAD, OFF + PAD)
SMOOTH_MS = 4.0                                 # boxcar smoothing of the SNN rates
Y_FLOOR = -0.42                                 # room under the traces for the drive
DRIVE_BAND = (-0.34, -0.10)
HALO = [pe.withStroke(linewidth=2.0, foreground="white")]


def _smooth(y, ms, bin_ms):
    n = max(1, int(round(ms / bin_ms)))
    return np.convolve(y, np.ones(n) / n, mode="same")


def drive_of_t(t_s):
    """External Poisson rate seen by both models (Hz), as a function of time in seconds."""
    t = t_s * 1e3
    if ON <= t <= OFF:
        return PROTOCOL["rate"] + PROTOCOL["A"] * np.sin(2 * np.pi * (t - ON) / PROTOCOL["period"])
    return PROTOCOL["rate"]


def draw_mf_vs_snn(ax, window=WINDOW):
    o = load_pickle(MEAN_FIELD / "mf_burst_sim.pkl")
    t, bin_ms = o["t_ms"], o["bin_ms"]
    m = (t >= window[0]) & (t <= window[1])
    init = t < 10.0
    tr = MF.drive_trace(drive_of_t, ELe=PROTOCOL["EL_exc"], ELi=PROTOCOL["EL_inh"],
                        b=PROTOCOL["b_val"], fe0=float(o["fr_exc"][init].mean()),
                        fi0=float(o["fr_inh"][init].mean()),
                        tmax=PROTOCOL["tot_time"] / 1e3)
    tm = (tr[:, 0] * 1e3 >= window[0]) & (tr[:, 0] * 1e3 <= window[1])

    top = 1.06 * max(_smooth(o["fr_inh"], SMOOTH_MS, bin_ms)[m].max(), tr[tm, 2].max())
    for key, col, mf_col in (("fr_inh", I_COL, 2), ("fr_exc", E_COL, 1)):
        ax.plot(t[m], _smooth(o[key], SMOOTH_MS, bin_ms)[m], color=col, linewidth=0.5,
                alpha=0.45, zorder=2)
        line, = ax.plot(tr[tm, 0] * 1e3, tr[tm, mf_col], color=col, linewidth=1.7, zorder=4)
        line.set_path_effects(HALO)
    for lab, col, mf_col in ((r"$\nu_i$", I_COL, 2), (r"$\nu_e$", E_COL, 1)):
        ax.text(window[1] - 0.006 * (window[1] - window[0]), tr[tm, mf_col][-1] + 0.085 * top,
                lab, fontsize=9, color=col, ha="right", va="bottom", zorder=6,
                path_effects=HALO)

    # the drive, rescaled into its own band under the traces
    d = tr[tm, 4]
    y0, y1 = DRIVE_BAND[0] * top, DRIVE_BAND[1] * top
    lo, hi = PROTOCOL["rate"] - PROTOCOL["A"], PROTOCOL["rate"] + PROTOCOL["A"]
    ax.plot(tr[tm, 0] * 1e3, y0 + (d - lo) / (hi - lo) * (y1 - y0), color=DRIVE_COL,
            linewidth=0.7, linestyle=(0, (1.4, 1.2)), zorder=3)
    ax.text(window[1] - 0.006 * (window[1] - window[0]), y1 + 0.035 * top, r"$\nu_{drive}$",
            fontsize=8, color=DRIVE_COL, ha="right", va="bottom", zorder=6,
            path_effects=HALO)

    # axis arrows in place of numbered axes
    ax.set_xlim(window[0] - 0.035 * (window[1] - window[0]), window[1])
    ax.set_ylim(Y_FLOOR * top, top)
    ax.axis("off")
    x0, yb = window[0] - 0.012 * (window[1] - window[0]), -0.015 * top
    ax.annotate("", xy=(x0, 0.42 * top), xytext=(x0, yb), annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7, shrinkA=0, shrinkB=0))
    ax.annotate("", xy=(x0 + 0.22 * (window[1] - window[0]), yb), xytext=(x0, yb),
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7, shrinkA=0, shrinkB=0))
    ax.text(x0 - 0.012 * (window[1] - window[0]), 0.21 * top, "Population rate", fontsize=6.4,
            color=INK2, rotation=90, ha="right", va="center")
    ax.text(x0 + 0.11 * (window[1] - window[0]), yb - 0.025 * top, "Time", fontsize=6.4,
            color=INK2, ha="center", va="top")
    ax.legend([Line2D([], [], color=INK2, lw=0.8, alpha=0.5),
               Line2D([], [], color=INK2, lw=1.7)],
              ["Spiking network", "Mean field"], loc="upper right", frameon=False, fontsize=6.6,
              handlelength=1.3, labelspacing=0.25, borderaxespad=0.3)

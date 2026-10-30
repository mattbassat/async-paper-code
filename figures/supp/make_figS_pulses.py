"""Supplementary figures: responses of the AdEx network to single pulses of afferent input."""
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
from asyncresp.io import load_pickle
from asyncresp.paths import OUTPUT, SUPP
from asyncresp.style import DRIVE_COL, E_COL, I_COL, INK

PULSES = SUPP / "pulse_inputs"
AMPLITUDE = SUPP / "pulse_amplitude" / "sweep_summary_combined.pkl"
PERIODS = [2000, 200, 20]                  # ms, figure columns
YLIM = (0, 35)                             # Hz
A_VALUES = [1, 5, 10, 25, 50]              # Hz
LW, LW_DRIVE, ALPHA = 1.0, 0.9, 0.2


def _condition(data, delta_EL, period):
    return next(c for c in data["conditions"]
                if c["delta_EL"] == delta_EL and c["period"] == period)


def trace_panel(ax, c):
    """Afferent rate (dashed) and mean +- SD over seeds of the E and I rates."""
    t = c["t_ms"]
    ax.plot(t, c["aff"].mean(axis=0), color=DRIVE_COL, lw=LW_DRIVE, ls=(0, (3, 2)))
    for key, col in (("exc", E_COL), ("inh", I_COL)):
        m, s = c[key].mean(axis=0), c[key].std(axis=0)
        ax.plot(t, m, color=col, lw=LW)
        ax.fill_between(t, m - s, m + s, color=col, alpha=ALPHA, lw=0)
    ax.set_xlim(*c["window"])
    ax.set_ylim(*YLIM)


def duration_bar(ax, c, t_center):
    """Horizontal bar spanning the pulse, labelled with its duration."""
    ymin, ymax = ax.get_ylim()
    y = ymax - 0.25 * (ymax - ymin)
    tick = 0.03 * (ymax - ymin)
    t0, t1 = t_center - c["pulse_ms"] / 2, t_center + c["pulse_ms"] / 2
    ax.hlines(y, t0, t1, colors=INK, lw=1.4)
    ax.vlines([t0, t1], y - tick, y + tick, colors=INK, lw=1.0)
    ax.text((t0 + t1) / 2, y + 2 * tick, f"{int(c['pulse_ms'])} ms", ha="center", va="bottom",
            fontsize=7)


def ei_key(fig, y):
    fig.legend(handles=[Patch(facecolor=E_COL, label="Excitatory"),
                        Patch(facecolor=I_COL, label="Inhibitory")],
               loc="upper center", bbox_to_anchor=(0.5, y), ncol=2, frameon=False,
               fontsize=7, handlelength=1.4, columnspacing=1.6)


def save(fig, name):
    out = OUTPUT / "supp"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.pdf")


def make_half_cycle():
    data = load_pickle(PULSES / "half_cycle.pkl")
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(3, 3, left=0.09, right=0.97, top=0.93, bottom=0.075, wspace=0.28,
                          hspace=0.42)
    for r, (dEL, letter) in enumerate(zip((-15, -5, 5), ("a.", "b.", "c."))):
        for col, P in enumerate(PERIODS):
            ax = fig.add_subplot(gs[r, col])
            c = _condition(data, dEL, P)
            trace_panel(ax, c)
            if r == 0:
                duration_bar(ax, c, data["t_center"])
            if col == 0:
                ax.set_ylabel("Firing rate (Hz)")
                p = ax.get_position()
                fig.text(0.01, p.y1 + 0.012, letter, fontsize=9, fontweight="bold",
                         va="bottom", ha="left")
            if r == 2:
                ax.set_xlabel("Time (ms)")
    ei_key(fig, 0.995)
    save(fig, "figS_half_cycle")
    return fig


def make_two_cycle():
    data = load_pickle(PULSES / "two_cycle.pkl")
    fig = plt.figure(figsize=(7.2, 2.35))
    gs = fig.add_gridspec(1, 3, left=0.09, right=0.97, top=0.83, bottom=0.2, wspace=0.28)
    for col, P in enumerate(PERIODS):
        ax = fig.add_subplot(gs[0, col])
        trace_panel(ax, _condition(data, 5, P))
        ax.set_xlabel("Time (ms)")
        if col == 0:
            ax.set_ylabel("Firing rate (Hz)")
    ei_key(fig, 0.995)
    save(fig, "figS_two_cycle")
    return fig


def make_pulse_amplitude():
    summary = load_pickle(AMPLITUDE)
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 6, left=0.08, right=0.985, top=0.94, bottom=0.1, wspace=1.6,
                          hspace=0.55)
    slots = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6], gs[1, 1:3], gs[1, 3:5]]
    for A, slot in zip(A_VALUES, slots):
        d = summary[A]
        ax = fig.add_subplot(slot)
        x, m, s = d["delta_ELs"], d["mean_exc"], d["std_exc"]
        ax.plot(x, m, color=E_COL, lw=1.2)
        ax.fill_between(x, m - s, m + s, color=E_COL, alpha=ALPHA, lw=0)
        ax.set_xlim(-15, 15)
        ax.set_xticks(np.arange(-15, 16, 5))
        ax.set_xlabel(r"$\Delta_{E_L}$ (mV)")
        ax.set_ylabel("Excitatory Amplitude (Hz)")
        ax.set_title(rf"$A={A}$ Hz")
    save(fig, "figS_pulse_amplitude")
    return fig


def make():
    SC.style()
    return make_half_cycle(), make_two_cycle(), make_pulse_amplitude()


if __name__ == "__main__":
    make()

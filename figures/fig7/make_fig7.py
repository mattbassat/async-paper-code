"""Fig. 7: single AdEx and HH cells under network-like synaptic bombardment."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supp"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import supp_common as SC
from asyncresp.io import load_pickle
from asyncresp.paths import SUPP, OUTPUT
from asyncresp.style import E_COL, I_COL, INK

DATA = SUPP / "single_cell_traces" / "traces.pkl"
YLIM = (-85, 55)
# (row, column) -> (stored key, panel letter, label, colour)
PANELS = {
    (0, 0): ("AdEx_RS", "a.", "AdEx RS", E_COL),
    (0, 1): ("AdEx_FS", "b.", "AdEx FS", I_COL),
    (1, 0): ("HH_RS", "c.", "HH RS", E_COL),
    (1, 1): ("HH_FS", "d.", "HH FS", I_COL),
}


def trace_panel(ax, rec, label, color):
    """Vm trace with dotted markers at the spike times."""
    for st in rec["spike_ms"]:
        ax.axvline(st, color=color, lw=0.8, ls=":", alpha=0.7, zorder=1)
    ax.plot(rec["t_ms"], rec["vm_mV"], color=color, lw=0.6, zorder=2)
    ax.set_xlim(0, 1000)
    ax.set_ylim(*YLIM)
    ax.set_yticks([-80, -40, 0, 40])


def make():
    SC.style()
    data = load_pickle(DATA)
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.10, wspace=0.14,
                        hspace=0.22)
    for (r, c), (key, letter, label, color) in PANELS.items():
        ax = axes[r, c]
        trace_panel(ax, data[key], label, color)
        SC.letter(ax, letter, dx=-0.13 if c == 0 else -0.06, dy=1.01)
        if r == 1:
            ax.set_xlabel("Time (ms)")
        if c == 0:
            ax.set_ylabel(r"$V_m$ (mV)")
    out = OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig7.png", dpi=300)
    fig.savefig(out / "fig7.pdf")
    return fig


if __name__ == "__main__":
    make()

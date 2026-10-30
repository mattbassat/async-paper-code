"""Fig. 1i: mean pairwise spike-count correlation (CC) against Delta_EL for the networks of the
Delta_EL sweep (nu_drive = 2 Hz, b = 0 pA)."""
import numpy as np

from asyncresp.io import load_pickle
from asyncresp.paths import SYNCHRONY
from asyncresp.style import INK2
from common import rocket_colors

ASYNC_REF = 0.1


def load_cc():
    """(Delta_EL, CC mean, CC SD) of the sweep, sorted by Delta_EL."""
    res = load_pickle(SYNCHRONY / "regimes_cc_r2b0.pkl")
    keys = sorted(res)
    d = np.array([float(k) for k in keys])
    cc = np.array([float(res[k]["CC"]) for k in keys])
    sd = np.array([float(res[k]["CC_sd"]) for k in keys])
    return d, cc, sd


def draw_panel_i(ax):
    d, cc, sd = load_cc()
    colors = rocket_colors(d)
    for x, y, e in zip(d, cc, sd):          # caps wider than the marker, so short bars stay visible
        ax.errorbar(x, y, yerr=e, fmt="o", markersize=4.2, color=colors[x],
                    markeredgecolor="black", markeredgewidth=0.45, ecolor=colors[x],
                    elinewidth=0.9, capsize=3.6, capthick=1.0, zorder=3)

    ax.set_yscale("log")
    lo = max(1e-5, (cc - sd)[cc - sd > 0].min() * 0.5)
    ax.set_ylim(lo, ASYNC_REF * 3.2)
    ax.axhline(ASYNC_REF, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
    ax.text(0.5, ASYNC_REF * 0.78, f"Asynchronous (CC < {ASYNC_REF:g})", fontsize=6.0,
            color=INK2, ha="center", va="top", transform=ax.get_yaxis_transform())
    ax.set_xlim(d.min() - 2.5, d.max() + 2.5)
    ax.set_xticks(d)
    ax.set_xticklabels(["0" if v == 0 else f"{v:+g}".replace("-", "−") for v in d])
    ax.set_xlabel(r"$\Delta_{E_L}$ (mV)")
    ax.set_ylabel("Average pairwise synchrony (CC)")

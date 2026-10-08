"""Fig. 3a: E and I responses to a slow (0.5 Hz) input as a function of Delta_EL.

One column per parameter set: SNN responses (mean +- SD over seeds) over the analytical
mean-field prediction of Eq. 4, on the OD / UD shading set by the mean-field boundary
Delta_EL^crit.  Also draws the column titles, the E/I and SNN/analytical keys above the panel,
and the symbols marking the example networks of Fig. 3c.
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from asyncresp.style import E_COL, I_COL, INK, INK2

from sweep_data import SETS, SLOW_F

OD_COL, UD_COL = "#e3e3e3", "#dbeefb"      # shading of the OD / UD regimes
UD_INK = "#2a6fb0"                         # "UD" label
XLIM = (-17.5, 17.5)                       # Delta_EL range (mV)
MARK_PAD = 0.14                            # room under the data for the example symbols
LEG = dict(frameon=True, fancybox=True, edgecolor="0.8", framealpha=1.0, fontsize=6.3,
           handlelength=1.8, columnspacing=1.6, handletextpad=0.5, borderpad=0.45)


def draw_slow_input(ax, agg, e, show_ylabel, ymax=8.2):
    """SNN (points) and Eq. 4 (lines) slow-input responses of one parameter set."""
    dcrit = e.delta_crit.iloc[0]
    lo, hi = XLIM
    ax.axvspan(lo, dcrit, color=OD_COL, linewidth=0, zorder=0)
    ax.axvspan(dcrit, hi, color=UD_COL, linewidth=0, zorder=0)
    ax.axvline(dcrit, color=INK, linewidth=0.6, linestyle=(0, (2, 2)))
    ax.text(dcrit - 0.5, 0.83 * ymax, r"$\Delta_{E_L}^{\mathrm{crit.}}$", ha="right",
            va="center", fontsize=6.3, color=INK)
    ax.text(0.5 * (lo + dcrit), 0.955 * ymax, "OD", ha="center", va="top", fontsize=6.5,
            weight="bold", color=INK2)
    ax.text(0.5 * (dcrit + hi), 0.955 * ymax, "UD", ha="center", va="top", fontsize=6.5,
            weight="bold", color=UD_INK)

    ax.plot(e.delta_EL, e.Hi0, color=I_COL, linewidth=1.1)
    ax.plot(e.delta_EL, e.He0, color=E_COL, linewidth=1.1)
    s = agg[np.isclose(agg.f, SLOW_F)].sort_values("delta_EL")
    for col, y, ysd in [(I_COL, s.amp_i, s.amp_i_sd), (E_COL, s.amp_e, s.amp_e_sd)]:
        ax.errorbar(s.delta_EL, y, yerr=ysd, fmt="o", ms=2.6, color=col, mec="white", mew=0.4,
                    elinewidth=0.6, capsize=2.4, capthick=0.6)
    ax.set_xlim(lo, hi); ax.set_ylim(-0.3, ymax); ax.set_xticks([-15, -5, 5, 15])
    ax.set_xlabel(r"$\Delta_{E_L}$ (mV)")
    if not show_ylabel:
        ax.set_yticklabels([])


def column_title(fig, ax, key):
    """Parameter-set title above a column, preceded by the set's marker used in Fig. 3b."""
    ax.set_title(SETS[key]["label"], fontsize=6.8, pad=4)
    fig.canvas.draw()
    bb = ax.title.get_window_extent().transformed(fig.transFigure.inverted())
    fig.add_artist(Line2D([bb.x0 - 0.011], [0.5 * (bb.y0 + bb.y1)], marker=SETS[key]["marker"],
                          ms=4.4, color=SETS[key]["colour"], mec="white", mew=0.5,
                          transform=fig.transFigure))


def ei_legends(fig, ax_left):
    """E / I and analytical / SNN keys, side by side above the slow-input columns."""
    p = ax_left.get_position()
    y = p.y1 + 0.075
    leg_ei = fig.legend(handles=[Patch(facecolor=E_COL, edgecolor=INK2, linewidth=0.5, label="E"),
                                 Patch(facecolor=I_COL, edgecolor=INK2, linewidth=0.5, label="I")],
                        ncol=2, loc="lower left", bbox_to_anchor=(p.x0, y),
                        bbox_transform=fig.transFigure,
                        **{**LEG, "handlelength": 2.2, "handleheight": 1.0})
    fig.canvas.draw()
    bb = leg_ei.get_window_extent().transformed(fig.transFigure.inverted())
    h_snn = ax_left.errorbar([np.nan], [np.nan], yerr=[1.0], fmt="o", ms=2.8, color=INK,
                             elinewidth=0.6, capsize=2.4, capthick=0.6)
    fig.legend(handles=[Line2D([], [], color=INK, linewidth=1.1), h_snn],
               labels=["Analytical prediction", "SNN simulations"],
               ncol=2, loc="lower left", bbox_to_anchor=(bb.x1 + 0.012, y),
               bbox_transform=fig.transFigure, **LEG)


def mark_examples(axes, ax, symbols):
    """Mark the example networks of Fig. 3c by their symbols, in a strip under the data of `ax`
    at their Delta_EL ({delta_EL: symbol}).

    The bottom of every column is lowered by the same amount, so that the three columns keep a
    common rate scale."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    for a in axes:
        a.set_ylim(lo - MARK_PAD * span, hi)
    for d, sym in symbols.items():
        ax.text(d, lo - 0.065 * span, sym, fontsize=10, color=INK, ha="center",
                va="center", zorder=6)

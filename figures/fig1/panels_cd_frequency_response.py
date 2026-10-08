"""Fig. 1c-d: frequency response of the AdEx network.  Mean (+/- SD over seeds) amplitude of the
evoked E rate modulation against the drive frequency f, for the Delta_EL sweep (c) and for the
leak-conductance, adaptation and E->E / E->I synaptic-strength sweeps (d).  Colours run from
dark (low-pass, OD side) to light (resonant, UD side) in every sweep."""
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator

from common import plain_log_ticks, rocket_colors

# Fig. 1d, in reading order: (sweep, reverse palette, title, parameter range from dark to light)
PANEL_D_SWEEPS = [
    ("delta_gL", False, r"Leak conductance, $\Delta g_L$", r"$-5 \rightarrow +5$ nS"),
    ("b", True, r"Adaptation, $b$", r"$150 \rightarrow 0$ pA"),
    ("Qee", False, r"E$\rightarrow$E synapses, $Q_{ee}$", r"$1.25 \rightarrow 1.6$ nS"),
    ("Qei", True, r"E$\rightarrow$I synapses, $Q_{ei}$", r"$2.0 \rightarrow 1.4$ nS"),
]


def style_frequency_axis(ax, xrange=(0.5, 100), ymax=3.0, xlabel=True, ylabel=True):
    """Logarithmic drive-frequency axis and linear response axis starting at zero."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=10))
    plain_log_ticks(ax.xaxis)
    ax.grid(False)
    ax.set_xlim(xrange)
    ax.set_ylim(0, ymax)
    if xlabel:
        ax.set_xlabel(r"$f$ (Hz)")
    else:
        ax.set_xticklabels([])
    if ylabel:
        ax.set_ylabel("E response amplitude (Hz)")
    else:
        ax.set_yticklabels([])


def plot_exc_responses(ax, data, param_name, colors, linewidth=1.3):
    """One curve per parameter value: mean E response amplitude with a +/- SD band."""
    for value in np.sort(data[param_name].unique()):
        sub = data[data[param_name] == value].sort_values("frequency")
        if sub.empty:
            continue
        freq, mean, std = sub["frequency"], sub["amp_exc_mean"], sub["amp_exc_std"]
        ax.fill_between(freq, mean - std, mean + std, color=colors[value], alpha=0.25, linewidth=0)
        ax.plot(freq, mean, color=colors[value], linestyle="-", linewidth=linewidth)


def draw_panel_c(ax, data):
    """Fig. 1c: the Delta_EL sweep, with its colour legend above the axes."""
    colors = rocket_colors(data["delta_EL"].unique())
    plot_exc_responses(ax, data, "delta_EL", colors)
    style_frequency_axis(ax, ymax=2.5)
    fmt = lambda v: "0" if v == 0 else f"{v:+g}".replace("-", "−")
    handles = [Patch(facecolor=colors[v], edgecolor="none", label=fmt(v))
               for v in np.sort(data["delta_EL"].unique())]
    ax.legend(handles=handles, title=r"$\Delta_{E_L}$ (mV)", ncol=len(handles),
              loc="lower left", bbox_to_anchor=(0.0, 1.03), frameon=True, fancybox=False,
              edgecolor="0.8", framealpha=1.0, handlelength=1.0, handleheight=0.9,
              columnspacing=0.9, handletextpad=0.4, title_fontsize=7, fontsize=6.5,
              borderaxespad=0.0, borderpad=0.5)


def draw_panel_d_axis(ax, data, param_name, reverse, title, value_range, xlabel, ylabel, ymax=3.0):
    """One of the four small multiples of Fig. 1d (the y-label is shared and set by the caller)."""
    colors = rocket_colors(data[param_name].unique(), reverse=reverse)
    plot_exc_responses(ax, data, param_name, colors, linewidth=1.0)
    style_frequency_axis(ax, ymax=ymax, xlabel=xlabel, ylabel=ylabel)
    ax.set_ylabel("")
    ax.set_title(title, loc="left", fontsize=7, pad=3)
    ax.text(0.97, 0.95, value_range, transform=ax.transAxes, ha="right", va="top", fontsize=6.2,
            color="0.3")


def draw_excitability_key(cax):
    """Shared colour key of Fig. 1d: dark = low-pass (OD) side, light = resonant (UD) side."""
    gradient = np.linspace(0, 1, 256)[None, :]
    cax.imshow(gradient, aspect="auto", cmap=sns.color_palette("rocket", as_cmap=True))
    cax.set_xticks([])
    cax.set_yticks([])
    for s in cax.spines.values():
        s.set_visible(False)
    cax.text(-0.02, 0.5, "low-pass (OD)", transform=cax.transAxes, ha="right", va="center", fontsize=6.5)
    cax.text(1.02, 0.5, "resonant (UD)", transform=cax.transAxes, ha="left", va="center", fontsize=6.5)
    cax.text(0.5, 1.6, "Excitability", transform=cax.transAxes, ha="center", va="bottom", fontsize=6.5)

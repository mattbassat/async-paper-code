"""Panels of the LFP-ringing supplementary figures."""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from asyncresp.analysis import rocket_colors

import ringdown as RD

STYLE = {
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "font.family": "sans-serif",
}

# Delta_EL colours: discrete rocket palette over the full -15 ... +15 mV grid, so that a
# network has the same colour in every panel and in both figures.
DELTA_EL_COLORS = rocket_colors([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0])
# Example-network markers of panel b, and the line colour of panel b.
ROLE_COLOR = {"OD": "#8c8c8c", "boundary": "#7a5195", "UD": "#1f6db4"}
LINE_COLOR = "#1f6db4"
LABEL_FONTSIZE = 10


def style():
    plt.rcParams.update(STYLE)


def rocket_color(delta_EL):
    return DELTA_EL_COLORS[delta_EL]


def panel_label(ax, letter):
    ax.text(-0.02, 1.06, f"{letter}.", transform=ax.transAxes, fontsize=LABEL_FONTSIZE,
            fontweight="bold", va="bottom", ha="right")


def panel_a_examples(fig, gs_slice, examples, role_delta_el, k_sigma):
    """`examples` maps role -> single driven run (data.example_run)."""
    pos = gs_slice.get_position(fig)
    inner = fig.add_gridspec(1, 3, left=pos.x0, right=pos.x1, bottom=pos.y0, top=pos.y1,
                             wspace=0.15)
    axes = [fig.add_subplot(inner[0, j]) for j in range(3)]
    for j, role in enumerate(["OD", "boundary", "UD"]):
        o = examples[role]
        t, x, T = o["t_state"], o["lfp_inh"], o["period_ms"]
        t_off = o["oscillation_offset"]
        fit = RD.analyze_ringdown(t, x, T, o["oscillation_onset"], t_off, o["total_ms"], k_sigma)
        t_curve, y_curve = RD.ringdown_curve(t, fit, t_off)

        zoom_lo, zoom_hi = t_off - 5 * T, min(o["total_ms"], t_off + 9 * T)
        m = (t >= zoom_lo) & (t <= zoom_hi)
        mc = (t_curve >= zoom_lo) & (t_curve <= zoom_hi)

        ax = axes[j]
        color = rocket_color(role_delta_el[role])
        ax.plot((t[m] - t_off) / T, x[m] / 1e6, color=color, alpha=0.55, linewidth=1.1)
        pre = mc & (t_curve <= t_off)
        post = mc & (t_curve >= t_off)
        ax.plot((t_curve[pre] - t_off) / T, y_curve[pre] / 1e6, color=color, linewidth=1.6)
        ax.plot((t_curve[post] - t_off) / T, y_curve[post] / 1e6, color=color,
                linewidth=1.6, linestyle=(0, (3, 1.5)))
        ax.axvline(0, color="0.3", linewidth=0.7, linestyle=":")

        ax.set_title(role, fontsize=9, color=color)
        ax.set_xlabel("cycles from offset")
        if j == 0:
            ax.set_ylabel("LFP (a.u., $\\times 10^6$)")
        else:
            ax.set_yticklabels([])
        ax.set_xlim((zoom_lo - t_off) / T, (zoom_hi - t_off) / T)
    ylims = [ax.get_ylim() for ax in axes]
    ylo, yhi = min(y[0] for y in ylims), max(y[1] for y in ylims)
    for ax in axes:
        ax.set_ylim(ylo, yhi)
    panel_label(axes[0], "a")
    return axes


def panel_b_cycles(ax, delta_EL, mean, sem, role_delta_el, xlim):
    ax.plot(delta_EL, mean, color=LINE_COLOR, linewidth=1.6, marker="o", markersize=3.5, zorder=3)
    ax.fill_between(delta_EL, mean - sem, mean + sem, color=LINE_COLOR, alpha=0.15, linewidth=0)
    for role, d in role_delta_el.items():
        ax.axvline(d, color=ROLE_COLOR[role], linewidth=0.6, linestyle=":", zorder=1, alpha=0.7)
    ax.set_xlabel(r"$\Delta_{E_L}$ (mV)")
    ax.set_ylabel("significant cycles\n(mean $\\pm$ sem)")
    ax.set_xlim(*xlim)
    panel_label(ax, "b")
    return ax


def panel_c_spectra(ax, spectra, resonance_lines, fmax=500):
    """`spectra`: {Delta_EL: (freqs, psd)}; `resonance_lines`: {Delta_EL: frequency (Hz)}."""
    for d, (f_, p_) in spectra.items():
        m = (f_ >= 2) & (f_ <= fmax)
        ax.plot(f_[m], p_[m], color=rocket_color(d), linewidth=1.3)
    for d, f_pk in resonance_lines.items():
        ax.axvline(f_pk, color=rocket_color(d), linewidth=1.1, linestyle=(0, (4, 2)), zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("spontaneous LFP PSD (a.u.)")
    ax.set_xlim(2, fmax)
    panel_label(ax, "c")
    return ax


def delta_el_key(fig, values):
    """Delta_EL colour key, centred above the panels."""
    handles = [mpatches.Patch(facecolor=rocket_color(v), edgecolor="black") for v in values]
    labels = [f"${v:.0f}$" for v in values]
    fig.legend(handles=handles, labels=labels, title=r"$\Delta_{E_L}$ (mV)", loc="upper center",
               bbox_to_anchor=(0.53, 0.995), ncol=len(values), fontsize=8, title_fontsize=9.5,
               frameon=True, handlelength=3.6, handleheight=1.6, handletextpad=0.5,
               columnspacing=1.5)

"""Build Fig. 1 from the stored results in data/ and write output/fig1.png (300 dpi) and
output/fig1.pdf."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import LogLocator

from asyncresp.paths import output_path
from common import rocket_colors
from panel_a_network import draw_panel_a
from panel_b_neuromodulation import draw_panel_b
from panel_g_od_ud import draw_panel_g
from panel_h_ringdown import draw_panel_h
from panel_i_synchrony import draw_panel_i
from panels_cd_frequency_response import (PANEL_D_SWEEPS, draw_excitability_key,
                                          draw_panel_c, draw_panel_d_axis)
from panels_ef_spectra import PANEL_F_SWEEPS, draw_panel_e, draw_panel_f_axis
from sweeps import load_sweep, sampled_resonance_peaks

# --- geometry, in inches ---
W_A, W_B, W_C = 5.88, 2.73, 3.01                 # panel widths
GAP_W, GAP_H = 0.52, 0.55
W_LEFT = W_B + GAP_W + W_C                       # left column: c + d (and e + f) side by side
H_A, H_TOP, H_BOT = 4.73, 2.36, 1.81
MARGIN = dict(left=0.55, right=0.10, top=0.42, bottom=0.40)
X = [MARGIN["left"], MARGIN["left"] + W_LEFT + GAP_W,
     MARGIN["left"] + W_LEFT + GAP_W + W_B + GAP_W]
GAP_H1 = 0.85                                    # row 1 -> row 2 (d's colour key sits in it)
ROW = [MARGIN["top"], MARGIN["top"] + H_A + GAP_H1,
       MARGIN["top"] + H_A + GAP_H1 + H_TOP + GAP_H]
FIGW = X[2] + W_C + MARGIN["right"]
FIGH = ROW[2] + H_BOT + MARGIN["bottom"]
X_D = X[0] + W_B + GAP_W                         # d and f sit next to c and e
LX = [x - 0.48 for x in X]                       # panel letters
B_DROP = 0.28                                    # panel b's content is shifted down by this much
GH_GAP = 0.22                                    # gap between g's two sub-panels
I_LABEL_GUTTER = 0.42                            # room for i's y-axis, left of its axes box

RC = {
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
}


def cell(fig, x, y_top, w, h):
    """A one-slot gridspec at (x, y_top) inches from the top left, w x h inches."""
    return fig.add_gridspec(1, 1, left=x / FIGW, right=(x + w) / FIGW,
                            top=1 - y_top / FIGH, bottom=1 - (y_top + h) / FIGH)[0, 0]


def letter(fig, s, x, y_top):
    fig.text(x / FIGW, 1 - y_top / FIGH, s, fontsize=10, fontweight="bold", va="bottom", ha="left")


def style_ticks(fig):
    """Short outward ticks on every axis, and minor ticks at 2-9 x each decade of the log axes.
    Axes whose time base is given by a scale bar keep no x ticks."""
    for ax in fig.axes:
        ax.tick_params(which="major", length=2.8, width=0.6, direction="out")
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if scale == "log":
                axis.set_minor_locator(LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1),
                                                  numticks=100))
        ax.tick_params(which="minor", length=1.5, width=0.5, direction="out")


def make_figure():
    sweeps = {name: load_sweep(name) for name in ("delta_EL", "delta_gL", "b", "Qee", "Qei")}
    data_dEL = sweeps["delta_EL"][0]

    sns.set_style("white")
    mpl.rcParams.update(RC)
    fig = plt.figure(figsize=(FIGW, FIGH))

    # --- row 1 ---
    draw_panel_a(fig, cell(fig, X[0], ROW[0], W_A, H_A))
    letter(fig, "a.", LX[0], ROW[0])

    draw_panel_b(fig, cell(fig, X[1] + 0.20, ROW[0] + 0.12 + B_DROP, W_LEFT - 0.20, H_A - 0.12))
    letter(fig, "b.", LX[1], ROW[0])

    # --- row 2 ---
    ax_c = fig.add_subplot(cell(fig, X[0], ROW[1], W_B, H_TOP))
    draw_panel_c(ax_c, data_dEL)
    letter(fig, "c.", LX[0], ROW[1])

    blk_d = cell(fig, X_D, ROW[1], W_C, H_TOP).subgridspec(2, 2, hspace=0.42, wspace=0.10)
    for k, (name, reverse, title, value_range) in enumerate(PANEL_D_SWEEPS):
        r, c = divmod(k, 2)
        data, param_name = sweeps[name]
        draw_panel_d_axis(fig.add_subplot(blk_d[r, c]), data, param_name, reverse, title,
                          value_range, xlabel=(r == 1), ylabel=(c == 0))
    fig.text((X_D - 0.34) / FIGW, 1 - (ROW[1] + H_TOP / 2) / FIGH, "E response amplitude (Hz)",
             rotation=90, ha="center", va="center", fontsize=7.5)
    key_w = 0.36 * W_C
    cax = fig.add_axes([(X_D + (W_C - key_w) / 2) / FIGW, 1 - (ROW[1] - 0.30) / FIGH,
                        key_w / FIGW, 0.008 * 6.3 / FIGH])
    draw_excitability_key(cax)
    letter(fig, "d.", X_D - 0.38, ROW[1])

    g_half = (W_LEFT - GH_GAP) / 2
    draw_panel_g(fig, cell(fig, X[1], ROW[1] + (H_TOP - H_BOT) / 2, W_LEFT, H_BOT),
                 wspace=GH_GAP / g_half)
    letter(fig, "g.", LX[1], ROW[1])

    # --- row 3 ---
    ax_e = fig.add_subplot(cell(fig, X[0], ROW[2], W_B, H_BOT))
    draw_panel_e(ax_e, data_dEL, sampled_resonance_peaks(data_dEL, "delta_EL"))
    letter(fig, "e.", LX[0], ROW[2])

    blk_f = cell(fig, X_D, ROW[2], W_C, H_BOT).subgridspec(2, 2, hspace=0.50, wspace=0.12)
    axes_f = []
    for k, (name, title, values, reverse, unit, fmt) in enumerate(PANEL_F_SWEEPS):
        r, c = divmod(k, 2)
        data, param_name = sweeps[name]
        ax = fig.add_subplot(blk_f[r, c], sharex=axes_f[0] if axes_f else None,
                             sharey=axes_f[0] if axes_f else None)
        draw_panel_f_axis(ax, name, title, values, reverse, unit, fmt,
                          sweep_values=data[param_name].unique(),
                          peaks=sampled_resonance_peaks(data, param_name),
                          show_ylabel=(c == 0), show_xlabel=(r == 1))
        axes_f.append(ax)
    fig.text((X_D - 0.38) / FIGW, 1 - (ROW[2] + H_BOT / 2) / FIGH, "Spontaneous LFP PSD (a.u.)",
             rotation=90, ha="center", va="center", fontsize=7.5)
    letter(fig, "f.", X_D - 0.38, ROW[2])

    # h is aligned with g's left sub-panel, leaving room for i's y-axis; i with g's right one
    w_h = g_half - I_LABEL_GUTTER
    x_i = X[1] + g_half + GH_GAP
    w_i = W_LEFT - g_half - GH_GAP
    blk_h = cell(fig, X[1], ROW[2], w_h, H_BOT).subgridspec(1, 2, wspace=0.10)
    ax_h1 = fig.add_subplot(blk_h[0, 0])
    ax_h2 = fig.add_subplot(blk_h[0, 1], sharey=ax_h1)
    draw_panel_h(ax_h1, ax_h2, rocket_colors(data_dEL["delta_EL"].unique()))
    txt_h = fig.text((X[1] + w_h / 2) / FIGW, 1 - (ROW[2] + H_BOT + 0.30) / FIGH,
                     "Cycles from stimulus offset", ha="center", va="top", fontsize=7.5)
    letter(fig, "h.", LX[1], ROW[2])

    ax_i = fig.add_subplot(cell(fig, x_i, ROW[2], w_i, H_BOT))
    draw_panel_i(ax_i)
    letter(fig, "i.", x_i - 0.48, ROW[2])

    fig.canvas.draw()                            # align h's x-label with e's
    bb = ax_e.xaxis.label.get_window_extent().transformed(fig.transFigure.inverted())
    txt_h.set_position((txt_h.get_position()[0], bb.y0))
    txt_h.set_va("bottom")

    style_ticks(fig)
    fig.savefig(output_path("fig1.png"), dpi=300)
    fig.savefig(output_path("fig1.pdf"))
    return fig


if __name__ == "__main__":
    make_figure()

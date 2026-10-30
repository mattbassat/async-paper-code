"""Fig. 2: the low-dimensional mean field explains the spiking network."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from asyncresp.paths import output_path
from fig2_common import style, C_OD, C_UD
from panel_schematic import draw_schematic
from panel_mf_vs_snn import draw_mf_vs_snn
from panel_phase_plane import draw_phase, draw_relaxation
from panels_stability import draw_eigen, draw_heatmap, draw_resonance_vs_balance

# --- geometry, in inches ---
W_L, W_R = 4.60, 6.26                                      # left / right column widths
GAP_W = 0.62
MARGIN = dict(left=0.62, right=0.12, top=0.42, bottom=0.40)
X = [MARGIN["left"], MARGIN["left"] + W_L + GAP_W]
H = dict(a=1.70, b=1.75, c=2.90, d=2.85, e=2.20, f=2.05)
GAP_L, GAP_R = (0.85, 0.85), (0.55, 0.40)
COL = dict(a=0, b=0, c=0, d=1, e=1, f=1)

Y = {"a": MARGIN["top"], "d": MARGIN["top"]}
Y["b"] = Y["a"] + H["a"] + GAP_L[0]
Y["c"] = Y["b"] + H["b"] + GAP_L[1]
Y["e"] = Y["d"] + H["d"] + GAP_R[0]
Y["f"] = Y["e"] + H["e"] + GAP_R[1]

FIGW = X[1] + W_R + MARGIN["right"]
FIGH = max(Y["c"] + H["c"], Y["f"] + H["f"]) + MARGIN["bottom"]


def box(key):
    """(x, y from top, width, height) of a panel, in inches."""
    return X[COL[key]], Y[key], (W_L if COL[key] == 0 else W_R), H[key]


def grid(fig, key, nrows=1, ncols=1, ins=None, **kw):
    """A gridspec inside a panel's box; `ins` gives the interior margins as fractions of it."""
    x, y, w, h = box(key)
    ins = ins or dict(left=0.0, right=1.0, top=1.0, bottom=0.0)
    return fig.add_gridspec(nrows, ncols,
                            left=(x + ins["left"] * w) / FIGW,
                            right=(x + ins["right"] * w) / FIGW,
                            top=1 - (y + (1 - ins["top"]) * h) / FIGH,
                            bottom=1 - (y + (1 - ins["bottom"]) * h) / FIGH, **kw)


def fig_rect(key, fx, fy, fw, fh):
    """A rectangle given in fractions of a panel's box, in figure coordinates."""
    x, y, w, h = box(key)
    return [(x + fx * w) / FIGW, 1 - (y + (1 - fy - fh) * h) / FIGH, fw * w / FIGW, fh * h / FIGH]


def letter(fig, s, key):
    x, y, _, _ = box(key)
    fig.text((x - 0.50) / FIGW, 1 - (y - 0.06) / FIGH, s, fontsize=10, fontweight="bold",
             va="bottom", ha="left")


def build():
    style()
    fig = plt.figure(figsize=(FIGW, FIGH))

    # a: the mean field
    ax = fig.add_subplot(grid(fig, "a", ins=dict(left=0.01, right=0.99, top=0.99, bottom=0.01))[0, 0])
    draw_schematic(ax)

    # b: mean field vs spiking network under the same drive
    gs = grid(fig, "b", ins=dict(left=0.035, right=0.995, top=0.98, bottom=0.03))
    draw_mf_vs_snn(fig.add_subplot(gs[0, 0]))

    # c: node vs focus
    gs = grid(fig, "c", 2, 2, ins=dict(left=0.135, right=0.99, top=0.92, bottom=0.115),
              height_ratios=[2.0, 1.0], hspace=0.62, wspace=0.28)
    for k, (dEL, lab, col) in enumerate([(-15.0, r"OD, $\Delta_{E_L}=-15$ mV", C_OD),
                                         (15.0, r"UD, $\Delta_{E_L}=+15$ mV", C_UD)]):
        draw_phase(fig.add_subplot(gs[0, k]), dEL, title=lab, colour=col, ylabel=(k == 0))
        draw_relaxation(fig.add_subplot(gs[1, k]), dEL, colour=col, ylabel=(k == 0))

    # d: eigenvalues and balance metric
    gs = grid(fig, "d", 2, 3, ins=dict(left=0.125, right=0.99, top=0.90, bottom=0.135),
              height_ratios=[1.45, 1.0], hspace=0.16, wspace=0.10)
    draw_eigen([fig.add_subplot(gs[0, j]) for j in range(3)],
               [fig.add_subplot(gs[1, j]) for j in range(3)])

    # e: SNN response maps with the mean-field eigenfrequency
    gs = grid(fig, "e", 1, 3, ins=dict(left=0.085, right=0.99, top=0.765, bottom=0.185),
              wspace=0.10)
    cax = fig.add_axes(fig_rect("e", 0.615, 0.935, 0.375, 0.032))
    handles = draw_heatmap([fig.add_subplot(gs[0, j]) for j in range(3)], cax=cax)
    fig.legend(handles=handles, loc="upper left", frameon=False, fontsize=5.9, ncol=2,
               handlelength=1.6, columnspacing=1.0, labelspacing=0.3,
               bbox_to_anchor=(fig_rect("e", 0.075, 1.005, 0, 0)[0],
                               fig_rect("e", 0, 1.005, 0, 0)[1]))

    # f: SNN resonance vs balance metric, centred under the maps of e
    gs = grid(fig, "f", ins=dict(left=0.215, right=0.865, top=0.96, bottom=0.215))
    draw_resonance_vs_balance(fig.add_subplot(gs[0, 0]))

    for k in "abcdef":
        letter(fig, k + ".", k)
    fig.savefig(output_path("fig2.png"), dpi=300)
    fig.savefig(output_path("fig2.pdf"))


if __name__ == "__main__":
    build()

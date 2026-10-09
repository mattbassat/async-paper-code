"""Build Fig. 4: mean-field transfer functions across the OD / UD transition.

    a  excitatory transfer function F_e for an overdamped network (Delta_EL = -15 mV), a network
       next to the low-pass / resonant boundary (-6.25 mV) and an underdamped network (+5 mV)
    b  the same for the inhibitory transfer function F_i

All networks have nu_drive = 2 Hz and b = 0 pA.  The panels are computed directly from the
mean-field transfer function (no stored data).  Writes output/fig4.png (300 dpi) and
output/fig4.pdf.  Usage: python figures/fig4/make_fig4.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib as mpl                                       # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402

from asyncresp.paths import output_path                        # noqa: E402
from asyncresp.style import INK                                # noqa: E402
from tf_panels import NORM, CMAP, operating_points, draw_tf_panel   # noqa: E402

STYLE = {
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
    "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

DRIVE, B_PA = 2.0, 0.0
CASES = [(-15.0, "OD"), (-6.25, "Boundary-adjacent"), (5.0, "UD")]


def column_title(delta_el, regime):
    sign = "+" if delta_el > 0 else "-"
    value = f"{abs(delta_el):g}"
    return rf"{regime}, $\Delta_{{E_L}} = {sign}{value}$ mV"


def make_fig4():
    mpl.rcParams.update(STYLE)
    ops = operating_points(DRIVE, [d for d, _ in CASES], B_PA)

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.885, top=0.92, bottom=0.08,
                          wspace=0.1, hspace=0.12)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]
    for r, pop in enumerate(("e", "i")):
        for c, op in enumerate(ops):
            mesh = draw_tf_panel(axes[r][c], pop, op, DRIVE,
                                 show_xlabel=(r == 1), show_ylabel=(c == 0))
            if c > 0:
                axes[r][c].set_yticklabels([])
            if r == 0:
                axes[r][c].set_xticklabels([])
    for c, (d, regime) in enumerate(CASES):
        axes[0][c].set_title(column_title(d, regime), color=INK, fontsize=7.5, pad=5)

    # shared colour bar
    top, bot = axes[0][2].get_position(), axes[1][2].get_position()
    cax = fig.add_axes([0.905, bot.y0 + 0.1 * (top.y1 - bot.y0), 0.014,
                        0.8 * (top.y1 - bot.y0)])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax)
    cb.set_label("Output firing rate (Hz)", labelpad=2)
    cb.outline.set_linewidth(0.5)
    cb.set_ticks([1, 10, 100])
    cb.set_ticklabels(["1", "10", "100"])

    for r, letter in enumerate(("a.", "b.")):
        axes[r][0].text(-0.27, 1.03, letter, transform=axes[r][0].transAxes, fontsize=9,
                        fontweight="bold", va="bottom", ha="left")
    fig.savefig(output_path("fig4.png"), dpi=300)
    fig.savefig(output_path("fig4.pdf"))
    return fig


if __name__ == "__main__":
    make_fig4()

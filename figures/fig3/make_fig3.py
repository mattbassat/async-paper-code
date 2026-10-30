"""Build Fig. 3: responses to slow input across the OD / UD transition."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

from asyncresp.paths import output_path
from sweep_data import SETS, load_set, slow_table
from panel_slow_input import draw_slow_input, column_title, ei_legends, mark_examples
from panel_balance import draw_balance
from panel_examples import draw_examples, example_symbols

STYLE = {
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
    "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
}


def fig_letters(items):
    """Bold panel letters, placed in axes coordinates: (ax, letter, x, y)."""
    for ax, letter, dx, dy in items:
        ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="bottom", ha="left")


def make_fig3():
    mpl.rcParams.update(STYLE)
    data = {k: load_set(k) for k in SETS}
    slow = {k: slow_table(*data[k]) for k in SETS}

    fig = plt.figure(figsize=(7.2, 4.75))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.7, 1.85], left=0.075, right=0.985,
                             top=0.845, bottom=0.055, hspace=0.50)
    top = outer[0, 0].subgridspec(1, 2, width_ratios=[3.2, 2.0], wspace=0.24)
    left = top[0, 0].subgridspec(1, 3, wspace=0.10)

    # a: one column per parameter set
    axes_a = []
    for j, k in enumerate(SETS):
        ax = fig.add_subplot(left[0, j])
        draw_slow_input(ax, *data[k], show_ylabel=(j == 0))
        axes_a.append(ax)
    for ax, k in zip(axes_a, SETS):
        column_title(fig, ax, k)
    axes_a[0].set_ylabel("Response to slow input\n($f = 0.5$ Hz)")
    ei_legends(fig, axes_a[0])
    mark_examples(axes_a, axes_a[list(SETS).index("r2b0")], example_symbols())

    # b: collapse against the balance metric
    ax_b = fig.add_subplot(top[0, 1])
    draw_balance(ax_b, data, slow)

    # c: example networks of the r2b0 set
    bottom = outer[1, 0].subgridspec(1, 3, wspace=0.10)
    axes_c = [fig.add_subplot(bottom[0, j]) for j in range(3)]
    draw_examples(axes_c)

    fig_letters([(axes_a[0], "a.", -0.36, 1.1), (ax_b, "b.", -0.2, 1.1),
                 (axes_c[0], "c.", -0.105, 1.16)])
    fig.savefig(output_path("fig3.png"), dpi=300)
    fig.savefig(output_path("fig3.pdf"))
    return fig


if __name__ == "__main__":
    make_fig3()

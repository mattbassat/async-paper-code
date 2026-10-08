"""Build Fig. 5: the transition between asynchronous states in a Hodgkin-Huxley network.

    a  E response amplitude against input frequency, one curve per Delta_EL
    b  the slow-input (0.5 Hz) E response against Delta_EL
    c  spontaneous LFP-proxy spectra of four networks, with the resonance peak of each
       network marked (the Hodgkin-Huxley counterpart of Fig. 1e)

    python figures/fig5/make_fig5.py      ->  output/fig5.png, output/fig5.pdf
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from asyncresp.paths import output_path
from hh_data import peak_frequencies, rocket_colors, sweep
from hh_panels import delta_colorbar, draw_responses, draw_slow, draw_spectra


def style():
    sns.set_style("white")
    mpl.rcParams.update({
        "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.8, "ytick.major.size": 2.8,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5, "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


# layout, in inches: three equal panels in a row (each the size of Fig. 1c)
W, H = 2.73, 2.36
GAP, MARGIN = 0.62, dict(left=0.66, right=0.12, top=0.52, bottom=0.44)
FIGW = MARGIN["left"] + 3 * W + 2 * GAP + MARGIN["right"]
FIGH = MARGIN["top"] + H + MARGIN["bottom"]


def make_figure():
    style()
    df = sweep()
    colors = rocket_colors(df.delta_EL.unique())
    peaks = peak_frequencies(df)

    fig = plt.figure(figsize=(FIGW, FIGH))
    axes = []
    for k in range(3):
        x = MARGIN["left"] + k * (W + GAP)
        gs = fig.add_gridspec(1, 1, left=x / FIGW, right=(x + W) / FIGW,
                              top=1 - MARGIN["top"] / FIGH,
                              bottom=1 - (MARGIN["top"] + H) / FIGH)
        axes.append(fig.add_subplot(gs[0, 0]))
        fig.text((x - 0.52) / FIGW, 1 - (MARGIN["top"] - 0.08) / FIGH, "abc"[k] + ".",
                 fontsize=10, fontweight="bold", va="bottom", ha="left")

    draw_responses(axes[0], df, colors)
    delta_colorbar(fig, axes[0], df.delta_EL.unique(), colors)
    draw_slow(axes[1], df, colors)
    draw_spectra(axes[2], peaks, colors)

    fig.savefig(output_path("fig5.png"), dpi=400)
    fig.savefig(output_path("fig5.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    make_figure()

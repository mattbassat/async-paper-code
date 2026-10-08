"""Build the simulated panels of Fig. 6 (acetylcholine case study).

Fig. 6 compares simulations with published experimental results (Howe et al., 2017,
J. Neurosci. 37:3215-3230): cue-evoked ACh release (their Fig. 3B), the LFP spectrogram
(their Fig. 6C) and the pre-cue vs cue LFP spectra (their Fig. 5C).  Those panels are
third-party material and are not reproduced here; this script draws only the simulated
counterparts (Fig. 6c), which were placed next to the experimental panels when the figure was
assembled:

    top     b(t) ramped from 75 to 10 pA during the cue, the z-scored multitaper spectrogram over
            63 runs (with colour bar), and its post-cue inset
    bottom  pre- vs post-cue Welch spectra

The panel sizes match the experimental panels they sit next to in the published figure.

    python figures/fig6/make_fig6.py      ->  output/fig6_simulated.png, output/fig6_simulated.pdf
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from ach_panels import SP_BOX, draw_sim_spectra, draw_sim_spectrogram
from asyncresp.paths import output_path

# layout, in inches
WC = 3.10                                   # column width
H_SPEC = WC / 0.838                         # height of the spectrogram block
H_SPECTRA = WC / 1.362                      # height of the spectra block
GAP_ROW = 0.30
MARGIN = dict(left=0.04, right=0.04, top=0.10, bottom=0.12)
FIGW = MARGIN["left"] + WC + MARGIN["right"]
FIGH = MARGIN["top"] + H_SPEC + GAP_ROW + H_SPECTRA + MARGIN["bottom"]
X0 = MARGIN["left"]
Y = [MARGIN["top"], MARGIN["top"] + H_SPEC + GAP_ROW]
FS_LABEL, FS_TICK = 8.5, 7.5


def cell(fig, x, y, w, h):
    """Axes at (x, y) from the top-left corner of the figure, w x h, all in inches."""
    return fig.add_axes([x / FIGW, 1 - (y + h) / FIGH, w / FIGW, h / FIGH])


def make_figure():
    mpl.rcParams.update({
        "font.size": FS_TICK, "axes.labelsize": FS_LABEL, "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK, "axes.linewidth": 0.8, "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(FIGW, FIGH))

    # b(t), the spectrogram with its colour bar, and the inset under it
    ax_b = cell(fig, X0 + 0.52, Y[0] + 0.06, WC - 1.02, 0.32)
    ax_spec = cell(fig, X0 + 0.52, Y[0] + 0.58, WC - 1.02, 1.36)
    ax_cb = cell(fig, X0 + WC - 0.42, Y[0] + 0.58, 0.075, 1.36)
    ax_inset = cell(fig, X0 + 0.88, Y[0] + 2.54, WC - 1.34, 0.90)
    draw_sim_spectrogram(fig, ax_b, ax_spec, ax_inset, ax_cb, FS_LABEL, FS_TICK)

    # pre- vs post-cue spectra
    sx, sy, sw, sh = SP_BOX
    draw_sim_spectra(cell(fig, X0 + sx, Y[1] + sy, sw, sh))

    fig.savefig(output_path("fig6_simulated.png"), dpi=400)
    fig.savefig(output_path("fig6_simulated.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    make_figure()

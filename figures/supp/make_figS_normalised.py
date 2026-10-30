"""Supplementary figure: frequency responses normalised by the baseline rate fluctuations."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import supp_common as SC
from asyncresp.io import load_pickle
from asyncresp.paths import OUTPUT, SUPP, SWEEPS

SETS = [("r2b0", 2.0, 0.0, r"$\nu_{drive}$ = 2 Hz, $b$ = 0 pA"),
        ("r4b0", 4.0, 0.0, r"$\nu_{drive}$ = 4 Hz, $b$ = 0 pA"),
        ("r4b60", 4.0, 60.0, r"$\nu_{drive}$ = 4 Hz, $b$ = 60 pA")]
FILES = {1: lambda t: SWEEPS / f"impulse_results_{t}a1_final.pkl",
         2: lambda t: SUPP / "amplitude_2hz" / f"impulse_results_{t}a2_final.pkl"}
DEL_GRID = np.linspace(-17.5, 17.5, 141)
FMT = lambda v: "0" if v == 0 else f"{v:+g}".replace("-", "−")
YMAX = {1: 4.0, 2: 8.0}


def normalised(path):
    """Seed mean and SD of the normalised E amplitude per (Delta_EL, frequency)."""
    df = pd.DataFrame(load_pickle(path))
    df["delta_EL"] = df.EL_exc - df.EL_inh
    df["frequency"] = 1000.0 / df.period
    return (df.groupby(["delta_EL", "frequency"])
              .agg(amp_exc_mean=("amp_exc_norm", "mean"), amp_exc_std=("amp_exc_norm", "std"))
              .reset_index())


def make():
    SC.style()
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 3, left=0.085, right=0.985, top=0.85, bottom=0.09, wspace=0.12,
                          hspace=0.5)
    first = []
    for r, A in enumerate((1, 2)):
        for c, (tag, nu, b, title) in enumerate(SETS):
            agg = normalised(FILES[A](tag))
            colors = SC.rocket(agg.delta_EL.unique())
            ei = SC.mean_field_sweep(DEL_GRID, nu, -70.0, -70.0 - DEL_GRID, b)
            ax = fig.add_subplot(gs[r, c])
            SC.response_panel(ax, agg, "delta_EL", colors, ymax=YMAX[A],
                              eigfreqs=SC.eigenfreqs_at(ei, colors))
            ax.set_ylabel("Normalised E response\n(baseline s.d.)")
            if c:
                ax.set_ylabel(""); ax.set_yticklabels([])
            else:
                first.append(ax)
    fig.legend(handles=[Patch(facecolor=colors[v], edgecolor="none", label=FMT(v))
                        for v in sorted(colors)],
               title=r"$\Delta_{E_L}$ (mV)", ncol=len(colors), loc="upper center",
               bbox_to_anchor=(0.53, 0.998), frameon=True, fancybox=False, edgecolor="0.8",
               handlelength=1.0, handleheight=0.9, columnspacing=0.9, handletextpad=0.4,
               title_fontsize=7, fontsize=6.5, borderpad=0.45)
    for ax, s in zip(first, ("a.", "b.")):
        p = ax.get_position()
        fig.text(0.008, p.y1 + 0.03, s, fontsize=9, fontweight="bold", va="bottom", ha="left")
    out = OUTPUT / "supp"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figS_normalised.png", dpi=300)
    fig.savefig(out / "figS_normalised.pdf")
    return fig


if __name__ == "__main__":
    make()

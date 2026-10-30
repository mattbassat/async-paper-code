"""Supplementary figures: the mean-field gains behind the slow-input responsiveness peak."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import supp_common as SC
from asyncresp.io import load_pickle
from asyncresp.paths import OUTPUT, SWEEPS

SETS = ["r2b0", "r4b0", "r4b60"]
XLIM, XTICKS = (-16.5, 16.5), [-15, -10, -5, 0, 5, 10, 15]
XLAB = r"$\Delta_{E_L}$ (mV)"

# (label, colour, line style) of every curve
GAIN = {"A": (r"$A \equiv \partial \dot\nu_e / \partial \nu_e$", "#1f3fe0", "-"),
        "B": (r"$B \equiv \partial \dot\nu_e / \partial \nu_i$", "#2a8a2a", "-"),
        "C": (r"$C \equiv \partial \dot\nu_i / \partial \nu_e$", "#f0a020", "-"),
        "D": (r"$D \equiv \partial \dot\nu_i / \partial \nu_i$", "#e8221c", "-"),
        "Tdet": (r"$T \det(J^*)$", "#1f1f1f", (0, (4, 2))),
        "AD": (r"$AD$", "#7b1f8f", "-"),
        "BC": (r"$BC$", "#f4a6bd", "-"),
        "disc": (r"$\Delta(J^*)$", "#7a1010", "-"),
        "4det": (r"$4 \det(J^*)$", "#11898a", "-"),
        "tr2": (r"$(\mathrm{tr}(J^*))^2$", "#9a7de0", "-")}


def table(tag):
    """Gains and derived quantities of one parameter set along Delta_EL."""
    e = load_pickle(SWEEPS / f"eigen_df_{tag}_dense.pkl").sort_values("delta_EL")
    x = e.delta_EL.values
    q = dict(A=e.dfedot_dfe.values, B=e.dfedot_dfi.values, C=e.dfidot_dfe.values,
             D=e.dfidot_dfi.values)
    det = q["A"] * q["D"] - q["B"] * q["C"]
    tr = q["A"] + q["D"]
    q.update(Tdet=e.T_stationary.values * det, AD=q["A"] * q["D"], BC=q["B"] * q["C"],
             disc=tr ** 2 - 4 * det, **{"4det": 4 * det, "tr2": tr ** 2})
    d = {k: np.gradient(v, x) for k, v in q.items()}
    return x, q, d, float(e.delta_crit.iloc[0])


def frame(ax, xcrit, zero=True):
    SC._shade(ax, *XLIM, xcrit)
    if zero:
        ax.axhline(0, color=SC.INK2, lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.set_xlim(XLIM); ax.set_xticks(XTICKS)
    ax.set_xlabel(XLAB)
    yl = ax.get_ylim()
    ax.text(xcrit - 0.6, yl[0] + 0.78 * (yl[1] - yl[0]), r"$\Delta_{E_L}^{\mathrm{crit.}}$",
            ha="right", va="center", fontsize=6.0, color=SC.INK)


def curves(ax, x, src, keys):
    for k in keys:
        lab, col, ls = GAIN[k]
        ax.plot(x, src[k], color=col, ls=ls, lw=1.2, zorder=3)


def key(fig, keys, y):
    fig.legend([Line2D([], [], color=GAIN[k][1], ls=GAIN[k][2], lw=1.3) for k in keys],
               [GAIN[k][0] for k in keys], loc="upper center", bbox_to_anchor=(0.53, y),
               ncol=len(keys), frameon=True, fancybox=False, edgecolor="0.8", fontsize=6.3,
               handlelength=1.8, columnspacing=1.2, borderpad=0.45)


def letters(fig, axes_rows, labels):
    for ax, s in zip(axes_rows, labels):
        p = ax.get_position()
        fig.text(0.008, p.y1 + 0.012, s, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    out = OUTPUT / "supp"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def fig_gains(data):
    rows = [(["A", "B", "C", "D", "Tdet"], "q", "Value", (-1100, 2100)),
            (["A", "B", "C", "D", "Tdet"], "d", r"$\frac{d}{d\Delta_{E_L}}$(Value)", (-40, 65)),
            (["AD", "BC"], "q", "Value", None)]
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.4),
                             gridspec_kw=dict(left=0.1, right=0.985, top=0.92, bottom=0.07,
                                              wspace=0.22, hspace=0.5))
    for c, tag in enumerate(SETS):
        x, q, d, xc = data[tag]
        for r, (keys, src, ylab, ylim) in enumerate(rows):
            ax = axes[r, c]
            curves(ax, x, q if src == "q" else d, keys)
            if ylim:
                ax.set_ylim(ylim)
            if r == 2:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            frame(ax, xc)
            if c == 0:
                ax.set_ylabel(ylab)
            elif ylim:                                # rows a, b share their y range
                ax.set_yticklabels([])
    key(fig, ["A", "B", "C", "D", "Tdet", "AD", "BC"], 0.998)
    letters(fig, axes[:, 0], ("a.", "b.", "c."))
    save(fig, "figS_gains")


def fig_discriminant(data):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True,
                             gridspec_kw=dict(left=0.1, right=0.985, top=0.8, bottom=0.18,
                                              wspace=0.1))
    for c, tag in enumerate(SETS):
        x, q, d, xc = data[tag]
        curves(axes[c], x, q, ["tr2", "4det", "disc"])
        axes[c].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        frame(axes[c], xc)
    axes[0].set_ylabel("Value")
    key(fig, ["tr2", "4det", "disc"], 0.995)
    save(fig, "figS_discriminant")


def fig_log_derivatives(data):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4),
                             gridspec_kw=dict(left=0.1, right=0.985, top=0.8, bottom=0.18,
                                              wspace=0.28))
    for c, tag in enumerate(SETS):
        x, q, d, xc = data[tag]
        logd = {k: d[k] / q[k] for k in ("C", "D", "Tdet")}
        curves(axes[c], x, logd, ["C", "D", "Tdet"])
        frame(axes[c], xc)
    axes[0].set_ylabel(r"$\frac{d}{d\Delta_{E_L}}$(Value) / Value")
    key(fig, ["C", "D", "Tdet"], 0.995)
    save(fig, "figS_log_derivatives")


def make():
    SC.style()
    data = {tag: table(tag) for tag in SETS}
    fig_gains(data)
    fig_discriminant(data)
    fig_log_derivatives(data)


if __name__ == "__main__":
    make()

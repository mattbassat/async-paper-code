"""Fig. 2d-f: mean-field stability along the ΔEL axis and its match to the spiking network."""
import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from asyncresp import mean_field as MF
from asyncresp.io import load_pickle
from asyncresp.paths import SWEEPS
from asyncresp.style import INK, INK2

from fig2_common import (CONDITIONS, SH_OD, SH_UD, C_RE, C_IM, C_BAL, B_SYM, B_CRIT,
                         FMIN_RES, balance, eigen_sweep)

DEL_TICKS = [-15, -10, -5, 0, 5, 10, 15]


# --- d: eigenvalues and B ---
def draw_eigen(axes_top, axes_bot, lo=-17.0, hi=17.0):
    """Top: real and imaginary parts of the two eigenvalues (s^-1). Bottom: B (log scale)."""
    for axt, axb, (tag, ext, b, title) in zip(axes_top, axes_bot, CONDITIONS):
        ei = eigen_sweep(ext, b)
        d, dc = ei["dEL"], ei["dcrit"]
        for ax in (axt, axb):
            ax.axvspan(lo, dc, color=SH_OD, lw=0, zorder=0)
            ax.axvspan(dc, hi, color=SH_UD, lw=0, zorder=0)
            ax.axvline(dc, color=INK, lw=0.6, ls=(0, (2, 2)), zorder=1)
            ax.set_xlim(lo, hi); ax.set_xticks(DEL_TICKS)

        re1, re2 = np.real(ei["lam1"]), np.real(ei["lam2"])
        im = np.abs(np.imag(ei["lam1"]))
        axt.axhline(0, color=INK2, lw=0.5, zorder=1)
        axt.plot(d, re1, color=C_RE, lw=1.2, zorder=3)
        axt.plot(d, re2, color=C_RE, lw=1.2, zorder=3)
        axt.plot(d, im, color=C_IM, lw=1.2, zorder=3)
        axt.plot(d, -im, color=C_IM, lw=1.2, zorder=3)
        axt.set_title(title, fontsize=7.5, pad=3)
        axt.tick_params(labelbottom=False)

        axb.set_yscale("log")                     # B spans more than a decade
        axb.plot(d, balance(ei), color=C_BAL, lw=1.2, zorder=3)
        axb.axhline(B_CRIT, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=2)
        axb.plot([dc], [B_CRIT], "o", color=INK, markersize=3.2, markeredgecolor="white",
                 markeredgewidth=0.5, zorder=4)
        axb.set_xlabel(r"$\Delta_{E_L}$ (mV)")
        axb.set_ylim(0.028, 3.2); axb.set_yticks([0.1, 1])

    yl = max(abs(v) for ax in axes_top for v in ax.get_ylim())
    for ax in axes_top:
        ax.set_ylim(-yl, yl)
    for ax in list(axes_top[1:]) + list(axes_bot[1:]):
        ax.tick_params(labelleft=False)
    axes_top[0].set_ylabel(r"Eigenvalues $\lambda_{1,2}$ (s$^{-1}$)")
    axes_bot[0].set_ylabel("Balance metric " + B_SYM, linespacing=1.3)
    axes_top[0].legend([Line2D([], [], color=C_RE, lw=1.2), Line2D([], [], color=C_IM, lw=1.2)],
                       [r"$\mathrm{Re}\{\lambda_{1,2}\}$", r"$\mathrm{Im}\{\lambda_{1,2}\}$"],
                       loc="upper left", frameon=False,
                       fontsize=6.2, handlelength=1.1, labelspacing=0.25, borderaxespad=0.25)
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    axes_bot[0].text(0.035, 0.30, "node\n(OD)", transform=axes_bot[0].transAxes, fontsize=6.2,
                     color=INK2, va="center", ha="left", linespacing=1.2, zorder=6,
                     path_effects=halo)
    axes_bot[0].text(0.965, 0.30, "focus\n(UD)", transform=axes_bot[0].transAxes, fontsize=6.2,
                     color="#2a6fb0", va="center", ha="right", linespacing=1.2, zorder=6,
                     path_effects=halo)


# --- e: response maps ---
IMPULSE_FILES = {"r2b0": "impulse_results_r2b0a1_final.pkl",
                 "r4b0": "impulse_results_r4b0a1_final.pkl",
                 "r4b60": "impulse_results_r4b60a1_final.pkl"}


def load_impulse(tag):
    """SNN response amplitude of the E population (Hz) to a sinusoidal drive, averaged over
    seeds, as a long table with columns delta_EL (mV), f (Hz) and amp_exc."""
    imp = pd.DataFrame(load_pickle(SWEEPS / IMPULSE_FILES[tag]))
    imp["delta_EL"] = imp.EL_exc - imp.EL_inh
    imp["f"] = 1000 / imp.period
    return imp.groupby(["delta_EL", "f"]).amp_exc.mean().reset_index()


def draw_heatmap(axes, cax, fmin=0.5):
    """The three response maps on one absolute colour scale, with the mean-field eigenfrequency
    and boundary on top. Returns the legend handles."""
    data = []
    for tag, ext, b, title in CONDITIONS:
        agg = load_impulse(tag)
        dels = np.sort(agg.delta_EL.unique())
        fs = np.sort(agg.f.unique()); fs = fs[fs >= fmin]
        Z = np.array([[agg[(agg.delta_EL == d) & np.isclose(agg.f, f)].amp_exc.item()
                       for d in dels] for f in fs])
        data.append((dels, fs, Z))
    vmax = max(Z.max() for _, _, Z in data)

    for j, (ax, (tag, ext, b, title), (dels, fs, Z)) in enumerate(zip(axes, CONDITIONS, data)):
        g = np.sqrt(fs[1:] * fs[:-1])                     # cell edges centred in log f
        ye = np.r_[fs[0] ** 2 / g[0], g, fs[-1] ** 2 / g[-1]]
        step = np.median(np.diff(dels)) / 2
        pcm = ax.pcolormesh(np.r_[dels - step, dels[-1] + step], ye, Z, cmap="viridis",
                            vmin=0, vmax=vmax, shading="flat", rasterized=True)

        ei = eigen_sweep(ext, b)
        ax.plot(ei["dEL"], ei["fd"], color=INK, lw=2.2, zorder=3)
        ax.plot(ei["dEL"], ei["fd"], color="white", lw=1.1, zorder=4)
        ax.axvline(ei["dcrit"], color="white", lw=0.8, ls=(0, (2, 2)), zorder=3)
        pk = np.array([fs[np.argmax(Z[:, i])] for i in range(len(dels))])
        res = (dels > ei["dcrit"]) & (pk > FMIN_RES)      # UD networks with a resonance
        ax.scatter(dels[res], pk[res], s=9,
                   color="white", edgecolor=INK, linewidth=0.5, zorder=5)

        ax.set_yscale("log"); ax.set_ylim(fmin, 100); ax.set_xlim(-17.5, 17.5)
        ax.set_xticks(DEL_TICKS)
        ax.set_yticks([1, 10, 100])
        ax.set_yticklabels(["1", "10", "100"] if j == 0 else [])
        ax.set_xlabel(r"$\Delta_{E_L}$ (mV)")
        ax.set_title(title, fontsize=7.5, pad=3)
    axes[0].set_ylabel("Input frequency (Hz)")
    cb = plt.colorbar(pcm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1.0, 1.5, 2.0])
    cb.ax.tick_params(labelsize=6.0, length=2.0, width=0.6, pad=1.5)
    cb.outline.set_linewidth(0.5)
    cb.set_label("E response amplitude (Hz)", fontsize=6.4, labelpad=2)
    return [Line2D([], [], color="white", lw=1.1,
                   path_effects=[pe.withStroke(linewidth=2.2, foreground=INK)],
                   label=r"MF eigenfrequency $|\mathrm{Im}\{\lambda_{1,2}\}|/2\pi$"),
            Line2D([], [], marker="o", ls="", ms=3.2, mfc="white", mec=INK, label="SNN peak (UD)"),
            Line2D([], [], color=INK, lw=0.8, ls=(0, (2, 2)),
                   label=f"MF boundary ({B_SYM} = {B_CRIT:g})")]


# --- f: resonance vs B ---
SETCOL = {"r2b0": "#2a78d6", "r4b0": "#eb6834", "r4b60": "#1baf7a"}
SETMK = {"r2b0": "o", "r4b0": "s", "r4b60": "^"}
SNN_MS = 6.2


def resonance_points(tag, ext, b):
    """For each network of a sweep: ΔEL, the mean-field damping ratio zeta (B = zeta^2), the
    natural frequency f0 (Hz) and the SNN resonance frequency, i.e. the swept input frequency
    of maximal E response, or NaN when that maximum is not above FMIN_RES."""
    agg = load_impulse(tag)
    rows = []
    for d, g in agg.groupby("delta_EL"):
        g = g.sort_values("f")
        ei = MF.eigen(MF.fixed_point(ext=ext, ELe=-70., ELi=-70. - float(d), b=b))
        fpk = float(g.f.values[int(np.argmax(g.amp_exc.values))])
        fpk = fpk if fpk > FMIN_RES else np.nan
        rows.append((float(d), float(ei["zeta"][0]), float(ei["f0"][0]), fpk))
    return np.array(rows)


def draw_resonance_vs_balance(ax):
    """SNN resonance frequency / f0 against B for all networks with B < 1, on the mean-field
    curve sqrt(1 - B), which reaches zero at the boundary."""
    xlo, xhi = 0.030, 1.30
    ylo, yhi = 0.0, 1.15
    curve = lambda x: np.sqrt(np.maximum(0.0, 1 - x))
    ax.set_xscale("log")
    xx = np.geomspace(xlo, B_CRIT, 400)
    ax.plot(xx, curve(xx), color=INK, lw=1.0, zorder=2)
    ax.axvspan(B_CRIT, xhi, color=SH_OD, lw=0, zorder=0)
    ax.axvspan(xlo, B_CRIT, color=SH_UD, lw=0, zorder=0)
    ax.axvline(B_CRIT, color=INK, lw=0.7, zorder=1)
    for tag, ext, b, title in CONDITIONS:
        P = resonance_points(tag, ext, b)
        P = P[P[:, 1] < 1.0]                              # focus side only
        P = P[np.isfinite(P[:, 3])]                       # networks with an SNN resonance
        ax.plot(P[:, 1] ** 2, P[:, 3] / P[:, 2], SETMK[tag], color=SETCOL[tag], ms=SNN_MS,
                ls="none", markeredgecolor="white", markeredgewidth=0.7, zorder=3, label=title,
                alpha=0.95)
    ax.set_xlim(xlo, xhi); ax.set_xticks([0.1, 1])
    ax.set_ylim(ylo, yhi); ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel(r"$f_{peak}\,/\,f_0$"); ax.set_xlabel("Balance metric " + B_SYM)
    xc = 0.075                                            # curve label, under the curve
    ax.text(xc, curve(xc) - 0.09 * (yhi - ylo), r"$f_{\mathrm{eig}}/f_0 = \sqrt{1-\mathcal{B}}$",
            fontsize=7.0, color=INK, ha="center", va="top")
    ax.text(B_CRIT + 0.52 * (xhi - B_CRIT), 0.5 * (ylo + yhi), "no resonance (OD)", fontsize=6.0,
            color=INK2, ha="center", va="center", rotation=90)
    ax.legend(frameon=False, fontsize=6.0, loc="lower left",
              handlelength=0.8, labelspacing=0.25, borderaxespad=0.3, handletextpad=0.4)

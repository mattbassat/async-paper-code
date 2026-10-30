"""Panel functions shared by the supplementary figures, drawn in the style of the main figures:"""
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.patheffects as pe
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from asyncresp import mean_field as MF
from asyncresp.style import E_COL, I_COL, INK, INK2

SH_OD, SH_UD = "#e3e3e3", "#dbeefb"         # background bands: OD / UD side
UD_INK = "#2a6fb0"
C_RE, C_IM, C_BAL = "#1f1f1f", "#2a78d6", "#a03d8a"
B_SYM = r"$\mathcal{B}$"
SLOW_F = 0.5                                # slow-input frequency (Hz)

STYLE = {
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.3, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def style():
    sns.set_style("white")
    mpl.rcParams.update(STYLE)


def rocket(values, reverse=False):
    """Discrete rocket palette keyed by the sorted values (dark -> light); reverse=True flips it
    so that light is always the resonant (UD) side of a sweep."""
    values = np.sort(np.unique(values))
    pal = sns.color_palette("rocket", n_colors=len(values))
    return dict(zip(values, pal[::-1] if reverse else pal))


def letter(ax, s, dx=-0.30, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom",
            ha="left")


# --- mean-field eigenvalues ---
def eigen_from_df(df, x="delta_EL"):
    """Eigenvalues, B and the boundary from a stored mean-field table (Jacobian entries A-D)."""
    df = df.sort_values(x)
    A, B, C, D = (df[k].values for k in ("dfedot_dfe", "dfedot_dfi", "dfidot_dfe", "dfidot_dfi"))
    tr, det = A + D, A * D - B * C
    disc = tr ** 2 - 4 * det
    sq = np.sqrt(np.abs(disc).astype(complex)) * np.where(disc >= 0, 1.0, 1j)
    xs = df[x].values
    j = np.where(np.diff(np.sign(disc)) != 0)[0]
    xcrit = float(np.interp(0, [disc[j[0]], disc[j[0] + 1]], [xs[j[0]], xs[j[0] + 1]])) \
        if len(j) else np.nan
    T = df["T_stationary"].values
    return dict(x=xs, lam1=(tr + sq) / 2, lam2=(tr - sq) / 2, B=tr ** 2 / (4 * det),
                fd=np.where(disc < 0, np.sqrt(np.maximum(-disc, 0)) / (4 * np.pi), np.nan),
                xcrit=xcrit, He0=-D / (T * det) - 1, Hi0=C / (T * det), od_low=bool(disc[0] > 0))


def _shade(ax, lo, hi, xcrit, od_left=True):
    left, right = (SH_OD, SH_UD) if od_left else (SH_UD, SH_OD)
    ax.axvspan(lo, xcrit, color=left, lw=0, zorder=0)
    ax.axvspan(xcrit, hi, color=right, lw=0, zorder=0)
    ax.axvline(xcrit, color=INK, lw=0.6, ls=(0, (2, 2)), zorder=1)


def eigen_panels(axt, axb, ei, xlim, xlabel, xticks, od_left=True, show_text=True,
                 show_legend=True):
    """Fig. 2d style: Re/Im parts of the eigenvalues (top) and B on a log axis (bottom)."""
    lo, hi = xlim
    for ax in (axt, axb):
        _shade(ax, lo, hi, ei["xcrit"], od_left)
        ax.set_xlim(lo, hi); ax.set_xticks(xticks)
    x = ei["x"]
    im = np.abs(np.imag(ei["lam1"]))
    axt.axhline(0, color=INK2, lw=0.5, zorder=1)
    for y, c in ((np.real(ei["lam1"]), C_RE), (np.real(ei["lam2"]), C_RE), (im, C_IM), (-im, C_IM)):
        axt.plot(x, y, color=c, lw=1.2, zorder=3)
    yl = max(abs(v) for v in axt.get_ylim())
    axt.set_ylim(-yl, yl)
    axt.tick_params(labelbottom=False)
    axt.set_ylabel(r"$\lambda_{1,2}$ (s$^{-1}$)")
    if show_legend:
        axt.legend([Line2D([], [], color=C_RE, lw=1.2), Line2D([], [], color=C_IM, lw=1.2)],
                   [r"$\mathrm{Re}\{\lambda_{1,2}\}$", r"$\mathrm{Im}\{\lambda_{1,2}\}$"],
                   loc="upper left" if od_left else "upper right", frameon=False, fontsize=6.2,
                   handlelength=1.1, labelspacing=0.25, borderaxespad=0.25)
    axb.set_yscale("log")
    axb.plot(x, ei["B"], color=C_BAL, lw=1.2, zorder=3)
    axb.axhline(1.0, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=2)
    axb.plot([ei["xcrit"]], [1.0], "o", color=INK, ms=3.2, mec="white", mew=0.5, zorder=4)
    axb.set_ylim(0.028, 3.2); axb.set_yticks([0.1, 1])
    axb.set_xlabel(xlabel)
    axb.set_ylabel(B_SYM)
    if show_text:
        halo = [pe.withStroke(linewidth=1.8, foreground="white")]
        xo, xu = (0.035, 0.965) if od_left else (0.965, 0.035)
        axb.text(xo, 0.30, "node\n(OD)", transform=axb.transAxes, fontsize=6.2, color=INK2,
                 va="center", ha="left" if od_left else "right", linespacing=1.2, zorder=6,
                 path_effects=halo)
        axb.text(xu, 0.30, "focus\n(UD)", transform=axb.transAxes, fontsize=6.2, color=UD_INK,
                 va="center", ha="right" if od_left else "left", linespacing=1.2, zorder=6,
                 path_effects=halo)


def mean_field_sweep(x, ext, ELe, ELi, b=0.0, T_fixed=None, **kw):
    """Mean-field quantities along a swept parameter x (all inputs broadcast against x):
    eigenvalues, B, ringing frequency |Im lambda| / 2 pi, the node/focus boundary (x at which
    tr^2 - 4 det changes sign; NaN if none in range) and the slow-input responses of Eq. 4.
    `T_fixed` (s) replaces the state-dependent time constant T = tau_V by a constant."""
    x = np.asarray(x, float)
    n = len(x)
    full = lambda v: np.broadcast_to(np.asarray(v, float), (n,)).copy()
    fp = MF.fixed_point(full(ext), full(ELe), full(ELi), full(b), **kw)
    if T_fixed is not None:
        fp["T"] = np.full(n, T_fixed)
    ei = MF.eigen(fp)
    A, B, C, D, tr, det, disc = (np.ravel(ei[k]) for k in ("A", "B", "C", "D", "tr", "det", "disc"))
    T = np.ravel(fp["T"])
    sq = np.sqrt(np.abs(disc).astype(complex)) * np.where(disc >= 0, 1.0, 1j)
    j = np.where(np.diff(np.sign(disc)) != 0)[0]
    xcrit = (float(np.interp(0, [disc[j[0]], disc[j[0] + 1]], [x[j[0]], x[j[0] + 1]]))
             if len(j) else np.nan)
    return dict(x=x, lam1=(tr + sq) / 2, lam2=(tr - sq) / 2, B=tr ** 2 / (4 * det),
                fd=np.where(disc < 0, np.sqrt(np.maximum(-disc, 0)) / (4 * np.pi), np.nan),
                xcrit=xcrit, He0=-D / (T * det) - 1, Hi0=C / (T * det), T=T,
                od_low=bool(disc[0] > 0))


def stored_crit(tag):
    """Delta_EL^crit of one of the three parameter sets of Figs. 2-3 ("r2b0", "r4b0", "r4b60"),
    as stored with its mean-field table and used in Fig. 3a and the text."""
    from asyncresp.io import load_pickle
    from asyncresp.paths import SWEEPS
    return float(load_pickle(SWEEPS / f"eigen_df_{tag}_dense.pkl")["delta_crit"].iloc[0])


def eigenfreqs_at(ei, values):
    """Mean-field ringing frequency (Hz) at each swept value; NaN on the node (OD) side."""
    f = np.nan_to_num(ei["fd"], nan=-1.0)
    out = {v: float(np.interp(v, ei["x"], f)) for v in values}
    return {v: (q if q > 0 else np.nan) for v, q in out.items()}


def eigen_panel(ax, ei, xlim, xlabel, xticks, od_left=True, show_legend=True, labels=True):
    """Fig. 2d (top) style: real (black) and imaginary (blue) parts of the two eigenvalues of
    the mean-field Jacobian against the swept parameter, on the OD / UD shading."""
    lo, hi = xlim
    xc = ei["xcrit"]
    if np.isfinite(xc):
        _shade(ax, lo, hi, xc, od_left)
    else:                                         # no boundary in range: one regime throughout
        ax.axvspan(lo, hi, color=SH_OD if ei["od_low"] else SH_UD, lw=0, zorder=0)
    ax.set_xlim(lo, hi); ax.set_xticks(xticks)
    x = ei["x"]
    im = np.abs(np.imag(ei["lam1"]))
    ax.axhline(0, color=INK2, lw=0.5, zorder=1)
    for y, c in ((np.real(ei["lam1"]), C_RE), (np.real(ei["lam2"]), C_RE), (im, C_IM), (-im, C_IM)):
        ax.plot(x, y, color=c, lw=1.2, zorder=3)
    yl = max(abs(v) for v in ax.get_ylim())
    ax.set_ylim(-yl, yl)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Eigenvalues $\lambda_{1,2}$ (s$^{-1}$)")
    if labels:
        for lab, col, w in (("OD", INK2, True), ("UD", UD_INK, False)):
            if np.isfinite(ei["xcrit"]):
                left = (lo + xc) / 2, (xc + hi) / 2
                xpos = left[0] if (lab == "OD") == od_left else left[1]
                width = ((xc - lo) if (lab == "OD") == od_left else (hi - xc)) / (hi - lo)
            else:
                xpos, width = (lo + hi) / 2, (1.0 if lab == "UD" else 0.0)
            if width > 0.12:
                ax.text(xpos, 0.96 * yl, lab, ha="center", va="top", fontsize=6.5,
                        weight="bold", color=col)
    if show_legend:
        ax.legend([Line2D([], [], color=C_RE, lw=1.2), Line2D([], [], color=C_IM, lw=1.2)],
                  [r"$\mathrm{Re}\{\lambda_{1,2}\}$", r"$\mathrm{Im}\{\lambda_{1,2}\}$"],
                  loc="lower left" if od_left else "lower right", frameon=False, fontsize=6.0,
                  handlelength=1.1, labelspacing=0.2, borderaxespad=0.2)


def eigen_key(fig, x, y, loc="upper left"):
    """Re / Im key for the eigenvalue panels, placed in figure coordinates."""
    fig.legend([Line2D([], [], color=C_RE, lw=1.2), Line2D([], [], color=C_IM, lw=1.2)],
               [r"$\mathrm{Re}\{\lambda_{1,2}\}$", r"$\mathrm{Im}\{\lambda_{1,2}\}$"],
               loc=loc, bbox_to_anchor=(x, y), ncol=2, frameon=False, fontsize=6.3,
               handlelength=1.4, columnspacing=1.0)


# --- frequency responses ---
def _plain_log(axis):
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def response_panel(ax, agg, param, colors, ymax, eigfreqs=None, xrange=(0.5, 100)):
    """Fig. 1c style: mean (+/- SD over seeds) E response vs drive frequency, one curve per
    parameter value. `eigfreqs` ({value: Hz}) adds the mean-field eigenfrequency of each UD
    network as a dashed line in the curve's colour."""
    for v in np.sort(agg[param].unique()):
        s = agg[agg[param] == v].sort_values("frequency")
        ax.fill_between(s.frequency, s.amp_exc_mean - s.amp_exc_std,
                        s.amp_exc_mean + s.amp_exc_std, color=colors[v], alpha=0.25, lw=0)
        ax.plot(s.frequency, s.amp_exc_mean, color=colors[v], lw=1.3)
        if eigfreqs is not None and np.isfinite(eigfreqs.get(v, np.nan)):
            ax.axvline(eigfreqs[v], color=colors[v], lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=10))
    _plain_log(ax.xaxis)
    ax.set_xlim(xrange); ax.set_ylim(0, ymax)
    ax.set_xlabel(r"$f$ (Hz)"); ax.set_ylabel("E response amplitude (Hz)")


def value_legend(ax, colors, title, fmt):
    """Discrete colour key above the axes (as in Fig. 1c)."""
    handles = [Patch(facecolor=colors[v], edgecolor="none", label=fmt(v)) for v in sorted(colors)]
    ax.legend(handles=handles, title=title, ncol=len(handles), loc="lower left",
              bbox_to_anchor=(0.0, 1.03), frameon=True, fancybox=False, edgecolor="0.8",
              framealpha=1.0, handlelength=0.8, handleheight=0.9, columnspacing=0.45,
              handletextpad=0.25, title_fontsize=6.8, fontsize=5.9, borderaxespad=0.0,
              borderpad=0.45)


# --- slow-input responses ---
def slow_input_panel(ax, agg, param, ei, xlim, xticks, xlabel, ymax, colors, od_left=True,
                     crit_label=r"$\Delta_{E_L}^{\mathrm{crit.}}$", crit_y=0.83):
    """E responses to a slow (0.5 Hz) input (mean +- SD over seeds), each point in the colour of
    its network as in the frequency-response panels, on the OD / UD shading of the mean-field
    boundary (dashed line, labelled `crit_label`)."""
    lo, hi = xlim
    xc = ei["xcrit"]
    if np.isfinite(xc):
        _shade(ax, lo, hi, xc, od_left)
        if crit_label:
            ax.text(xc + (-0.012 if od_left else 0.012) * (hi - lo), crit_y * ymax, crit_label,
                    ha="right" if od_left else "left", va="center", fontsize=6.3, color=INK)
        od_x, ud_x = ((0.5 * (lo + xc), 0.5 * (xc + hi)) if od_left
                      else (0.5 * (xc + hi), 0.5 * (lo + xc)))
        od_w = (xc - lo if od_left else hi - xc) / (hi - lo)
    else:
        ax.axvspan(lo, hi, color=SH_OD if ei["od_low"] else SH_UD, lw=0, zorder=0)
        od_x = ud_x = 0.5 * (lo + hi)
        od_w = 1.0 if ei["od_low"] else 0.0
    for x0, txt, col, w in ((od_x, "OD", INK2, od_w), (ud_x, "UD", UD_INK, 1 - od_w)):
        if w > 0.12:                              # skip the label of a very narrow band
            ax.text(x0, 0.955 * ymax, txt, ha="center", va="top", fontsize=6.5, weight="bold",
                    color=col)
    s = agg[np.isclose(agg.frequency, SLOW_F)].sort_values(param)
    for v, m, sd in zip(s[param], s.amp_exc_mean, s.amp_exc_std):
        c = colors.get(v, INK)
        ax.errorbar([v], [m], yerr=[sd], fmt="o", ms=3.0, color=c, mec="white", mew=0.4,
                    ecolor=c, elinewidth=0.7, capsize=2.4, capthick=0.7, zorder=4)
    ax.set_xlim(lo, hi); ax.set_ylim(0, ymax); ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"E response to slow input\n($f = {SLOW_F:g}$ Hz) (Hz)")


def ei_key(fig, ax, right=False):
    """E / I and analytical / SNN key in one row above the axes (right-aligned if `right`)."""
    h_snn = ax.errorbar([np.nan], [np.nan], yerr=[1.0], fmt="o", ms=2.6, color=INK,
                        elinewidth=0.6, capsize=2.4, capthick=0.6)
    ax.legend(handles=[Patch(facecolor=E_COL, label="E"), Patch(facecolor=I_COL, label="I"),
                       Line2D([], [], color=INK, lw=1.1), h_snn],
              labels=["E", "I", "Eq. 4", "SNN"], ncol=4,
              loc="lower right" if right else "lower left",
              bbox_to_anchor=(1.0 if right else 0.0, 1.03), frameon=True, fancybox=False, edgecolor="0.8",
              framealpha=1.0, fontsize=6.3, handlelength=1.4, columnspacing=0.9,
              handletextpad=0.4, borderaxespad=0.0, borderpad=0.5)

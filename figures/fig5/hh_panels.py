"""Panels of Fig. 5 (Hodgkin-Huxley network).

    a  E response amplitude against input frequency, one curve per Delta_EL (mean +/- s.d.
       over 10 seeds), with a Delta_EL colour bar above the axes
    b  the slow-input (0.5 Hz) E response against Delta_EL
    c  spontaneous LFP-proxy spectra of four networks; triangles mark the frequency of the
       peak mean evoked E response of the same network (panel a), when above 10 Hz, i.e. the
       low-gamma E/I resonance rather than the high-gamma peak of the I-I loop
"""
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

from hh_data import LOW_FMAX, SLOW_F, SPECTRA_DEL, response_curves, spont_spectrum


def _log_freq_axis(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _mv_label(d):
    """Signed Delta_EL label with a typographic minus, e.g. "−10", "0", "+10"."""
    return "0" if d == 0 else f"{d:+g}".replace("-", "−")


def delta_colorbar(fig, ax, values, colors, label=r"$\Delta_{E_L}$ (mV)"):
    """A thin discrete Delta_EL colour bar over panel a, in place of a 13-entry legend."""
    v = np.sort(np.unique(values))
    step = np.median(np.diff(v))
    bounds = np.r_[v - step / 2, v[-1] + step / 2]
    p = ax.get_position()
    cax = fig.add_axes([p.x0 + 0.18 * p.width, p.y1 + 0.030, 0.64 * p.width, 0.016])
    cb = ColorbarBase(cax, cmap=ListedColormap([colors[x] for x in v]),
                      norm=BoundaryNorm(bounds, len(v)), orientation="horizontal",
                      ticks=[-20, -10, 0, 10])
    cb.ax.set_xticklabels(["−20", "−10", "0", "+10"])
    cb.ax.tick_params(labelsize=5.8, length=1.6, width=0.5, pad=1.2)
    cb.outline.set_linewidth(0.4)
    cax.set_title(label, fontsize=6.4, pad=2)
    return cax


def draw_responses(ax, df, colors, xrange=(0.5, 200)):
    """Fig. 5a: frequency-response curves, mean +/- s.d. over seeds."""
    g = response_curves(df)
    for d, sub in g.groupby("delta_EL"):
        sub = sub.sort_values("f")
        m = (sub.f >= xrange[0]) & (sub.f <= xrange[1])
        col = colors[d]
        ax.fill_between(sub.f[m], (sub["mean"] - sub["std"])[m], (sub["mean"] + sub["std"])[m],
                        color=col, alpha=0.16, linewidth=0)
        ax.plot(sub.f[m], sub["mean"][m], color=col, linewidth=1.0)
    ax.set_xscale("log")
    _log_freq_axis(ax)
    ax.set_xlim(*xrange); ax.set_ylim(0, None)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("E response amplitude (Hz)")


def draw_slow(ax, df, colors):
    """Fig. 5b: E response to the slow input against Delta_EL, mean +/- s.d. over seeds."""
    s = df[np.isclose(df.f, SLOW_F)].groupby("delta_EL").amp_exc.agg(["mean", "std"]).reset_index()
    for _, r in s.iterrows():
        ax.errorbar(r.delta_EL, r["mean"], yerr=r["std"], fmt="o", markersize=4.2,
                    color=colors[r.delta_EL], markeredgecolor="black", markeredgewidth=0.45,
                    ecolor=colors[r.delta_EL], elinewidth=0.9, capsize=3.6, capthick=1.0,
                    zorder=3)
    ticks = [-20, -15, -10, -5, 0, 5, 10]
    ax.set_xlim(s.delta_EL.min() - 2.5, s.delta_EL.max() + 2.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([_mv_label(v) for v in ticks])
    ax.set_ylim(0, None)
    ax.set_xlabel(r"$\Delta_{E_L}$ (mV)")
    ax.set_ylabel(rf"E response amplitude (Hz)" + "\n" + rf"($f = {SLOW_F:g}$ Hz)")


def draw_spectra(ax, peaks, colors, values=SPECTRA_DEL, fmin=2, fmax=500):
    """Fig. 5c: spontaneous LFP spectra, with a triangle above each curve at the network's
    resonance peak (from the evoked responses of panel a) when that peak lies above 10 Hz."""
    for d in values:
        f_, p_ = spont_spectrum(d)
        m = (f_ >= fmin) & (f_ <= fmax)
        col = colors[d]
        ax.plot(f_[m], p_[m], color=col, linewidth=1.0, label=_mv_label(d) + " mV")
        f_pk = peaks.get(d)
        if f_pk is not None and f_pk > LOW_FMAX:
            y = np.exp(np.interp(np.log(f_pk), np.log(f_[m]), np.log(p_[m])))
            ax.scatter([f_pk], [y * 1.45], marker="v", s=20, color=col, edgecolor="black",
                       linewidth=0.45, zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    _log_freq_axis(ax)
    ax.set_xlim(fmin, fmax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Spontaneous LFP PSD (a.u.)")
    leg = ax.legend(loc="lower left", frameon=False, fontsize=5.8, handlelength=1.2,
                    borderaxespad=0.25, labelspacing=0.25,
                    title=r"$\Delta_{E_L}$", title_fontsize=5.8)
    leg._legend_box.align = "left"
    ax.add_artist(leg)
    marker = Line2D([], [], marker="v", ls="", ms=4.2, color="0.45", mec="black", mew=0.45)
    ax.legend([marker], ["SNN resonance peak (a)"], loc="upper left", frameon=False,
              fontsize=5.8, handlelength=1.0, borderaxespad=0.25)

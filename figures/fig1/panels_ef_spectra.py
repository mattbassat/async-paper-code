"""Fig. 1e-f: power spectra of the spontaneous (undriven) LFP proxy."""
import numpy as np
from matplotlib.lines import Line2D

from asyncresp.io import load_pickle
from asyncresp.paths import SPONTANEOUS
from common import plain_log_ticks, rocket_colors, welch_psd_lfp

SEEDS = list(range(300, 310))
WINDOW_MS = (2000.0, 10000.0)               # analysis window; the first 2 s are discarded
PANEL_E_DELTA_EL = [-15.0, -5.0, 5.0, 15.0]

# Fig. 1f, in reading order: (sweep, title, (OD value, UD value), reverse palette, unit, label format)
PANEL_F_SWEEPS = [
    ("delta_gL", r"Leak conductance, $\Delta g_L$", (-5.0, 5.0), False, "nS",
     lambda v: f"{v:+g}".replace("-", "−")),
    ("b", r"Adaptation, $b$", (150.0, 0.0), True, "pA", lambda v: f"{v:g}"),
    ("Qee", r"E$\rightarrow$E synapses, $Q_{ee}$", (1.25, 1.6), False, "nS", lambda v: f"{v:.2f}"),
    ("Qei", r"E$\rightarrow$I synapses, $Q_{ei}$", (2.0, 1.4), True, "nS", lambda v: f"{v:.1f}"),
]


def _signed_tag(x, fmt):
    return format(x, fmt).replace("+", "p").replace("-", "m")


def delta_el_files(delta_EL):
    """Spontaneous runs of the Delta_EL sweep network (nu_drive = 2 Hz, b = 0 pA)."""
    tag = _signed_tag(delta_EL, "+.1f").replace(".", "_")
    return [SPONTANEOUS / "adex_r2b0" / f"r2b0_spontWS_{tag}_{s}.pkl" for s in SEEDS]


def param_files(param, value):
    """Spontaneous runs of one network of the Delta_gL, b, Q_ee or Q_ei sweep."""
    return [SPONTANEOUS / "adex_params" / f"spont_{param}_{_signed_tag(value, '+.2f')}_{s}.pkl"
            for s in SEEDS]


def spont_spectrum(paths, window=WINDOW_MS):
    """Seed-averaged Welch PSD of the inhibitory LFP proxy within `window` (ms)."""
    psds, freqs = [], None
    for path in paths:
        o = load_pickle(path)
        x, dt_lfp = o["lfp_inh"], o["dt_lfp_ms"]
        i0, i1 = int(round(window[0] / dt_lfp)), int(round(window[1] / dt_lfp))
        freqs, p = welch_psd_lfp(x[i0:i1], dt_lfp)
        psds.append(p)
    return freqs, np.mean(psds, axis=0)


def _mark_peak(ax, f_, p_, f_pk, color, lift, size, edge_lw):
    """Triangle above the spectrum at the network's resonance frequency."""
    y_pk = np.exp(np.interp(np.log(f_pk), np.log(f_), np.log(p_)))
    ax.scatter([f_pk], [y_pk * lift], marker="v", s=size, color=color, edgecolor="black",
               linewidth=edge_lw, zorder=5)


def draw_panel_e(ax, data_dEL, peaks, fmin=2, fmax=500):
    """Fig. 1e: spectra of four networks of the Delta_EL sweep, colours as in Fig. 1c."""
    colors = rocket_colors(data_dEL["delta_EL"].unique())
    for dEL in PANEL_E_DELTA_EL:
        f_, p_ = spont_spectrum(delta_el_files(dEL))
        m = (f_ >= fmin) & (f_ <= fmax)
        ax.plot(f_[m], p_[m], color=colors[dEL], linewidth=1.0, label=f"{dEL:+g} mV".replace("-", "−"))
        if peaks.get(dEL) is not None:
            _mark_peak(ax, f_[m], p_[m], peaks[dEL], colors[dEL], lift=1.9, size=26, edge_lw=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_ticks(ax.xaxis)
    ax.set_xlim(fmin, fmax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Spontaneous LFP PSD (a.u.)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="v", linestyle="", markersize=5, color="0.55",
                          markeredgecolor="black", markeredgewidth=0.5))
    labels.append("SNN resonance peak (c)")
    ax.legend(handles, labels, loc="lower left", frameon=False, fontsize=6.3,
              handlelength=1.3, borderaxespad=0.3)


def draw_panel_f_axis(ax, param, title, values, reverse, unit, fmt, sweep_values, peaks,
                      show_ylabel=True, show_xlabel=True, fmin=2, fmax=500):
    """One of the four small multiples of Fig. 1f; colours are the palette of the full sweep
    (`sweep_values`), as in Fig. 1d."""
    colors = rocket_colors(sweep_values, reverse=reverse)
    for v in values:
        f_, p_ = spont_spectrum(param_files(param, v))
        m = (f_ >= fmin) & (f_ <= fmax)
        col = colors[min(colors, key=lambda c: abs(c - v))]
        ax.plot(f_[m], p_[m], color=col, linewidth=1.0, label=f"{fmt(v)} {unit}")
        if peaks.get(v) is not None:
            _mark_peak(ax, f_[m], p_[m], peaks[v], col, lift=1.45, size=20, edge_lw=0.45)

    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_ticks(ax.xaxis)
    ax.set_xlim(fmin, fmax)
    ax.set_title(title, loc="left", fontsize=7, pad=3)
    if show_xlabel:
        ax.set_xlabel("Frequency (Hz)")
    if not show_ylabel:
        ax.tick_params(labelleft=False)
    ax.legend(loc="lower left", frameon=False, fontsize=5.4, handlelength=1.2,
              borderaxespad=0.2, labelspacing=0.25)

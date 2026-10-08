"""Simulated panels of Fig. 6c: the adaptation time course b(t), the group z-scored
spectrogram with its colour bar, the post-cue inset, and the pre-/post-cue LFP spectra.

The spectra axes are built to the specification of the experimental panel beside them (Howe
et al. 2017, Fig. 5C, a 790 x 580 px image shown 3.10 in wide, i.e. 0.283 pt per pixel): the
same frame size and rule weight, label and tick sizes, and legend layout, measured off it.
"""
import numpy as np
from matplotlib.patches import ConnectionPatch

from ach_spectral import (B_POST, B_PRE, FMAX, INSET_WINDOW, T_CUE_END, T_CUE_START, TOTAL_MS,
                          WINDOW_MS, driven_raws, group_spectrogram, inset_stats, lfp_inh,
                          spont_psd)

PRECUE_START = 1000.0      # start of the displayed pre-cue period (ms): 1.5 s before cue onset
FMIN_PLOT = 1.0            # Welch removes each segment's mean, so the lowest bins are not shown
PRE_COL, POST_COL = "blue", "red"
CMAP, YELLOW = "jet", "#F5D300"

# spectra panel, matched to the experimental one: axes frame 542 x 407 px inset 210 px / 22 px
# into the image, 4.5 px rule, 36 px tick digits, 30 px legend text
SP_BOX = (0.824, 0.086, 2.127, 1.597)      # x, y inset into the column, then w, h (inches)
SP_LW = 1.27                               # frame and tick rule (pt)
SP_FS_TICK, SP_FS_LABEL, SP_FS_LEG = 14.2, 14.2, 12.0
SP_TICK_PAD, SP_LABEL_PAD = 3.1, 11.6      # tick-label and axis-label gaps (pt)
SP_LEG = ((0.285, 0.873), (0.285, 0.749), (0.79, 0.97))   # legend text x,y rows; rule x span


def b_of_t(dt_ms=1.0):
    """The imposed adaptation b(t) (pA): B_PRE before the cue, a linear ramp during it, B_POST
    after."""
    t = np.arange(0, TOTAL_MS + dt_ms, dt_ms)
    b = np.where(t < T_CUE_START, B_PRE,
                 np.where(t > T_CUE_END, B_POST,
                          B_PRE + (B_POST - B_PRE) * (t - T_CUE_START) / (T_CUE_END - T_CUE_START)))
    return t, b


def t_s(t_ms):
    """Time from cue onset, in seconds."""
    return (np.asarray(t_ms) - T_CUE_START) / 1000.0


def draw_sim_spectrogram(fig, ax_b, ax_spec, ax_inset, ax_cb, fs_label, fs_tick,
                         inset_window=INSET_WINDOW, signal=lfp_inh):
    """b(t), the group spectrogram and the inset, with dashed leader lines from the inset
    window on the spectrogram's time axis to the inset below it."""
    freqs, t_ms, z_mean, ins, _ = group_spectrogram(driven_raws(), inset_window, signal=signal)
    draw_spectrogram_panel(fig, ax_b, ax_spec, ax_cb, freqs, t_ms, z_mean, fs_label, fs_tick)
    draw_inset_panel(ax_inset, freqs, ins, fs_label - 1, fs_tick - 1)
    for x_data, x_frac in ((t_s(inset_window[0]), 0.0), (t_s(inset_window[1]), 1.0)):
        fig.add_artist(ConnectionPatch(xyA=(x_data, 0), coordsA=ax_spec.transData,
                                       xyB=(x_frac, 1), coordsB=ax_inset.transAxes,
                                       color="k", linewidth=0.8, linestyle="--", alpha=0.7))


def draw_spectrogram_panel(fig, ax_b, ax_spec, ax_cb, freqs, t_ms, z_mean, fs_label, fs_tick):
    """b(t) above the group z-scored spectrogram, with its colour bar.  The colour range is
    the 2nd-95th percentile of the displayed values."""
    xlim = (t_s(PRECUE_START), t_s(TOTAL_MS))        # -1.5 to +2.5 s: the 1 s cue is central
    tb, b = b_of_t()
    ax_b.plot(t_s(tb), b, color="0.15", linewidth=1.3)
    ax_b.set_ylabel("$b$ (pA)", fontsize=fs_label)
    ax_b.set_xlim(*xlim)
    ax_b.set_ylim(0, max(B_PRE, B_POST) + 10)
    ax_b.set_yticks([B_POST, B_PRE])
    ax_b.tick_params(labelsize=fs_tick, bottom=False, labelbottom=False)
    for s in ("top", "right", "bottom"):
        ax_b.spines[s].set_visible(False)
    ax_b.axvspan(0.0, 1.0, color=YELLOW, alpha=0.35, lw=0)

    x = t_s(t_ms)
    disp = (x >= xlim[0]) & (x <= xlim[1])
    vmin, vmax = np.percentile(z_mean[:, disp], (2.0, 95.0))
    # pad the grid so the colour reaches 0 Hz and FMAX, and each end window is painted out to
    # its own edge (a window's value is reported at its centre)
    fp = np.r_[0.0, freqs, FMAX]
    zp = np.vstack([z_mean[0:1], z_mean, z_mean[-1:]])
    half = WINDOW_MS / 2000.0
    xp = np.r_[x[0] - half, x, x[-1] + half]
    zp = np.hstack([zp[:, 0:1], zp, zp[:, -1:]])
    mesh = ax_spec.pcolormesh(xp, fp, zp, shading="gouraud", cmap=CMAP, vmin=vmin, vmax=vmax)
    for xv in (0.0, 1.0):
        ax_spec.axvline(xv, color="k", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_spec.axvspan(0.0, 1.0, color=YELLOW, alpha=0.12, lw=0)
    ax_spec.set_xlim(*xlim)
    ax_spec.set_ylim(0, FMAX); ax_spec.set_yticks([0, 20, 40, 60, 80, 100])
    ax_spec.set_xlabel("Time from cue onset (s)", fontsize=fs_label)
    ax_spec.set_ylabel("Frequency (Hz)", fontsize=fs_label)
    ax_spec.tick_params(labelsize=fs_tick)
    cb = fig.colorbar(mesh, cax=ax_cb, extend="both")
    cb.set_label("z-score", fontsize=fs_label - 1)
    cb.ax.tick_params(labelsize=fs_tick - 1)


def draw_inset_panel(ax_inset, freqs, ins, fs_label, fs_tick, edge=True):
    """The z-scored spectrogram averaged over the inset window: mean over runs with its 95%
    confidence interval.  With `edge`, a dashed line at FMAX meets the leader line from above."""
    mean_z, ci95 = inset_stats(ins)
    ax_inset.axhline(0.0, color="0.6", linewidth=0.8, linestyle=":")
    ax_inset.plot(freqs, mean_z, color=PRE_COL, linewidth=1.4)
    ax_inset.fill_between(freqs, mean_z - ci95, mean_z + ci95, color=PRE_COL, alpha=0.25, lw=0)
    if edge:
        ax_inset.axvline(FMAX, color="k", linewidth=1.0, linestyle="--", alpha=0.85, zorder=5)
    ax_inset.set_xlim(0, FMAX)
    ax_inset.set_xlabel("Frequency (Hz)", fontsize=fs_label)
    ax_inset.set_ylabel("Z-scored power", fontsize=fs_label)
    ax_inset.tick_params(labelsize=fs_tick)
    for s in ("top", "right"):
        ax_inset.spines[s].set_visible(False)


def draw_sim_spectra(ax, ymin=5.9, ymax=10.0, signal=lfp_inh):
    """Spontaneous LFP spectra (log10 power) at b = 75 pA (pre-cue) and b = 10 pA (post-cue).
    The top of the y range leaves room for the inline legend above the curves."""
    for b, col in ((B_PRE, PRE_COL), (B_POST, POST_COL)):
        f, p = spont_psd(b, signal=signal)
        m = (f >= FMIN_PLOT) & (f <= FMAX)
        ax.plot(f[m], np.log10(p[m]), color=col, linewidth=1.9)
    ax.set_xlim(0, FMAX); ax.set_ylim(ymin, ymax)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks(list(range(int(np.ceil(ymin)), int(np.floor(ymax)) + 1)))
    ax.set_xlabel("Frequency (Hz)", fontsize=SP_FS_LABEL, labelpad=SP_LABEL_PAD)
    ax.set_ylabel("log Power (a.u.)", fontsize=SP_FS_LABEL, labelpad=SP_LABEL_PAD * 0.45)
    ax.tick_params(labelsize=SP_FS_TICK, width=SP_LW, length=4.5, pad=SP_TICK_PAD)
    for s in ax.spines.values():                     # a full box, as in the experimental panel
        s.set_visible(True); s.set_linewidth(SP_LW)
    (xt, y_pre), (_, y_post), (x0, x1) = SP_LEG
    for y, txt, col in ((y_pre, "PRE-CUE", PRE_COL), (y_post, "POST-CUE", POST_COL)):
        ax.text(xt, y, txt, transform=ax.transAxes, color=col, fontsize=SP_FS_LEG,
                va="center", ha="left")
        ax.plot([x0, x1], [y, y], transform=ax.transAxes, color=col, linewidth=2.3,
                solid_capstyle="butt", clip_on=False)

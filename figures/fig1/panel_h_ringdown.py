"""Fig. 1h: LFP proxy around the offset of the drive, for the OD (Delta_EL = -10 mV) and UD (+10
mV) networks of Fig. 1g (single seed, 300)."""
import numpy as np

from asyncresp.io import load_pickle
from asyncresp.paths import DRIVEN
from ringdown import analyze_ringdown, ringdown_curve

NETS = {"OD": -10.0, "UD": 10.0}     # delta_EL (mV)
DRIVE_HZ = 55.0
SEED = 300
CYCLES_BEFORE, CYCLES_AFTER = 2.5, 4.5


def load_run(delta_EL):
    tag = f"{delta_EL:+.0f}".replace("+", "p").replace("-", "m")
    return load_pickle(DRIVEN / "od_ud_55hz" / f"h_{tag}_f{DRIVE_HZ:g}_{SEED}.pkl")


def draw_ringdown(ax, run, color, title, ylabel=True, n_before=CYCLES_BEFORE, n_after=CYCLES_AFTER):
    """LFP trace and the fitted curves, from n_before cycles before to n_after cycles after the
    offset."""
    t = np.asarray(run["t_state"], dtype=float)
    x = np.asarray(run["lfp_inh"], dtype=float)
    t_off, period = run["oscillation_offset"], run["period_ms"]
    fit = analyze_ringdown(t, x, period, run["oscillation_onset"], t_off, run["total_ms"])
    y_fit = ringdown_curve(t, fit, t_off)

    zoom_lo, zoom_hi = t_off - n_before * period, min(run["total_ms"], t_off + n_after * period)
    m = (t >= zoom_lo) & (t <= zoom_hi)
    ax.plot((t[m] - t_off) / period, x[m] / 1e6, color=color, alpha=0.6, linewidth=0.8)

    pre = (t >= zoom_lo) & (t <= t_off)
    post = (t >= t_off) & (t <= zoom_hi)
    ax.plot((t[pre] - t_off) / period, y_fit[pre] / 1e6, color=color, linewidth=0.9)
    ax.plot((t[post] - t_off) / period, y_fit[post] / 1e6, color=color, linewidth=0.9,
            linestyle=(0, (3, 1.5)))
    ax.axvline(0, color="0.3", linewidth=0.6, linestyle=":")

    ax.set_title(title, fontsize=7, color=color, pad=3)
    ax.set_xlim(-n_before, n_after)
    if ylabel:
        ax.set_ylabel(r"LFP (a.u., $\times 10^6$)")
    else:
        ax.tick_params(labelleft=False)


def draw_panel_h(ax_od, ax_ud, colors):
    """OD and UD networks on a shared y-axis; colours from the Delta_EL palette of Fig. 1c."""
    od, ud = NETS["OD"], NETS["UD"]
    draw_ringdown(ax_od, load_run(od), colors[od], rf"OD, $\Delta_{{E_L}}={od:+g}$ mV")
    draw_ringdown(ax_ud, load_run(ud), colors[ud], rf"UD, $\Delta_{{E_L}}={ud:+g}$ mV", ylabel=False)
    ax_od.set_ylim(0, None)

"""Fig. 1g: population rates of an OD (Delta_EL = -10 mV) and a UD (Delta_EL = +10 mV) network,
before and during the same sinusoidal drive."""
import numpy as np

from asyncresp.io import load_pickle
from asyncresp.paths import DRIVEN
from asyncresp.style import DRIVE_COL, E_COL, I_COL, INK, INK2, WHITE
from common import rebin, rocket_colors

NETS = {"OD": -10.0, "UD": 10.0}     # delta_EL (mV)
DRIVE_HZ = 55.0
SEEDS = list(range(12, 22))
WINDOW = (1940.0, 2160.0)            # 60 ms of spontaneous activity, then ~9 driven cycles
RATE_BIN_MS = 1.0
DRIVE_BAND = (-0.22, -0.06)          # where the drive is drawn, as a fraction of the rate axis


def run_path(delta_EL, seed):
    tag = f"{delta_EL:+.0f}".replace("+", "p").replace("-", "m")
    return DRIVEN / "od_ud_55hz" / f"g_{tag}_f{DRIVE_HZ:g}_{seed}.pkl"


def mean_traces(delta_EL, seeds=SEEDS):
    """Seed-averaged E and I rates, the single runs, and the fitted baseline of each average."""
    sims = [load_pickle(run_path(delta_EL, s)) for s in seeds]
    o = dict(sims[0])
    t = rebin(o["t_ms"], o["fr_exc"], RATE_BIN_MS)[0]
    for key, off_k in [("fr_exc", "offset_exc"), ("fr_inh", "offset_inh")]:
        ys = np.array([rebin(s["t_ms"], s[key], RATE_BIN_MS)[1] for s in sims])
        y = ys.mean(axis=0)
        o[key + "_all"], o[key + "_mean"] = ys, y
        # least-squares fit of a*cos(wt) + b*sin(wt) + offset at the drive frequency
        m = t >= o["fit_cutoff_ms"]
        w = 2 * np.pi / o["period"]
        X = np.column_stack([np.cos(w * t[m]), np.sin(w * t[m]), np.ones(m.sum())])
        c, *_ = np.linalg.lstsq(X, y[m], rcond=None)
        o[off_k] = float(c[2])
    o["t_binned"] = t
    return o


def rate_max(sims, window=WINDOW):
    """Largest averaged rate within the window, over all networks (sets the shared y-axis)."""
    hi = 0.0
    for o in sims:
        for key in ("fr_exc_mean", "fr_inh_mean"):
            m = (o["t_binned"] >= window[0]) & (o["t_binned"] <= window[1])
            hi = max(hi, o[key][m].max())
    return hi


def draw_one(ax, o, title, title_color, top, window=WINDOW, ylabel=True, scalebar_ms=50,
             legend=False):
    onset = o["oscillation_onset"]
    ax.set_xlim(*window)
    ax.set_ylim(DRIVE_BAND[0] * top, top)

    for key, col, lab, off_k in [("fr_inh", I_COL, "I", "offset_inh"),
                                 ("fr_exc", E_COL, "E", "offset_exc")]:
        t, y = o["t_binned"], o[key + "_mean"]
        m = (t >= window[0]) & (t <= window[1])
        for yi in o[key + "_all"]:                       # the single runs, behind the average
            ax.plot(t[m], yi[m], color=col, linewidth=0.35, alpha=0.20, zorder=1, rasterized=True)
        ax.plot(t[m], y[m], color=col, linewidth=0.85, alpha=1.0, zorder=3)
        off = o[off_k]
        ax.plot([window[0] + 8, window[1]], [off] * 2, color=INK2, linewidth=0.4,
                linestyle=(0, (1, 2.2)), zorder=1)
        ax.text(window[0] + 6, off + (0.055 if lab == "I" else 0.02) * top, lab, fontsize=8,
                color=col, weight="bold", va="bottom", ha="left", path_effects=WHITE)

    # the same drive under both panels
    per = o["period"]
    t_d = np.linspace(window[0], window[1], 2500)
    d = np.where(t_d >= onset, np.sin(2 * np.pi * (t_d - onset) / per), 0.0)
    y0, y1 = DRIVE_BAND[0] * top, DRIVE_BAND[1] * top
    ax.plot(t_d, 0.5 * (y0 + y1) + 0.5 * (y1 - y0) * d, color=DRIVE_COL, linewidth=0.7,
            linestyle=(0, (1.4, 1.2)), zorder=3)
    ax.axvline(onset, color="0.45", linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
    if legend:
        ax.text(window[1] - 6, y1 + 0.005 * top, r"$\nu_{drive}$", fontsize=6.5, color=DRIVE_COL,
                va="bottom", ha="right")

    ax.set_title(title, fontsize=7.5, color=title_color, pad=3)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([])
    if ylabel:
        ax.set_ylabel("Firing rate (Hz)")
        ax.spines["left"].set_bounds(0, top)
        ax.set_yticks([0, 10, 20, 30, 40])
    else:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False, labelleft=False)   # shared y: never set_yticks here
        x0 = window[1] - scalebar_ms - 6                  # time scale bar
        ax.plot([x0, x0 + scalebar_ms], [y0 * 1.05] * 2, color=INK, linewidth=1.4,
                solid_capstyle="butt", clip_on=False, zorder=5)
        ax.text(x0 + scalebar_ms / 2, y0 * 1.15, f"{scalebar_ms:g} ms", fontsize=6.8,
                ha="center", va="top", color=INK)


def draw_panel_g(fig, spec, window=WINDOW, wspace=0.10):
    """OD (left) and UD (right) on one shared rate axis; colours as in Fig. 1c."""
    sims = {role: mean_traces(d) for role, d in NETS.items()}
    top = rate_max(sims.values(), window) * 1.06
    colors = rocket_colors([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0])
    gs = spec.subgridspec(1, 2, wspace=wspace)
    axes = []
    for k, role in enumerate(NETS):
        o = sims[role]
        ax = fig.add_subplot(gs[0, k], sharey=axes[0] if axes else None)
        draw_one(ax, o, rf"{role}, $\Delta_{{E_L}} = {o['delta_EL']:+g}$ mV".replace("-", "−"),
                 colors[o["delta_EL"]], top, window=window, ylabel=(k == 0), legend=(k == 1))
        axes.append(ax)
    return axes

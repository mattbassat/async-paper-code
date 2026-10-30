"""Fig. 3c: example responses to the slow (0.5 Hz) input for Delta_EL = -15, -5 and +5 mV."""
import numpy as np
import seaborn as sns

from asyncresp.analysis import rebin
from asyncresp.io import load_pickle
from asyncresp.paths import DRIVEN
from asyncresp.style import E_COL, I_COL, INK, DRIVE_COL, WHITE

EXAMPLES_DIR = DRIVEN / "slow_input_examples"
SEEDS = list(range(12, 22))
ONSET = 2000.0                                         # start of the sinusoidal drive (ms)
PERIOD = 2000.0                                        # drive period (ms), 0.5 Hz
WINDOW = (500.0, ONSET + 2 * PERIOD)                   # 1.5 s spontaneous + 2 cycles
RATE_BIN_MS = 20.0                                     # re-binning of the 1 ms population rates
DRIVE_BAND = (-0.26, -0.08)                            # drive trace, as a fraction of the axis
Y_FLOOR = -0.40                                        # room under it for the time scale bar
# (file tag, Delta_EL in mV, regime label, symbol)
EXAMPLES = [("m15", -15.0, "OD", r"$\ast$"),
            ("m5", -5.0, None, r"$\dagger$"),
            ("p5", 5.0, "UD", r"$\ddagger$")]
DEL_GRID = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]  # Delta_EL colour scale shared with Fig. 2


def example_symbols():
    """{Delta_EL: symbol} of the example networks, for marking them in Fig. 3a."""
    return {d: sym for _, d, _, sym in EXAMPLES}


def mean_traces(tag):
    """Re-binned E and I population rates of every seed of one network, and their average."""
    sims = [load_pickle(EXAMPLES_DIR / f"fig3_example_{tag}_{seed}.pkl") for seed in SEEDS]
    o = dict(sims[0])
    o["t_binned"] = rebin(o["t_ms"], o["fr_exc"], RATE_BIN_MS)[0]
    for key in ("fr_exc", "fr_inh"):
        ys = np.array([rebin(s["t_ms"], s[key], RATE_BIN_MS)[1] for s in sims])
        o[key + "_all"], o[key + "_mean"] = ys, ys.mean(axis=0)
    return o


def draw_one(ax, o, title, colour, symbol, top, ylabel, drive_label, scalebar_ms=None):
    """Rates of one network: individual seeds, seed average, drive, onset line and labels."""
    window = WINDOW
    ax.set_xlim(*window)
    ax.set_ylim(Y_FLOOR * top, top)
    for key, col, lab in (("fr_inh", I_COL, "I"), ("fr_exc", E_COL, "E")):
        t, y = o["t_binned"], o[key + "_mean"]
        m = (t >= window[0]) & (t <= window[1])
        for yi in o[key + "_all"]:
            ax.plot(t[m], yi[m], color=col, linewidth=0.35, alpha=0.20, zorder=1,
                    rasterized=True)
        ax.plot(t[m], y[m], color=col, linewidth=0.85, alpha=1.0, zorder=3)
        ax.text(window[0] + 0.012 * (window[1] - window[0]), y[m][:8].mean() + 0.035 * top, lab,
                fontsize=7.5, color=col, weight="bold", va="bottom", ha="left",
                path_effects=WHITE, zorder=5)

    t_d = np.linspace(*window, 2500)
    d = np.where(t_d >= ONSET, np.sin(2 * np.pi * (t_d - ONSET) / PERIOD), 0.0)
    y0, y1 = DRIVE_BAND[0] * top, DRIVE_BAND[1] * top
    ax.plot(t_d, 0.5 * (y0 + y1) + 0.5 * (y1 - y0) * d, color=DRIVE_COL, linewidth=0.7,
            linestyle=(0, (1.4, 1.2)), zorder=3)
    ax.axvline(ONSET, color="0.45", linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
    if drive_label:
        ax.text(window[1] - 20, y1 + 0.01 * top, r"$\nu_{drive}$", fontsize=6.2,
                color=DRIVE_COL, va="bottom", ha="right")

    ax.set_title(title, fontsize=7.2, color=colour, pad=3)
    ax.text(0.985, 0.965, symbol, transform=ax.transAxes, fontsize=10, color=INK, ha="right",
            va="top", path_effects=WHITE, zorder=6)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([])
    if ylabel:
        ax.set_ylabel("Firing rate (Hz)")
        ax.spines["left"].set_bounds(0, top)
        ax.set_yticks([y for y in range(0, int(top) + 1, 5)])
    else:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False, labelleft=False)
    if scalebar_ms:
        x0 = window[1] - scalebar_ms - 0.02 * (window[1] - window[0])
        yb = (Y_FLOOR + 0.075) * top
        ax.plot([x0, x0 + scalebar_ms], [yb] * 2, color=INK, linewidth=1.4,
                solid_capstyle="butt", zorder=5)
        ax.text(x0 + scalebar_ms / 2, yb - 0.015 * top, f"{scalebar_ms / 1000:g} s",
                fontsize=6.4, ha="center", va="top", color=INK)


def draw_examples(axes):
    """Panel c on three axes, one per example network, with a common rate axis."""
    sims = {tag: mean_traces(tag) for tag, _, _, _ in EXAMPLES}
    top = 0.0
    for o in sims.values():
        m = (o["t_binned"] >= WINDOW[0]) & (o["t_binned"] <= WINDOW[1])
        top = max(top, o["fr_inh_mean"][m].max(), o["fr_exc_mean"][m].max())
    top *= 1.08
    colors = dict(zip(DEL_GRID, sns.color_palette("rocket", n_colors=len(DEL_GRID))))
    last = len(EXAMPLES) - 1
    for k, (ax, (tag, _, regime, sym)) in enumerate(zip(axes, EXAMPLES)):
        o = sims[tag]
        d = o["delta_EL"]
        ds = f"{d:+g}" if d else "0"
        title = (rf"{regime}, $\Delta_{{E_L}} = {ds}$ mV" if regime
                 else rf"$\Delta_{{E_L}} = {ds}$ mV")
        draw_one(ax, o, title.replace("-", "−"), colors[d], sym, top, ylabel=(k == 0),
                 drive_label=(k == last), scalebar_ms=1000 if k == last else None)

"""Fig. 1a: network schematic with a recording electrode, and an example of the evoked activity:
the sinusoidal drive, a spike raster, the I and E population rates with their fitted sinusoids,
and the LFP proxy.

The schematic shows an excitatory (8000) and an inhibitory (2000) population of randomly
connected AdEx neurons, the four recurrent projections, and the external Poisson population
(rate nu_drive) that drives both.  The traces come from one stored simulation (nu_drive = 4 Hz,
b = 0 pA, EL_exc = -70 mV, EL_inh = -60 mV, A = 1 Hz, f = 2 Hz, drive onset 2 s, seed 12).  The
LFP proxy is the sum over 500 excitatory cells of |G_I (E_I - V_ref)|, V_ref = -60 mV, sampled
every 0.5 ms.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.patches import ConnectionPatch, FancyArrowPatch

from asyncresp.io import load_pickle
from asyncresp.paths import DRIVEN
from asyncresp.style import DRIVE_COL, E_COL, EXT_COL, I_COL, INK, INK2, WHITE
from common import rebin

ELECTRODE = Path(__file__).resolve().parent / "assets" / "electrode_schematic.png"

WINDOW = (1000.0, 5000.0)                # 1 s of spontaneous activity, then the driven cycles
RASTER_FRAC = dict(exc=0.10, inh=0.10)   # fraction of spikes drawn
RATE_BIN_MS = 5.0                        # re-binning of the 1 ms population rates
LFP_SKIP_MS = 100.0                      # conductances start from zero: skip the initial ramp
STRIP_UNITS = dict(drive=0.75, raster=1.45, lfp=0.75, exc_rate=0.87)   # see rate_split()
AMP_CREST = 3                            # the response-amplitude caliper sits in this crest of the fit
RNG_SEED = 7                             # layout of the schematic only
ONSET_MS = 2000.0                        # start of the sinusoidal drive
CALIPER_MS, CALIPER_LW = 7.0, 0.9        # one style for every double-headed arrow of the panel


def load_example():
    """The stored example simulation shown in this panel."""
    return load_pickle(DRIVEN / "fig1_example_sim.pkl")


# ----------------------------------------------------------------- network schematic
def _blob(n, centre, radius, rng, wobble=(0.14, 0.09)):
    """n points inside a slightly irregular disc."""
    th = rng.uniform(0, 2 * np.pi, n)
    shape = 1 + wobble[0] * np.sin(3 * th + 0.7) + wobble[1] * np.sin(5 * th + 2.1)
    r = radius * shape * np.sqrt(rng.uniform(0, 1, n))
    return np.column_stack([centre[0] + r * np.cos(th), centre[1] + r * np.sin(th)])


def _cloud(ax, pts, colour, n_edges, rng, node_size, edge_lw=0.22, edge_alpha=0.22):
    """A population drawn as nodes with random connections between them."""
    i = rng.integers(0, len(pts), n_edges)
    j = rng.integers(0, len(pts), n_edges)
    for a, b in zip(pts[i], pts[j]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=colour, linewidth=edge_lw, alpha=edge_alpha, zorder=1)
    ax.scatter(pts[:, 0], pts[:, 1], s=node_size, color=colour, zorder=2, linewidths=0)


def _arrow(ax, p0, p1, colour, rad, lw=1.3, ms=8.5):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", color=colour,
                                 arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
                                 shrinkA=2, shrinkB=2, zorder=4, capstyle="round"))


def _electrode_rgba(path=ELECTRODE):
    """The electrode drawing with its white background made transparent."""
    img = plt.imread(path)
    if img.ndim == 2:
        img = np.dstack([img] * 3)
    if img.dtype != np.float32 and img.max() > 1:
        img = img / 255.0
    if img.shape[2] == 3:
        alpha = 1.0 - img[..., :3].min(axis=2)          # white -> transparent, black -> opaque
        img = np.dstack([img[..., :3], alpha])
    return img


def draw_electrode(ax, tip=(-0.72, 0.12), angle_deg=135.0, length=2.45):
    """Electrode with its tip on the network; returns the far end of the shank."""
    arr = _electrode_rgba()
    h, w = arr.shape[:2]
    width = length * w / h
    im = ax.imshow(arr, extent=[tip[0] - width / 2, tip[0] + width / 2, tip[1], tip[1] + length],
                   interpolation="bicubic", zorder=6, clip_on=False)
    im.set_transform(mtransforms.Affine2D().rotate_deg_around(tip[0], tip[1], angle_deg) + ax.transData)
    th = np.deg2rad(angle_deg + 90.0)                    # shank direction, tip -> back
    back = (tip[0] + length * np.cos(th), tip[1] + length * np.sin(th))
    ax.text(back[0] + 1.00, back[1] + 0.32, "LFP\nelectrode", fontsize=6.6, color=INK2,
            ha="center", va="top")
    return back


def draw_network_schematic(ax, ylim=(-3.3, 2.9)):
    """E and I populations, their projections, the external drive and the electrode."""
    rng = np.random.default_rng(RNG_SEED)
    e_pts = _blob(340, (0.0, 1.15), 1.05, rng)
    i_pts = _blob(150, (0.05, -1.15), 0.62, rng)
    _cloud(ax, e_pts, E_COL, 1000, rng, node_size=5.0)
    _cloud(ax, i_pts, I_COL, 420, rng, node_size=3.6)

    # the four projections: one arrow each, endpoints marked by slightly larger neurons
    src_ei, dst_ei = np.array([0.35, 0.25]), np.array([0.38, -0.60])       # E -> I
    src_ie, dst_ie = np.array([-0.30, -0.62]), np.array([-0.45, 0.30])     # I -> E
    src_ee, dst_ee = np.array([-0.70, 1.95]), np.array([0.60, 2.00])       # E -> E (recurrent)
    src_ii, dst_ii = np.array([0.45, -1.62]), np.array([-0.40, -1.63])     # I -> I (recurrent)
    _arrow(ax, src_ei, dst_ei, E_COL, rad=-0.45)
    _arrow(ax, src_ie, dst_ie, I_COL, rad=-0.45)
    _arrow(ax, src_ee, dst_ee, E_COL, rad=-0.85)
    _arrow(ax, src_ii, dst_ii, I_COL, rad=-0.95)
    for p, c in [(src_ei, E_COL), (dst_ei, I_COL), (src_ie, I_COL), (dst_ie, E_COL),
                 (src_ee, E_COL), (dst_ee, E_COL), (src_ii, I_COL), (dst_ii, I_COL)]:
        ax.scatter([p[0]], [p[1]], s=9, color=c, zorder=5, linewidths=0)

    # external population: a small cluster of unconnected Poisson units driving both populations
    ext_pts = _blob(55, (2.15, 0.12), 0.40, rng, wobble=(0.10, 0.07))
    ax.scatter(ext_pts[:, 0], ext_pts[:, 1], s=3.6, color=EXT_COL, zorder=2, linewidths=0)
    ax.text(2.15, -0.45, r"$\nu_{drive}$", fontsize=8, va="top", ha="center", color=INK)
    _arrow(ax, np.array([1.78, 0.36]), np.array([0.80, 0.90]), INK, rad=0.28, lw=1.1, ms=7.5)
    _arrow(ax, np.array([1.80, -0.14]), np.array([0.52, -1.02]), INK, rad=-0.28, lw=1.1, ms=7.5)

    back = draw_electrode(ax)

    ax.text(1.45, 2.05, "E\n8000", fontsize=7.5, color=E_COL, va="center", ha="center", weight="bold")
    ax.text(1.62, -1.45, "I\n2000", fontsize=7.5, color=I_COL, va="center", ha="center", weight="bold")
    ax.set_xlim(-3.0, 3.1)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return back


# ----------------------------------------------------------------- example simulation
def _sine(t, amp, phase_rad, offset, period_ms):
    """Fitted response: r(t) = amp * cos(2 pi t / period + phase) + offset."""
    return offset + amp * np.cos(2 * np.pi * t / period_ms + phase_rad)


def _caliper(ax, x, y0, y1, label=None, dx_label=-40, fs=7.5, ms=CALIPER_MS):
    """Vertical double arrow with a white halo, so that it reads over the traces."""
    a = FancyArrowPatch((x, y0), (x, y1), arrowstyle="<->", mutation_scale=ms, color=INK,
                        linewidth=CALIPER_LW, shrinkA=0, shrinkB=0, zorder=6)
    a.set_path_effects(WHITE)
    ax.add_patch(a)
    if label:
        ax.text(x + dx_label, 0.5 * (y0 + y1), label, fontsize=fs, color=INK, zorder=6,
                ha="right" if dx_label < 0 else "left", va="center", path_effects=WHITE)


def _strip_axes(ax, window):
    ax.set_xlim(*window)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_drive(ax, o, window=WINDOW):
    """Afferent rate nu_drive(t), with its amplitude A and period 1/f marked."""
    t, y = o["t_drive_ms"], o["drive_hz"]
    m = (t >= window[0]) & (t <= window[1])
    base, per, A = o["rate"], o["period"], o["A"]
    onset = o["oscillation_onset"]
    ax.axhline(base, color=INK2, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    ax.fill_between(t[m], base, y[m], color=DRIVE_COL, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(t[m], y[m], color=DRIVE_COL, linewidth=1.1, zorder=3)

    # amplitude: a caliper in the baseline stretch, just before the first cycle
    x_a = onset - 210
    ax.plot([x_a, onset + per / 4], [base + A] * 2, color=INK2, linewidth=0.5,
            linestyle=(0, (1, 1.6)), zorder=3)
    _caliper(ax, x_a, base, base + A, "$A$", dx_label=-55)
    # period
    t_pk = onset + per / 4
    y_per = base + A * 1.6
    ax.annotate("", xy=(t_pk, y_per), xytext=(t_pk + per, y_per), zorder=4,
                arrowprops=dict(arrowstyle="<->", color=INK, lw=CALIPER_LW,
                                mutation_scale=CALIPER_MS, shrinkA=0, shrinkB=0))
    ax.text(t_pk + per / 2, y_per + 0.08, "$1/f$", fontsize=7.5, ha="center", va="bottom", color=INK)

    ax.set_xlim(*window)
    ax.set_ylim(base - A * 1.35, base + A * 2.2)
    ax.set_ylabel(r"$\nu_{drive}$ (Hz)", fontsize=7, color=INK2, rotation=0, ha="right",
                  va="center", labelpad=6)
    _strip_axes(ax, window)


def draw_raster(ax, o, window=WINDOW, frac=RASTER_FRAC, rng_seed=1):
    """Spike raster of a random fraction of the spikes (E: neurons 0-7999, I: 8000-9999)."""
    rng = np.random.default_rng(rng_seed)
    n_exc, n_inh = 8000, 2000
    for key, key_t, key_i, col, y0 in [("inh", "spk_inh_t", "spk_inh_i", I_COL, n_exc),
                                       ("exc", "spk_exc_t", "spk_exc_i", E_COL, 0)]:
        t, idx = o[key_t], o[key_i]
        m = (t >= window[0]) & (t <= window[1])
        t, idx = t[m], idx[m]
        keep = rng.random(len(t)) < frac[key]
        ax.scatter(t[keep], idx[keep] + y0, s=0.3, color=col, alpha=0.5, linewidths=0, rasterized=True)
    ax.text(window[0] + 40, n_exc + n_inh * 0.55, "I", fontsize=8, color=I_COL, weight="bold",
            va="center", path_effects=WHITE)
    ax.text(window[0] + 40, n_exc * 0.5, "E", fontsize=8, color=E_COL, weight="bold",
            va="center", path_effects=WHITE)
    ax.set_xlim(*window)
    ax.set_ylim(0, n_exc + n_inh)
    ax.set_ylabel("neuron index", fontsize=7, color=INK2, rotation=0, ha="right",
                  va="center", labelpad=6)
    _strip_axes(ax, window)


def rate_traces(o, window=WINDOW):
    """I and E population rates re-binned to RATE_BIN_MS, within the window."""
    t, fe = rebin(o["t_ms"], o["fr_exc"], RATE_BIN_MS)
    _, fi = rebin(o["t_ms"], o["fr_inh"], RATE_BIN_MS)
    m = (t >= max(window[0], 25.0)) & (t <= window[1])
    return t[m], fi[m], fe[m]


def rate_limits(o, window=WINDOW, head=(0.16, 0.14)):
    """y-limits of the I and E rate axes (with headroom for the labels)."""
    _, fi, fe = rate_traces(o, window)
    out = []
    for tr, h in zip((fi, fe), head):
        lo, hi = tr.min(), tr.max()
        out.append((lo - 0.10 * (hi - lo), hi + h * (hi - lo)))
    return out


def rate_split(o, window=WINDOW):
    """Height ratios of the I and E rate axes, and the height of the pair in strip units.
    The heights are proportional to the y-ranges, so both axes share one Hz-per-inch scale."""
    (i0, i1), (e0, e1) = rate_limits(o, window)
    span_i, span_e = i1 - i0, e1 - e0
    return [span_i, span_e], STRIP_UNITS["exc_rate"] * (span_i + span_e) / span_e


def crest_time(o, pop, k=AMP_CREST):
    """Time of the k-th crest of the fitted response of `pop` after the onset of the drive."""
    per, onset = o["period"], o["oscillation_onset"]
    ph = o[f"phase_{pop}_raw"]
    n = np.ceil((onset / per) + ph / (2 * np.pi))          # first crest at or after the onset
    return per * (n - ph / (2 * np.pi)) + (k - 1) * per


def draw_rates(axes, o, window=WINDOW):
    """I and E population rates on a common Hz-per-inch scale, with the fitted sinusoids.

    The evoked response amplitude is the amplitude of the fitted sinusoid (fitted offset to
    crest) and is marked by a caliper of that length in the third crest of each fit."""
    ax_i, ax_e = axes
    t, fi, fe = rate_traces(o, window)
    onset = o["oscillation_onset"]
    t_fit = np.linspace(onset, window[1], 900)
    lims = rate_limits(o, window)

    specs = [(ax_i, fi, I_COL, o["amp_inh"], o["offset_inh"], o["phase_inh_raw"], "I", "inh"),
             (ax_e, fe, E_COL, o["amp_exc"], o["offset_exc"], o["phase_exc_raw"], "E", "exc")]
    marks = {}
    for (ax, tr, col, amp, off, ph, lab, pop), ylim in zip(specs, lims):
        ax.plot(t, tr, color=col, linewidth=0.7, alpha=0.7, zorder=2)
        ax.plot(t_fit, _sine(t_fit, amp, ph, off, o["period"]), color=INK, linewidth=0.9, zorder=3)
        ax.set_xlim(*window)
        ax.set_ylim(*ylim)
        x_cal = crest_time(o, pop)
        ax.plot([onset, window[1]], [off] * 2, color=INK2, linewidth=0.4,
                linestyle=(0, (1, 2.2)), zorder=1)        # baseline of the fit
        # arrow heads scaled to the caliper's length, so that a short caliper keeps a visible body
        y0d = ax.transData.transform((x_cal, off))[1]
        y1d = ax.transData.transform((x_cal, off + amp))[1]
        L = abs(y1d - y0d) * 72.0 / ax.figure.dpi
        _caliper(ax, x_cal, off, off + amp, ms=float(np.clip(0.40 * L, 3.0, CALIPER_MS)))
        # the label arrow points at the near end of the caliper, so its head stays clear of it
        marks[lab] = (ax, x_cal, off if lab == "I" else off + amp)
        ax.text(window[1] - 40, tr.max(), lab, fontsize=8, color=col, weight="bold",
                va="top", ha="right", path_effects=WHITE)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    fig = ax_i.figure
    p_i, p_e = ax_i.get_position(), ax_e.get_position()
    ax_i.set_ylabel("firing rate\n(Hz)", fontsize=7, color=INK2, rotation=0, ha="right",
                    va="center", labelpad=6)             # centred on the pair, not on the I axes
    y_mid = 0.5 * (p_i.y1 + p_e.y0)
    ax_i.yaxis.set_label_coords(-0.019, (y_mid - p_i.y0) / (p_i.y1 - p_i.y0))

    # one "response amplitude" label, with an arrow to each caliper
    x_lab = min(x for _, x, _ in marks.values()) - 1.6 * o["period"]
    fx = p_i.x0 + (x_lab - window[0]) / (window[1] - window[0]) * (p_i.x1 - p_i.x0)
    fy = 0.5 * (p_i.y0 + p_e.y1)
    txt = fig.text(fx, fy, "response amplitude", fontsize=7, ha="center", va="center",
                   color=INK, path_effects=WHITE, zorder=8)
    bb = txt.get_window_extent(renderer=fig.canvas.get_renderer())
    bb = bb.transformed(fig.transFigure.inverted())    # arrows start on the edge of the label
    for lab, y0 in (("I", bb.y1), ("E", bb.y0)):
        ax, x_cal, y_cal = marks[lab]
        cp = ConnectionPatch(xyA=(bb.x1, y0), coordsA=fig.transFigure,
                             xyB=(x_cal, y_cal), coordsB=ax.transData,
                             arrowstyle="-|>", mutation_scale=8, linewidth=0.8,
                             color=INK, shrinkA=1, shrinkB=4, zorder=7)
        fig.add_artist(cp)


def draw_lfp(ax, o, window=WINDOW, scalebar_ms=500):
    """LFP proxy, with a time scale bar."""
    t, x = o["t_lfp_ms"], o["lfp_inh"]
    m = (t >= max(window[0], LFP_SKIP_MS)) & (t <= window[1])
    x = x / 1e3
    ax.plot(t[m], x[m], color=INK, linewidth=0.5, zorder=2, rasterized=True)
    lo, hi = x[m].min(), x[m].max()
    ax.set_xlim(*window)
    ax.set_ylim(lo - 0.10 * (hi - lo), hi + 0.34 * (hi - lo))
    ax.set_ylabel("LFP\n(a.u.)", fontsize=7, color=INK2, rotation=0, ha="right", va="center", labelpad=6)
    _strip_axes(ax, window)

    y0 = lo - 0.10 * (hi - lo)
    x0 = window[1] - scalebar_ms - 60
    ax.plot([x0, x0 + scalebar_ms], [y0] * 2, color=INK, linewidth=1.4, solid_capstyle="butt",
            clip_on=False, zorder=5)
    ax.text(x0 + scalebar_ms / 2, y0 - 0.04 * (hi - lo), f"{scalebar_ms:g} ms", fontsize=6.8,
            ha="center", va="top", color=INK)


# ----------------------------------------------------------------- panel
def draw_panel_a(fig, spec):
    """The whole panel in one gridspec slot: schematic on the left, the four strips on the right."""
    o = load_example()
    split, rate_units = rate_split(o)
    strip_h = [STRIP_UNITS["drive"], STRIP_UNITS["raster"], rate_units, STRIP_UNITS["lfp"]]
    gs = spec.subgridspec(1, 2, width_ratios=[1.0, 1.225], wspace=0.14)
    ax_net = fig.add_subplot(gs[0, 0])
    back = draw_network_schematic(ax_net)
    strips = gs[0, 1].subgridspec(4, 1, height_ratios=strip_h, hspace=0.16)
    ax_drive = fig.add_subplot(strips[0])
    draw_drive(ax_drive, o)
    ax_rast = fig.add_subplot(strips[1])
    draw_raster(ax_rast, o)
    rate_gs = strips[2].subgridspec(2, 1, height_ratios=split, hspace=0.06)
    draw_rates((fig.add_subplot(rate_gs[0]), fig.add_subplot(rate_gs[1])), o)
    ax_lfp = fig.add_subplot(strips[3])
    draw_lfp(ax_lfp, o)

    # wire the electrode to the LFP strip
    cp = ConnectionPatch(xyA=(back[0] + 0.10, back[1] - 0.16), coordsA=ax_net.transData,
                         xyB=(0.015, 0.12), coordsB=ax_lfp.transAxes, arrowstyle="-|>",
                         mutation_scale=8, linewidth=0.9, color=INK2,
                         connectionstyle="arc3,rad=0.2", shrinkA=2, shrinkB=4, zorder=10)
    fig.add_artist(cp)

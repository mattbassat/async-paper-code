"""Fig. 1b: how acetylcholine and serotonin move the network along the excitability axis swept
in Fig. 1c-f."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from asyncresp.style import INK, INK2

MEMBRANE = "#d9d2c5"
ROCKET = sns.color_palette("rocket", n_colors=7)
C_OD, C_UD = ROCKET[1], ROCKET[5]          # dark = low-pass end, light = resonant end
K_COL = "#3a6ea8"                          # K+ ions
CELL = dict(C=200.0, gL=10.0, EL=-70.0, DT=2.0, VT=-50.0, tauw=500.0, a=0.0, Vr=-65.0,
            b=0.0, refr=5.0)


# --- single AdEx cell ---
def adex(I_pA, T=900.0, dt=0.02, t_on=150.0, t_off=800.0, **kw):
    """Forward-Euler integration of one AdEx cell (units: pA, mV, nS, pF, ms) receiving a
    current step I_pA between t_on and t_off; vectorised over I_pA."""
    p = dict(CELL); p.update(kw)
    I = np.atleast_1d(np.asarray(I_pA, float))
    n = int(T / dt)
    t = np.arange(n) * dt
    V = np.full(I.shape, p["EL"]); w = np.zeros_like(V)
    ref = np.zeros_like(V)
    Vcut = p["VT"] + 5 * p["DT"]
    Vout = np.empty((n, I.size)); spk = np.zeros((n, I.size), bool)
    for k in range(n):
        drive = I * ((t[k] >= t_on) & (t[k] < t_off))
        exp_term = p["gL"] * p["DT"] * np.exp(np.clip((V - p["VT"]) / p["DT"], -50, 20))
        dV = (-p["gL"] * (V - p["EL"]) + exp_term - w + drive) / p["C"]
        live = ref <= 0
        V = np.where(live, V + dV * dt, V)
        w = w + ((p["a"] * (V - p["EL"]) - w) / p["tauw"]) * dt
        ref = np.maximum(ref - dt, 0.0)
        fired = V > Vcut
        Vout[k] = np.where(fired, 20.0, np.minimum(V, Vcut))    # cosmetic spike
        spk[k] = fired
        V = np.where(fired, p["Vr"], V)
        w = np.where(fired, w + p["b"], w)
        ref = np.where(fired, p["refr"], ref)
    return t, Vout, spk


# --- drawing helpers ---
def membrane(ax, y=0.60, h=0.13, x0=0.02, x1=0.98):
    ax.add_patch(Rectangle((x0, y), x1 - x0, h, facecolor=MEMBRANE, edgecolor="none", zorder=1))
    for yy in (y, y + h):
        ax.plot([x0, x1], [yy, yy], color="#b5ab99", lw=0.7, zorder=2)


def channel(ax, x, y=0.60, h=0.13, w=0.075, colour="#8d8d8d", open_=True, label=None):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.012), w, h + 0.024, boxstyle="round,pad=0.008",
                                facecolor=colour, edgecolor=INK, linewidth=0.6, zorder=3))
    if open_:                                                    # a pore down the middle
        ax.add_patch(Rectangle((x - 0.012, y - 0.008), 0.024, h + 0.016, facecolor="white",
                               edgecolor="none", zorder=4))
    if label:
        ax.text(x, y - 0.095, label, fontsize=6.2, color=INK, ha="center", va="top")


def receptor(ax, x, y=0.60, h=0.13, colour="#c9a227", label=None):
    ax.add_patch(FancyBboxPatch((x - 0.045, y - 0.012), 0.09, h + 0.024, boxstyle="round,pad=0.008",
                                facecolor=colour, edgecolor=INK, linewidth=0.6, zorder=3))
    ax.add_patch(plt.Circle((x, y + h + 0.035), 0.021, facecolor=colour, edgecolor=INK,
                            linewidth=0.5, zorder=4))
    if label:
        ax.text(x, y - 0.095, label, fontsize=6.2, color=INK, ha="center", va="top")


def cascade(ax, x0, x1, y=0.53, block=True, colour=INK):
    """intracellular arrow from receptor to channel: -| for block, -> for opening"""
    style = "|-|,widthA=0.0,widthB=0.35" if block else "-|>"
    a = FancyArrowPatch((x0, y), (x1, y), connectionstyle="arc3,rad=-0.35", arrowstyle=style,
                        mutation_scale=7, linewidth=1.0, color=colour, shrinkA=3, shrinkB=3, zorder=5)
    ax.add_patch(a)


def k_flux(ax, x, y=0.75, n=3, up=True, blocked=False):
    for i in range(n):
        ax.add_patch(plt.Circle((x - 0.03 + 0.03 * i, y + 0.02 * (i % 2)), 0.013,
                                facecolor=K_COL, edgecolor="none", zorder=4))
    dy = 0.10 if up else -0.10
    ax.add_patch(FancyArrowPatch((x, y + 0.045), (x, y + 0.045 + dy), arrowstyle="-|>",
                                 mutation_scale=7, linewidth=0.9,
                                 color="0.65" if blocked else K_COL, zorder=4))
    if blocked:
        ax.plot([x - 0.035, x + 0.035], [y + 0.10, y + 0.16], color="#c0392b", lw=1.3, zorder=6)
        ax.plot([x - 0.035, x + 0.035], [y + 0.16, y + 0.10], color="#c0392b", lw=1.3, zorder=6)
    ax.text(x + 0.055, y + 0.02, "K$^+$", fontsize=6.2, color=K_COL, va="center", ha="left")


def transmitter(ax, x, y=0.90, colour="#c9a227", label=None, dx_label=-0.115):
    """the motif: three free molecules above the receptor, plus the one bound to it"""
    for i, dx in enumerate((-0.05, 0.0, 0.05)):
        ax.add_patch(plt.Circle((x + dx, y + 0.03 * (i % 2)), 0.017, facecolor=colour,
                                edgecolor=INK, linewidth=0.4, zorder=4))
    if label:
        ax.text(x + dx_label, y + 0.015, label, fontsize=7.5, color=INK, ha="right",
                va="center", weight="bold")


def param_box(ax, x, y, text, colour, w=0.42, h=0.20):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02",
                                facecolor=colour, alpha=0.18, edgecolor=colour, linewidth=0.8,
                                zorder=2))
    ax.text(x, y, text, fontsize=7.5, color=INK, ha="center", va="center", zorder=3)


# --- the three rows ---
def draw_mechanism_ach(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0.22, 1.06); ax.axis("off")
    membrane(ax)
    receptor(ax, 0.30, colour="#c9a227", label="M1")
    channel(ax, 0.72, colour="#9fb3c8", open_=False, label="K$^+$")
    cascade(ax, 0.34, 0.68, block=True)
    k_flux(ax, 0.72, blocked=True)
    transmitter(ax, 0.30, colour="#c9a227", label="ACh")
    ax.text(0.5, 0.29, "suppresses K$^+$ currents, reduces spike-dependent\nadaptation and depolarises cells",
            fontsize=6.8, color=INK2, ha="center", va="center", linespacing=1.4)


def draw_mechanism_5ht(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0.22, 1.06); ax.axis("off")
    membrane(ax)
    for x, lab, colour, sign, mol_lab, effect in ((0.30, "5-HT$_{1A}$", C_OD, -1, "5-HT",
                                                   "hyperpolarises cells"),
                                                  (0.78, "5-HT$_{2A}$", C_UD, +1, None,
                                                   "depolarises cells")):
        receptor(ax, x, colour="#b58fc2", label=lab)
        transmitter(ax, x, colour="#b58fc2", label=mol_lab)
        ax.add_patch(FancyArrowPatch((x - 0.145, 0.36 - sign * 0.105),
                                     (x - 0.145, 0.36 + sign * 0.105), arrowstyle="-|>",
                                     mutation_scale=8, linewidth=1.3, color=colour, zorder=5))
        ax.text(x - 0.115, 0.36, effect, fontsize=6.8, color=colour, ha="left", va="center")


def draw_signature_ach(axes, I=420.0):
    """spike-frequency adaptation: the iconic cholinergic experiment, here in the AdEx cell"""
    for ax, (b, col, lab) in zip(axes, ((60.0, "0.45", "control  ($b$ = 60 pA)"),
                                        (0.0, C_UD, "+ ACh  ($b$ = 0 pA)"))):
        t, V, _ = adex(I, b=b)
        ax.plot(t, V[:, 0], color=col, linewidth=0.7)
        ax.set_xlim(100, 900); ax.set_ylim(-80, 35)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.015, 1.0, lab, transform=ax.transAxes, fontsize=6.4, color=col,
                ha="left", va="top")
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("$V_m$", fontsize=7.5, rotation=0, ha="right", va="center", labelpad=4)
    axes[1].plot([700, 900], [-78, -78], color=INK, lw=1.2, solid_capstyle="butt", clip_on=False)
    axes[1].text(800, -84, "200 ms", fontsize=6.2, color=INK, ha="center", va="top")


def draw_signature_5ht(axes, I=300.0):
    """the same current step into the same cell; only E_L differs (1A hyperpolarises, 2A
    depolarises)"""
    win = (110.0, 700.0)
    specs = ((-60.0, C_UD, "5-HT$_{2A}$   ($E_L$ = $-$60 mV)"),
             (-70.0, "0.45", "control   ($E_L$ = $-$70 mV)"),
             (-80.0, C_OD, "5-HT$_{1A}$   ($E_L$ = $-$80 mV)"))
    for ax, (EL, col, lab) in zip(axes, specs):
        t, V, _ = adex(I, EL=EL, b=0.0)
        ax.plot(t, V[:, 0], color=col, linewidth=0.7)             # spikes run off the top
        ax.set_xlim(*win); ax.set_ylim(-90, -40)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.0, 1.02, lab, transform=ax.transAxes, fontsize=6.3, color=col,
                ha="left", va="bottom")
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[1].set_ylabel("$V_m$", fontsize=7.5, rotation=0, ha="right", va="center", labelpad=4)
    axes[2].plot([win[1] - 210, win[1] - 10], [-88, -88], color=INK, lw=1.2, solid_capstyle="butt")
    axes[2].text(win[1] - 110, -93, "200 ms", fontsize=6.2, color=INK, ha="center", va="top")


def draw_params(ax):
    """The AdEx parameters each neuromodulator changes, with the direction of the change."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.0, 0.52, "Model\nparameters (AdEx)", fontsize=6.3, color=INK2,
            ha="left", va="center", style="italic", linespacing=1.4)
    param_box(ax, 0.305, 0.52, r"$\downarrow b$    $\uparrow E_L$    $\downarrow g_L$",
              C_UD, w=0.27, h=0.52)
    param_box(ax, 0.655, 0.52, r"$\downarrow E_L$", C_OD, w=0.175, h=0.52)
    param_box(ax, 0.895, 0.52, r"$\uparrow E_L$", C_UD, w=0.175, h=0.52)
    ax.text(0.655, 0.99, "5-HT$_{1A}$", fontsize=6.4, color=C_OD, ha="center", va="top")
    ax.text(0.900, 0.99, "5-HT$_{2A}$", fontsize=6.4, color=C_UD, ha="center", va="top")


def draw_axis_bar(ax):
    """the excitability key of Fig. 1d, with an arrow from each neuromodulator"""
    grad = np.linspace(0, 1, 256)[None, :]
    ax.imshow(grad, aspect="auto", cmap=sns.color_palette("rocket", as_cmap=True),
              extent=[0.22, 0.78, 0.56, 0.74])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.80, "Excitability", fontsize=7, ha="center", va="bottom", color=INK2)
    for x0, x1, colour, lab in ((0.50, 0.74, C_UD, "ACh,  5-HT$_{2A}$"),
                                (0.50, 0.26, C_OD, "5-HT$_{1A}$")):
        ax.add_patch(FancyArrowPatch((x0, 0.40), (x1, 0.40), arrowstyle="-|>", mutation_scale=9,
                                     linewidth=1.5, color=colour, shrinkA=0, shrinkB=0))
        ax.text(0.5 * (x0 + x1), 0.28, lab, fontsize=6.6, color=colour, ha="center", va="top")


# --- panel ---
def draw_panel_b(fig, spec):
    """The whole panel in one gridspec slot."""
    gs = spec.subgridspec(4, 2, height_ratios=[1.30, 1.55, 0.50, 0.50], hspace=0.28, wspace=0.26)
    ax_m1 = fig.add_subplot(gs[0, 0]); draw_mechanism_ach(ax_m1)
    ax_m2 = fig.add_subplot(gs[0, 1]); draw_mechanism_5ht(ax_m2)
    ach_gs = gs[1, 0].subgridspec(2, 1, hspace=0.12)
    draw_signature_ach([fig.add_subplot(ach_gs[0]), fig.add_subplot(ach_gs[1])])
    ht_gs = gs[1, 1].subgridspec(3, 1, hspace=0.55)
    draw_signature_5ht([fig.add_subplot(ht_gs[k]) for k in range(3)])
    draw_params(fig.add_subplot(gs[2, :]))
    draw_axis_bar(fig.add_subplot(gs[3, :]))
    ax_m1.set_title("Acetylcholine", fontsize=8.5, color="#7a6200", pad=1)
    ax_m2.set_title("Serotonin", fontsize=8.5, color="#6b4a7a", pad=1)

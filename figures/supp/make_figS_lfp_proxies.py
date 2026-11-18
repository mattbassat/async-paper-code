"""Supplementary figure: the spontaneous spectra of Fig. 1e-f for alternative LFP proxies and the
population firing rates."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "fig1"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import supp_common as SC
from asyncresp.io import load_pickle
from asyncresp.paths import OUTPUT, SUPP
from common import plain_log_ticks
from sweeps import load_sweep, sampled_resonance_peaks

FMIN, FMAX, FMAX_KERNEL = 2, 500, 150
ROWS = [("lfp_inh", "Inhibitory currents\n(Fig. 1 proxy)"),
        ("lfp_exc", "Excitatory currents"),
        ("absmag", "Absolute-magnitude\nsum"),
        ("kernel", "Kernel method"),
        ("rate_exc", "E population rate"),
        ("rate_inh", "I population rate")]
# (data set, sweep name, conditions, reverse palette, legend format), as in Fig. 1e-f
COLS = [("r2b0", "delta_EL", [(-15.0,), (-5.0,), (5.0,), (15.0,)], False,
         lambda v: f"{v:+g} mV".replace("-", "−")),
        ("params", "delta_gL", [("delta_gL", -5.0), ("delta_gL", 5.0)], False,
         lambda v: f"{v:+g} nS".replace("-", "−")),
        ("params", "b", [("b", 150.0), ("b", 0.0)], True, lambda v: f"{v:g} pA"),
        ("params", "Qee", [("Qee", 1.25), ("Qee", 1.6)], False, lambda v: f"{v:.2f} nS"),
        ("params", "Qei", [("Qei", 2.0), ("Qei", 1.4)], True, lambda v: f"{v:.1f} nS")]
COL_LABEL = [r"$\Delta_{E_L}$", r"$\Delta_{g_L}$", r"$b$", r"$Q_{ee}$", r"$Q_{ei}$"]


def make():
    SC.style()
    data = load_pickle(SUPP / "lfp_proxies" / "spectra.pkl")["spectra"]
    fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(7.2, 8.4),
                             gridspec_kw=dict(left=0.12, right=0.99, top=0.95, bottom=0.06,
                                              wspace=0.32, hspace=0.38))
    for j, (dset, sweep, conds, rev, fmt) in enumerate(COLS):
        agg, pname = load_sweep(sweep)
        colors = SC.rocket(agg[pname].unique(), reverse=rev)
        peaks = sampled_resonance_peaks(agg, pname)
        for i, (sig, label) in enumerate(ROWS):
            ax = axes[i, j]
            fmax = FMAX_KERNEL if sig == "kernel" else FMAX
            for cond in conds:
                v = cond[-1]
                e = data[(dset, cond)]
                f, p = e["f"], e[sig].mean(axis=0)
                m = (f >= FMIN) & (f <= fmax)
                col = colors[min(colors, key=lambda c: abs(c - v))]
                ax.plot(f[m], p[m], color=col, lw=0.9, label=fmt(v))
                fr = peaks.get(v)
                if fr is not None:
                    y = np.exp(np.interp(np.log(fr), np.log(f[m]), np.log(p[m])))
                    ax.scatter([fr], [y * 1.6], marker="v", s=14, color=col, edgecolor="black",
                               linewidth=0.4, zorder=5)
            ax.set_xscale("log"); ax.set_yscale("log")
            plain_log_ticks(ax.xaxis)
            ax.set_xlim(FMIN, fmax)
            ax.tick_params(labelsize=5.8)
            ax.yaxis.set_minor_formatter(plt.NullFormatter())
            if i == 0:
                ax.legend(fontsize=5.3, frameon=False, loc="lower left", handlelength=1.1,
                          borderaxespad=0.2, labelspacing=0.2, title=COL_LABEL[j],
                          title_fontsize=5.8)
            if j == 0:
                ax.set_ylabel(label + "\nPSD (a.u.)", fontsize=6.5)
            if i == len(ROWS) - 1:
                ax.set_xlabel("Frequency (Hz)", fontsize=6.5)
    out = OUTPUT / "supp"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figS_lfp_proxies.png", dpi=300)
    fig.savefig(out / "figS_lfp_proxies.pdf")
    return fig


if __name__ == "__main__":
    make()

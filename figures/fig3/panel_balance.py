"""Fig. 3b: normalised slow-input E responses against the balance metric B."""
import numpy as np

from asyncresp.style import INK, INK2

from sweep_data import SETS
from panel_slow_input import OD_COL, UD_COL, UD_INK

B_SYM = r"$\mathcal{B}$"
B_LIM = (0.033, 2.25)


def analytic_vs_balance(slow, e):
    """Eq. 4 E response against the balance metric, normalised by its maximum."""
    m = (e.delta_EL >= slow.delta_EL.min()) & (e.delta_EL <= slow.delta_EL.max())
    bal, he0 = (e.zeta[m] ** 2).values, e.He0[m].values
    return bal, he0 / he0.max()


def draw_balance(ax, data, slow):
    """Panel b. `data[k]` = (agg, e) and `slow[k]` = slow-input table of parameter set k."""
    lo, hi = B_LIM
    ax.set_xscale("log")
    ax.axvspan(1.0, hi, color=OD_COL, linewidth=0, zorder=0)
    ax.axvspan(lo, 1.0, color=UD_COL, linewidth=0, zorder=0)
    ax.axvline(1.0, color=INK, linewidth=0.7, zorder=1)
    for k, cfg in SETS.items():
        s = slow[k].copy()
        s["balance"] = s["zeta"] ** 2
        s = s.sort_values("balance")
        bal, he0 = analytic_vs_balance(slow[k], data[k][1])
        ax.plot(bal, he0, color=cfg["colour"], linewidth=1.1, zorder=2, alpha=0.95)
        ax.plot(s.balance, s.amp_e / s.amp_e.max(), marker=cfg["marker"], ls="none",
                color=cfg["colour"], ms=3.8, mec="white", mew=0.4, label=cfg["short"], zorder=3)
    ax.text(np.sqrt(1.0 * hi), 1.17, "OD", ha="center", va="top", fontsize=6.8, weight="bold",
            color=INK2)
    ax.text(np.sqrt(lo * 1.0), 1.17, "UD", ha="center", va="top", fontsize=6.8, weight="bold",
            color=UD_INK)
    # the analytical response becomes negative deep in the UD regime (as in panel a)
    ax.set_xlim(lo, hi); ax.set_ylim(-0.38, 1.2)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Balance metric " + B_SYM)
    ax.set_ylabel("E response to slow input\n(normalised)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.005, 0.93), frameon=False,
              fontsize=5.8, title=r"$\nu_{drive}$, $b$", title_fontsize=5.8,
              handlelength=1.5, borderaxespad=0.3, labelspacing=0.3)

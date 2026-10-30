"""Fig. 2c: phase planes and relaxation of the linearised mean field for an OD (stable node) and
a UD (stable focus) network."""
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from asyncresp import mean_field as MF
from asyncresp.style import E_COL, I_COL, INK, INK2

from fig2_common import B_SYM


def linear_system(dEL, ext=2.0, b=0.0):
    """Jacobian of the mean field at its fixed point, in rate units rescaled so that its two
    off-diagonal elements are equal in magnitude (S = diag(r_e, r_i), r_i / r_e =
    sqrt(|C/B|); M = S^-1 J S)."""
    fp = MF.fixed_point(ext=ext, ELe=-70., ELi=-70. - dEL, b=b)
    ei = MF.eigen(fp)
    A, B, C, D = (float(ei[k][0]) for k in "ABCD")
    k = np.sqrt(abs(C / B))
    M = np.array([[A, B * k], [C / k, D]])
    return ei, M


def lin_traj(M, x0, tmax, n=900):
    """x(t) = exp(M t) x0, by diagonalisation (M is 2x2 and never defective here)."""
    t = np.linspace(0, tmax, n)
    lam, V = np.linalg.eig(M)
    c = np.linalg.solve(V, np.asarray(x0, complex))
    return t, np.real(V @ (c[:, None] * np.exp(np.outer(lam, t))))


def draw_phase(ax, dEL, ext=2.0, b=0.0, n_traj=6, tmax=0.10, title=None, colour=INK,
               ylabel=True):
    ei, M = linear_system(dEL, ext, b)
    # just enough room for the whole family of trajectories
    span = 1.14 * max(np.abs(lin_traj(M, [np.cos(t), np.sin(t)], tmax)[1]).max()
                      for t in np.linspace(0, 2 * np.pi, 48, endpoint=False))

    # nullclines of the linearised flow (tangent to the true ones at the fixed point)
    s = np.array([-span * 1.6, span * 1.6])
    for row, col in ((0, E_COL), (1, I_COL)):
        a, c = M[row, 0], M[row, 1]
        xs, ys = (s, -a / c * s) if abs(c) > abs(a) else (-c / a * s, s)
        ax.plot(xs, ys, color=col, linewidth=0.8, linestyle=(0, (4, 2)), zorder=2)

    for j, th in enumerate(np.linspace(0, 2 * np.pi, n_traj, endpoint=False)):
        t, X = lin_traj(M, [np.cos(th), np.sin(th)], tmax)
        ax.plot(X[0], X[1], color=colour, linewidth=0.7, alpha=0.9, zorder=3)
        ax.plot(X[0, 0], X[1, 0], "o", color=colour, markersize=2.4, zorder=4)
        if j == n_traj * 3 // 4:                         # one arrow for the flow direction
            k = max(2, int(0.03 * X.shape[1]))
            ax.annotate("", xy=(X[0, k], X[1, k]), xytext=(X[0, 0], X[1, 0]), zorder=5,
                        arrowprops=dict(arrowstyle="-|>", color=colour, lw=0.8, shrinkA=0, shrinkB=0))
    ax.plot([0], [0], "o", color=INK, markersize=4.5, zorder=6,
            markeredgecolor="white", markeredgewidth=0.6)

    ax.set_xlim(-span, span); ax.set_ylim(-span, span)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\nu_e - \nu_e^*$  (a.u.)")
    if ylabel:
        ax.set_ylabel(r"$\nu_i - \nu_i^*$  (a.u.)")
    if title:
        ax.set_title(title, fontsize=7.5, color=colour, pad=3)
    od = float(ei["zeta"][0]) >= 1
    state = B_SYM + r" $> 1$" if od else r"$0 < $ " + B_SYM + r" $< 1$"
    lab = state + "\n" + (r"$\lambda_{1,2}$ real, $<0$" if od else r"$\lambda_{1,2}$ complex")
    ax.text(0.035, 0.965, lab, transform=ax.transAxes, fontsize=6.6, color=INK2, ha="left",
            va="top", linespacing=1.5, zorder=7,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
    if not od:                                           # the nullcline key, in the UD plane
        ax.legend([Line2D([], [], color=E_COL, lw=0.8, ls=(0, (4, 2))),
                   Line2D([], [], color=I_COL, lw=0.8, ls=(0, (4, 2)))],
                  [r"$\nu_e$ nullcline", r"$\nu_i$ nullcline"], loc="lower right", frameon=False,
                  fontsize=5.8, handlelength=1.4, labelspacing=0.2, borderaxespad=0.15,
                  handletextpad=0.5)


def draw_relaxation(ax, dEL, ext=2.0, b=0.0, tmax=0.10, colour=INK, ylabel=True):
    """The same linearised system, released from a pure E perturbation."""
    _, M = linear_system(dEL, ext, b)
    t, X = lin_traj(M, [1.0, 0.0], tmax, n=1600)
    ax.axhline(0.0, color=INK2, linewidth=0.5, linestyle=(0, (1, 2.2)), zorder=1)
    ax.plot(t * 1e3, X[0], color=colour, linewidth=1.0, zorder=2)
    ax.set_xlabel("Time (ms)")
    if ylabel:
        ax.set_ylabel(r"$\nu_e - \nu_e^*$" + "\n(a.u.)")
    ax.set_xlim(0, tmax * 1e3)
    lo = min(-0.35, 1.16 * X[0].min())
    ax.set_ylim(lo, 1.15); ax.set_yticks([0, 1])

"""Settings shared by the panels of Fig. 2: the three parameter sets, colours, plot style, and
the mean-field stability quantities (eigenvalues and the balance metric B) along the ΔEL axis.

Balance metric.  For the Jacobian J* of the two-dimensional rate dynamics at the fixed point,
B = (tr J*)^2 / (4 det J*), which equals the square of the damping ratio.  B = 1 marks the
boundary between a stable node (B > 1, over-damped, OD) and a stable focus (B < 1,
under-damped, UD); on the focus side the ringing frequency of the mean field relative to its
natural frequency f0 = sqrt(det J*) / 2 pi is f_eig / f0 = sqrt(1 - B).
"""
import numpy as np
import matplotlib as mpl
import seaborn as sns

from asyncresp import mean_field as MF

# (tag, nu_drive in Hz, adaptation b in pA, panel title)
CONDITIONS = [("r2b0", 2.0, 0.0, r"$\nu_{drive}$ = 2 Hz, $b$ = 0 pA"),
              ("r4b0", 4.0, 0.0, r"$\nu_{drive}$ = 4 Hz, $b$ = 0 pA"),
              ("r4b60", 4.0, 60.0, r"$\nu_{drive}$ = 4 Hz, $b$ = 60 pA")]

ROCKET = sns.color_palette("rocket", n_colors=7)
C_OD, C_UD = ROCKET[1], ROCKET[5]            # the OD and UD example networks of Fig. 2c
SH_OD, SH_UD = "#e3e3e3", "#dbeefb"          # background bands: node (OD) / focus (UD) side
C_RE, C_IM = "#1f1f1f", "#2a78d6"            # real / imaginary parts of the eigenvalues
C_BAL = "#a03d8a"                            # the balance metric B

B_SYM = r"$\mathcal{B}$"
B_CRIT = 1.0                                 # node/focus boundary
FMIN_RES = 10.0                              # an SNN response peak counts as a resonance above this (Hz)


def style():
    """Font sizes and line widths used throughout the figure."""
    sns.set_style("white")
    mpl.rcParams.update({
        "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def balance(ei):
    """Balance metric B = tr^2 / (4 det) of the Jacobian (dimensionless)."""
    return ei["tr"] ** 2 / (4 * ei["det"])


def eigen_sweep(ext, b, lo=-17.0, hi=17.0, n=161):
    """Mean-field eigenvalues along ΔEL = EL_exc - EL_inh (EL_exc fixed at -70 mV), plus the
    ΔEL at which the discriminant tr^2 - 4 det changes sign (the node/focus boundary)."""
    d = np.linspace(lo, hi, n)
    ei = MF.eigen(MF.fixed_point(ext=ext, ELe=-70., ELi=-70. - d, b=b))
    ei["dEL"] = d
    s = np.sign(ei["disc"])
    j = np.where(np.diff(s) != 0)[0]
    ei["dcrit"] = (float(np.interp(0, [ei["disc"][j[0]], ei["disc"][j[0] + 1]],
                                   [d[j[0]], d[j[0] + 1]])) if len(j) else np.nan)
    return ei

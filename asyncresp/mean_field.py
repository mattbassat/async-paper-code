"""
First-order mean field of the AdEx network (di Volo et al., 2019; Zerlaut et al., 2018).

The model describes the mean excitatory and inhibitory firing rates (nu_e, nu_i) and the mean
excitatory adaptation current w.  This module provides the semi-analytic transfer function,
the stationary fixed point, the Jacobian of the two-dimensional rate dynamics (adaptation held
at its stationary value), its eigenvalues, and helpers for integrating the rate equations.

Note on tau_V: the inhibitory contribution to the autocorrelation time tau_V is weighted by
(U_i * T_i)^2.  An earlier version of this code weighted it by (Q_i * U_i)^2; the two are
identical whenever the excitatory and inhibitory synaptic time constants are equal, as they
are for every network in the paper.

Typical use:
    fp = fixed_point(ext=2.0, ELe=-70., ELi=-60., b=0.)   # nu_e*, nu_i*, w*, T
    ei = eigen(fp)                                         # tr, det, damping ratio, eigenvalues
"""
import numpy as np
from scipy.special import erf

# Polynomial coefficients of the effective threshold, fitted to the AdEx RS and FS cells
# (di Volo et al., 2019)
PRS = [-0.0498, 0.00506, -0.025, 0.0014, -0.00041, 0.0105, -0.036, 0.0074, 0.0012, -0.0407]
PFS = [-0.0514, 0.004, -0.0083, 0.0002, -0.0005, 0.0014, -0.0146, 0.0045, 0.0028, -0.0153]

Cm, Ee, Ei = 200e-12, 0.0, -80e-3
Te, Ti = 5e-3, 5e-3                    # synaptic time constants
pconnec, gei, Ntot = 0.05, 0.2, 10000
TAU_W = 0.5                            # s


def TF(P, fexc, finh, adapt, EL_mV, Qe=1.5e-9, Qi=5e-9, Gl=10e-9):
    """mean firing rate (Hz) and tau_V (s) of a cell of type P given pre-synaptic rates"""
    fe = fexc * (1. - gei) * pconnec * Ntot
    fi = finh * gei * pconnec * Ntot
    El = EL_mV * 1e-3
    muGe, muGi = Qe * Te * fe, Qi * Ti * fi
    muG = Gl + muGe + muGi
    muV = (muGe * Ee + muGi * Ei + Gl * El - adapt) / muG
    Tm = Cm / muG
    Ue, Ui = Qe / muG * (Ee - muV), Qi / muG * (Ei - muV)
    sV = np.sqrt(fe * (Ue * Te) ** 2 / 2. / (Te + Tm) + fi * (Ui * Ti) ** 2 / 2. / (Ti + Tm))
    fe, fi = fe + 1e-9, fi + 1e-9
    Tv = ((fe * (Ue * Te) ** 2 + fi * (Ui * Ti) ** 2)
          / (fe * (Ue * Te) ** 2 / (Te + Tm) + fi * (Ui * Ti) ** 2 / (Ti + Tm)))
    TvN = Tv * Gl / Cm
    x1, x2, x3 = (muV + 60e-3) / 10e-3, (sV - 4e-3) / 6e-3, TvN - 0.5
    vthr = (P[0] + P[1] * x1 + P[2] * x2 + P[3] * x3 + P[4] * x1 * x1 + P[5] * x2 * x2
            + P[6] * x3 * x3 + P[7] * x1 * x2 + P[8] * x1 * x3 + P[9] * x2 * x3)
    return .5 / TvN * Gl / Cm * (1 - erf((vthr - muV) / np.sqrt(2) / sV)), Tv


def _rates(fe, fi, w, ext, ELe, ELi, Qee, Qei, gLe, gLi):
    Fe = TF(PRS, fe + ext, fi, w, ELe, Qe=Qee, Gl=gLe)
    Fi = TF(PFS, fe + ext, fi, 0.0, ELi, Qe=Qei, Gl=gLi)
    return Fe, Fi


def fixed_point(ext, ELe=-70., ELi=-70., b=0., Qee=1.5e-9, Qei=1.5e-9, gLe=10e-9, gLi=10e-9,
                relax_s=5.0, dt=5e-4, newton=10):
    """stationary (nu_e*, nu_i*): relax the mean field, then polish with Newton.
    Adaptation is adiabatic (tau_w = 500 ms >> T): w* = b * tau_w * nu_e*."""
    ext, ELe, ELi, b = np.broadcast_arrays(*[np.atleast_1d(np.asarray(v, float))
                                             for v in (ext, ELe, ELi, b)])
    n = ext.size
    fe, fi, w, T = np.full(n, 5.), np.full(n, 10.), np.full(n, 600e-12), np.full(n, 0.01)
    bA = b * 1e-12
    for _ in range(int(relax_s / dt)):                       # relax to the neighbourhood
        feold = fe
        (Fe, Tv), (Fi, _) = _rates(fe, fi, w, ext, ELe, ELi, Qee, Qei, gLe, gLi)
        fe = fe + dt / T * (Fe - fe)
        w = w + dt * (-w / TAU_W + bA * feold)
        fi = fi + dt / T * (Fi - fi)
        T = Tv

    def G(a, c):                                             # residual with w slaved to a
        (Fe, _), (Fi, _) = _rates(a, c, bA * TAU_W * a, ext, ELe, ELi, Qee, Qei, gLe, gLi)
        return Fe - a, Fi - c

    for _ in range(newton):
        g0 = G(fe, fi)
        he, hi = 1e-6 * np.maximum(1, fe), 1e-6 * np.maximum(1, fi)
        ga, gc = G(fe + he, fi), G(fe, fi + hi)
        j11, j21 = (ga[0] - g0[0]) / he, (ga[1] - g0[1]) / he
        j12, j22 = (gc[0] - g0[0]) / hi, (gc[1] - g0[1]) / hi
        d = j11 * j22 - j12 * j21
        fe = fe - (j22 * g0[0] - j12 * g0[1]) / d
        fi = fi - (-j21 * g0[0] + j11 * g0[1]) / d
    w = bA * TAU_W * fe
    T = TF(PRS, fe + ext, fi, w, ELe, Qe=Qee, Gl=gLe)[1]
    return dict(fe=fe, fi=fi, w=w, T=T, ext=ext, ELe=ELe, ELi=ELi, b=b,
                Qee=Qee, Qei=Qei, gLe=gLe, gLi=gLi)


def flow(fe, fi, fp):
    """(d nu_e/dt, d nu_i/dt) at (fe, fi), with w frozen at its stationary value"""
    (Fe, _), (Fi, _) = _rates(fe, fi, fp["w"], fp["ext"], fp["ELe"], fp["ELi"],
                              fp["Qee"], fp["Qei"], fp["gLe"], fp["gLi"])
    return (Fe - fe) / fp["T"], (Fi - fi) / fp["T"]


def jacobian(fp, rel=1e-6):
    fe, fi = fp["fe"], fp["fi"]
    he, hi = np.maximum(1e-9, rel * np.maximum(1, fe)), np.maximum(1e-9, rel * np.maximum(1, fi))
    p, m = flow(fe + he, fi, fp), flow(fe - he, fi, fp)
    A, C = (p[0] - m[0]) / (2 * he), (p[1] - m[1]) / (2 * he)
    p, m = flow(fe, fi + hi, fp), flow(fe, fi - hi, fp)
    B, D = (p[0] - m[0]) / (2 * hi), (p[1] - m[1]) / (2 * hi)
    return np.array([[A, B], [C, D]])


def eigen(fp):
    """tr, det, damping ratio zeta, natural frequency f0, eigenvalues and ringing frequency"""
    J = jacobian(fp)
    A, B, C, D = J[0, 0], J[0, 1], J[1, 0], J[1, 1]
    tr, det = A + D, A * D - B * C
    disc = tr ** 2 - 4 * det
    sq = np.sqrt(np.abs(disc).astype(complex)) * np.where(disc >= 0, 1.0, 1j)
    lam1, lam2 = (tr + sq) / 2, (tr - sq) / 2
    zeta = -tr / (2 * np.sqrt(det))
    f0 = np.sqrt(det) / (2 * np.pi)
    fd = np.where(disc < 0, np.sqrt(np.maximum(-disc, 0)) / (4 * np.pi), np.nan)
    return dict(J=J, A=A, B=B, C=C, D=D, tr=tr, det=det, disc=disc, zeta=zeta,
                f0=f0, fd=fd, lam1=lam1, lam2=lam2,
                vecs=[np.linalg.eig(np.array([[A[k], B[k]], [C[k], D[k]]]))[1]
                      for k in range(np.size(A))] if np.ndim(A) else None)


def trajectory(fe0, fi0, fp, tmax=0.25, dt=2e-5):
    """integrate the 2-D mean field from (fe0, fi0) with w frozen (exact for b = 0)"""
    n = int(tmax / dt)
    fe, fi = float(fe0), float(fi0)
    out = np.empty((n, 3))
    one = {k: (v[0] if np.ndim(v) else v) for k, v in fp.items()}
    for k in range(n):
        out[k] = (k * dt, fe, fi)
        d_fe, d_fi = flow(np.array([fe]), np.array([fi]), one)
        fe, fi = fe + dt * float(d_fe), fi + dt * float(d_fi)
    return out


def relax_trace(ext, ELe=-70., ELi=-70., b=0., Qee=1.5e-9, Qei=1.5e-9, gLe=10e-9, gLi=10e-9,
                fe0=0.0, fi0=0.0, w0=0.0, tmax=1.5, dt=1e-4):
    """the first-order mean field run forward in time from (fe0, fi0, w0) under a constant
    external drive: the adaptation builds up over tau_w and the rates settle on the fixed
    point.  Returns t (s), nu_e, nu_i, w."""
    n = int(tmax / dt)
    fe, fi, w, T = float(fe0), float(fi0), float(w0), 0.01
    bA = b * 1e-12
    out = np.empty((n, 4))
    for k in range(n):
        out[k] = (k * dt, fe, fi, w)
        (Fe, Tv) = TF(PRS, fe + ext, fi, w, ELe, Qe=Qee, Gl=gLe)
        (Fi, _) = TF(PFS, fe + ext, fi, 0.0, ELi, Qe=Qei, Gl=gLi)
        fe_new = fe + dt / T * (Fe - fe)
        fi = fi + dt / T * (Fi - fi)
        w = w + dt * (-w / TAU_W + bA * fe)
        fe, T = fe_new, max(float(Tv), 4 * dt)     # keep the Euler step stable at very low rates
    return out


def drive_trace(ext_of_t, ELe=-70., ELi=-70., b=0., Qee=1.5e-9, Qei=1.5e-9, gLe=10e-9,
                gLi=10e-9, fe0=1.0, fi0=5.0, w0=0.0, tmax=4.0, dt=1e-4):
    """the first-order mean field under a time-varying external drive nu_drive(t) (Hz), given
    as a callable of t in seconds.  Returns t (s), nu_e, nu_i, w, nu_drive."""
    n = int(tmax / dt)
    fe, fi, w, T = float(fe0), float(fi0), float(w0), 0.01
    bA = b * 1e-12
    out = np.empty((n, 5))
    for k in range(n):
        t = k * dt
        ext = float(ext_of_t(t))
        out[k] = (t, fe, fi, w, ext)
        (Fe, Tv) = TF(PRS, fe + ext, fi, w, ELe, Qe=Qee, Gl=gLe)
        (Fi, _) = TF(PFS, fe + ext, fi, 0.0, ELi, Qe=Qei, Gl=gLi)
        fe_new = fe + dt / T * (Fe - fe)
        fi = fi + dt / T * (Fi - fi)
        w = w + dt * (-w / TAU_W + bA * fe)
        fe, T = fe_new, max(float(Tv), 4 * dt)
    return out

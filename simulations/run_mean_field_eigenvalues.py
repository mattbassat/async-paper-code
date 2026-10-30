"""Linear stability of the mean-field fixed point as a function of Delta_EL (Fig. 3a curves)."""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import numpy.linalg as la
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asyncresp import mean_field as MF
from asyncresp.paths import SWEEPS

SETS = {"r2b0": dict(rate=2, b=0), "r4b0": dict(rate=4, b=0), "r4b60": dict(rate=4, b=60)}
EL_EXC = -70
EL_INH_RANGE = np.arange(-85, -54.5, 0.5)      # Delta_EL = +15 ... -15 mV
TW = 0.5                                       # adaptation time constant (s)


def run_mf_dynamic_T(fe0, fi0, w0, b, ext_drive, EL_exc, EL_inh, tfinal=5, dt=0.001, T_init=0.010):
    """Euler integration of the first-order mean field; T follows tau_V of the RS population.
    Returns the time series of nu_e, nu_i (Hz), w (A) and T (s)."""
    t = np.linspace(0, tfinal, int(tfinal / dt))
    fecont, ficont, w, T = fe0, fi0, w0, T_init
    bRS = b * 1.e-12
    LSfe, LSfi, LSw, LST = [], [], [], []
    for _ in range(len(t)):
        fecontold = fecont
        Fe, Tv = MF.TF(MF.PRS, fecont + ext_drive, ficont, w, EL_exc)
        Fi = MF.TF(MF.PFS, fecontold + ext_drive, ficont, 0., EL_inh)[0]
        fecont += dt / T * (Fe - fecont)
        w += dt * (-w / TW + bRS * fecontold)
        ficont += dt / T * (Fi - ficont)
        T = Tv
        LSfe.append(float(fecont))
        LSfi.append(float(ficont))
        LSw.append(float(w))
        LST.append(float(T))
    return LSfe, LSfi, LSw, LST


def fixed_point_jacobian(fe0, fi0, w0, ext, EL_exc, EL_inh, T, eps_rel=1e-6, eps_abs=1e-9):
    """2x2 Jacobian of (d nu_e/dt, d nu_i/dt) at (fe0, fi0) with w = w0 fixed (central
    differences)."""
    def flow(fe, fi):
        fe_in = fe + ext
        return np.array([(MF.TF(MF.PRS, fe_in, fi, w0, EL_exc)[0] - fe) / T,
                         (MF.TF(MF.PFS, fe_in, fi, 0.0, EL_inh)[0] - fi) / T], dtype=float)

    x0 = np.array([fe0, fi0], dtype=float)
    steps = np.maximum(eps_abs, eps_rel * np.maximum(1.0, np.abs(x0)))
    J = np.zeros((2, 2))
    for j in range(2):
        dx = np.zeros(2)
        dx[j] = steps[j]
        J[:, j] = (flow(*(x0 + dx)) - flow(*(x0 - dx))) / (2 * dx[j])
    return J


def eigen_row(rate, b, EL_inh, EL_exc=EL_EXC):
    """Stationary state, Jacobian and eigen-decomposition for one Delta_EL."""
    fe, fi, w, T = run_mf_dynamic_T(5, 10, 600e-12, b, rate, EL_exc, EL_inh)
    fe_stat, fi_stat = np.mean(fe[-1000:]), np.mean(fi[-1000:])
    w_stat, T_stat = np.mean(w[-1000:]), np.mean(T[-1000:])
    J = fixed_point_jacobian(fe_stat, fi_stat, w_stat, rate, EL_exc, EL_inh, T_stat)
    eigs, vecs = la.eig(J)
    imag_parts = np.abs(np.imag(eigs))
    idx = np.argmax(imag_parts)
    omega = np.abs(np.imag(eigs[idx])) / (2 * np.pi) if imag_parts[idx] > 1e-12 else 0.0
    return {
        "EL_inh": EL_inh, "delta_EL": EL_exc - EL_inh,
        "Re_lambda1": eigs[0].real, "Im_lambda1": eigs[0].imag,
        "Re_lambda2": eigs[1].real, "Im_lambda2": eigs[1].imag,
        "fe_stationary": fe_stat, "fi_stationary": fi_stat,
        "w_stationary": w_stat, "T_stationary": T_stat,
        "dfedot_dfe": J[0, 0], "dfedot_dfi": J[0, 1], "dfidot_dfe": J[1, 0], "dfidot_dfi": J[1, 1],
        "eigenvector1_real": vecs[:, 0].real, "eigenvector1_imag": vecs[:, 0].imag,
        "eigenvector2_real": vecs[:, 1].real, "eigenvector2_imag": vecs[:, 1].imag,
        "eigenfrequency": omega,
    }


def eigen_table(rate, b):
    df = pd.DataFrame([eigen_row(rate, b, EL_inh) for EL_inh in EL_INH_RANGE])
    # boundary: midway between the last complex and the first real eigenvalue pair
    imag_vals, delta_vals = df["Im_lambda1"].values, df["delta_EL"].values
    i = next(k for k in range(len(imag_vals)) if abs(imag_vals[k]) == 0)
    df["delta_crit"] = 0.5 * (delta_vals[i - 1] + delta_vals[i])
    return df


def check(key="r2b0", row=0):
    """Recompute one Delta_EL row of a stored table and report the largest differences."""
    stored = pd.read_pickle(SWEEPS / f"eigen_df_{key}_dense.pkl").iloc[row]
    new = eigen_row(SETS[key]["rate"], SETS[key]["b"], stored["EL_inh"])
    print(f"eigen_df_{key}_dense.pkl, Delta_EL = {stored['delta_EL']} mV")
    worst = 0.0
    for k, v in new.items():
        a, s = np.asarray(v, float), np.asarray(stored[k], float)
        d = float(np.max(np.abs(a - s)))
        rel = d / max(float(np.max(np.abs(s))), 1e-300)
        worst = max(worst, rel)
        print(f"  {k:20s} max |diff| = {d:.3e}  (relative {rel:.1e})")
    print(f"largest relative difference: {worst:.1e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    ap.add_argument("--out", type=Path, default=SWEEPS)
    ap.add_argument("--check", action="store_true", help="recompute one stored row and compare")
    args = ap.parse_args()
    if args.check:
        check()
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for key in args.sets:
        df = eigen_table(SETS[key]["rate"], SETS[key]["b"])
        path = args.out / f"eigen_df_{key}_dense.pkl"
        with open(path, "wb") as f:
            pickle.dump(df, f)
        print(f"{key}: delta_crit = {df['delta_crit'].iloc[0]} mV -> {path}")


if __name__ == "__main__":
    main()

"""Evoked responses to a slow half-cycle pulse vs Delta_EL, for pulse amplitudes up to 50 Hz."""
import os
os.environ.setdefault("OPENBLAS_CORETYPE", "Prescott")    # bit-reproducible fits (simulations/README.md)

import argparse
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from brian2 import prefs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pulse_inputs as pulses
from asyncresp.io import load_pickle
from asyncresp.paths import SUPP

OUT_DIR = SUPP / "pulse_amplitude"
SUMMARY = "sweep_summary_combined.pkl"
RATE, B, EL_EXC = 2, 0, -70                # Hz, pA, mV
PERIOD, T_CENTER, BINSIZE = 2000, 2500, 1  # ms
BASELINE_START = 500                       # ms
A_VALUES = [1, 5, 10, 25, 50]              # Hz
PARTS = {1: np.arange(-15, 1, 1), 2: np.arange(0, 16, 1)}   # Delta_EL (mV) of the two parts
SEEDS = list(range(10))
TARGET = "cython"


def fit_half_sinusoid(time_vec, rate_vec, t_center, period, t_baseline_start=BASELINE_START):
    """Least-squares amplitude of sin(pi (t - t_start) / (period / 2)) fitted to the
    baseline-subtracted rate over the pulse t_start = t_center - period / 4 <= t <= t_center
    + period / 4; the baseline is the mean rate for t_baseline_start <= t < t_start."""
    t_start = t_center - period / 4
    t_end = t_center + period / 4
    mask = (time_vec >= t_start) & (time_vec <= t_end)
    t_win = time_vec[mask]
    r_win = rate_vec[mask]
    if mask.sum() == 0:
        return 0.0, np.zeros_like(rate_vec)
    baseline_mask = (time_vec >= t_baseline_start) & (time_vec < t_start)
    baseline = rate_vec[baseline_mask].mean() if baseline_mask.sum() > 0 else 0.0
    r_win = r_win - baseline
    phase = (t_win - t_start) / (t_end - t_start) * np.pi
    X = np.sin(phase).reshape(-1, 1)
    amp, _, _, _ = np.linalg.lstsq(X, r_win, rcond=None)
    amp = float(amp[0])
    fitted = np.zeros_like(rate_vec)
    fitted[mask] = amp * np.sin(phase)
    return amp, fitted


def run_task(task):
    """Evoked E and I amplitudes of one run (A, Delta_EL, seed, target)."""
    A, delta_EL, seed, target = task
    prefs.codegen.target = target
    t, exc, inh, _ = pulses.simulate_pulse(B, RATE, EL_EXC - delta_EL, EL_EXC, A=A,
                                           period=PERIOD, t_center=T_CENTER, binsize=BINSIZE,
                                           seed=int(seed), shape="half_cycle")
    return (fit_half_sinusoid(t, exc, T_CENTER, PERIOD)[0],
            fit_half_sinusoid(t, inh, T_CENTER, PERIOD)[0])


def run_tasks(tasks, workers):
    if workers <= 1:
        return [run_task(t) for t in tasks]
    with Pool(workers) as pool:
        return pool.map(run_task, tasks, chunksize=1)


def summarise(per_delta, delta_ELs):
    """Mean and SD over seeds of the amplitudes, in the order of delta_ELs."""
    d = np.array(delta_ELs)
    return {"delta_ELs": d,
            "mean_exc": np.array([per_delta[x]["amp_exc"].mean() for x in d]),
            "std_exc": np.array([per_delta[x]["amp_exc"].std() for x in d]),
            "mean_inh": np.array([per_delta[x]["amp_inh"].mean() for x in d]),
            "std_inh": np.array([per_delta[x]["amp_inh"].std() for x in d])}


def combine(s1, s2):
    """Concatenate the two parts and sort by Delta_EL."""
    delta = np.concatenate((s1["delta_ELs"], s2["delta_ELs"]))
    order = np.argsort(delta)
    return {k: np.concatenate((s1[k], s2[k]))[order] for k in s1}


def sweep(amplitudes, seeds, workers, target):
    tasks = [(A, int(d), s, target) for A in amplitudes for part in (1, 2)
             for d in PARTS[part] for s in seeds]
    amps = iter(run_tasks(tasks, workers))
    per_seed, summary = {}, {}
    for A in amplitudes:
        parts = {}
        for part in (1, 2):
            res = {}
            for d in PARTS[part]:
                a = [next(amps) for _ in seeds]
                res[int(d)] = {"EL_inh": int(EL_EXC - d),
                               "amp_exc": np.array([x[0] for x in a]),
                               "amp_inh": np.array([x[1] for x in a])}
            per_seed[(A, part)] = res
            parts[part] = summarise(res, PARTS[part])
        summary[A] = combine(parts[1], parts[2])
    return per_seed, summary


def check(workers, target):
    """Re-run A = 1 Hz, Delta_EL = -15 mV (10 seeds) and compare with the stored summary."""
    stored = load_pickle(OUT_DIR / SUMMARY)[1]
    i = int(np.where(stored["delta_ELs"] == -15)[0][0])
    res = run_tasks([(1, -15, s, target) for s in SEEDS], workers)
    exc, inh = np.array([r[0] for r in res]), np.array([r[1] for r in res])
    for name, new in (("mean_exc", exc.mean()), ("std_exc", exc.std()),
                      ("mean_inh", inh.mean()), ("std_inh", inh.std())):
        old = stored[name][i]
        print(f"{name}: new {new:.12g}  stored {old:.12g}  rel. diff "
              f"{abs(new - old) / abs(old):.1e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--amplitudes", nargs="+", type=float, default=A_VALUES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--target", default=TARGET, choices=["cython", "numpy"],
                    help="Brian2 code-generation target (default: that of the stored file)")
    ap.add_argument("--check", action="store_true", help="re-run one stored point and compare")
    args = ap.parse_args()
    if args.check:
        check(args.workers, args.target)
        return
    amplitudes = [int(a) if float(a).is_integer() else a for a in args.amplitudes]
    per_seed, summary = sweep(amplitudes, args.seeds, args.workers, args.target)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, obj in ((SUMMARY, summary), ("sweep_amplitudes.pkl", per_seed)):
        with open(args.out / name, "wb") as f:
            pickle.dump(obj, f)
    print(f"wrote {args.out / SUMMARY}")


if __name__ == "__main__":
    main()

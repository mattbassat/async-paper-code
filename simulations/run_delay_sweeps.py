"""Frequency-response sweeps of the AdEx network with synaptic delays (supplementary figure)."""
import os
os.environ.setdefault("OPENBLAS_CORETYPE", "Prescott")    # bit-reproducible fits (simulations/README.md)

import argparse
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adex_impulse as impulse
from asyncresp.paths import SUPP

OUT_DIR = SUPP / "delays"
RATE, B, A_VALUE, EL_EXC = 4, 0, 1, -70     # Hz, pA, Hz, mV
DELAYS = [1, 2]                             # ms
FREQ_LOW = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30,
            35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]    # Hz
FREQ_HIGH = [120, 140, 160, 180, 200]                                  # Hz
DELTA_EL_FINAL = {1: [-15, -10, -5, 0, 5, 10, 15], 2: [-15, -10, -5, 0, 5]}   # mV
DELTA_EL_HIGH = [-15, -10, -5, 0, 5]                                   # mV
DELTA_EL_ADDED = 10                         # mV, added at all frequencies for the 1 ms delay
SEEDS = list(range(12, 22))

# Brian2 code-generation target with which the stored files were computed (see --check)
TARGET = "numpy"


def tasks_for_delay(delay, seeds):
    """(EL_inh, period, seed) of every run for one delay, in stored order."""
    final = [(EL_EXC - dEL, per, s) for dEL in DELTA_EL_FINAL[delay]
             for per in 1000 / np.array(FREQ_LOW) for s in seeds]
    high = [(EL_EXC - dEL, per, s) for dEL in DELTA_EL_HIGH
            for per in 1000 / np.array(FREQ_HIGH) for s in seeds]
    added = [] if delay != 1 else [(EL_EXC - DELTA_EL_ADDED, per, s)
                                   for per in 1000 / np.array(FREQ_LOW + FREQ_HIGH)
                                   for s in seeds]
    return final + high + added


def run_task(task):
    delay, EL_inh, period, seed, target = task
    impulse.set_target(target)
    fit = impulse.run_impulse_simulation(b_val=B, rate=RATE, EL_inh=EL_inh, EL_exc=EL_EXC,
                                         A=A_VALUE, period=period, seed=seed,
                                         phase_from_onset=True, delay_ms=delay)
    return impulse.impulse_record(B, RATE, EL_inh, EL_EXC, A_VALUE, period, fit)


def run_tasks(tasks, workers):
    if workers <= 1:
        return [run_task(t) for t in tasks]
    with Pool(workers) as pool:
        return pool.map(run_task, tasks, chunksize=1)


def check(target):
    """Re-run the first stored run of the 2 ms-delay file (Delta_EL = -15 mV, f = 0.5 Hz, seed
    12) and compare every stored field."""
    stored = impulse.load_records(OUT_DIR / "impulse_results_r4b0a1delay2_extended.pkl")[0]
    print(f"impulse_results_r4b0a1delay2_extended.pkl, first record (seed 12), target {target}:")
    new = run_task((2, stored["EL_inh"], stored["period"], SEEDS[0], target))
    return impulse.report_comparison(new, stored)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--delays", nargs="+", type=int, default=DELAYS, choices=DELAYS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--target", default=TARGET, choices=["numpy", "cython"],
                    help="Brian2 code-generation target (default: that of the stored files)")
    ap.add_argument("--check", action="store_true", help="re-run one stored run and compare")
    args = ap.parse_args()
    if args.check:
        check(args.target)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for delay in args.delays:
        tasks = [(delay, EL_inh, per, s, args.target)
                 for EL_inh, per, s in tasks_for_delay(delay, args.seeds)]
        results = run_tasks(tasks, args.workers)
        path = args.out / f"impulse_results_r{RATE}b{B}a{A_VALUE}delay{delay}_extended.pkl"
        with open(path, "wb") as f:
            pickle.dump(results, f)
        print(f"delay {delay} ms: {len(results)} runs -> {path}", flush=True)


if __name__ == "__main__":
    main()

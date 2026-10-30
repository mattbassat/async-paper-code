"""Fine-grained responsiveness to a slow input near the responsiveness peaks (supplementary
figure)."""
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

OUT_DIR = SUPP / "fine_grained"
EL_EXC, A_VALUE = -70, 1                   # mV, Hz
FREQUENCY = 0.5                            # Hz
PERIOD = 1000 / FREQUENCY                  # ms
SEEDS = list(range(12, 22))

# (nu_drive in Hz, b in pA, Delta_EL grid in mV) of each parameter set
PARAM_SETS = {
    "r2b0a1": (2, 0, np.linspace(-10, 0, 101)),
    "r4b0a1": (4, 0, np.linspace(-15, -5, 101)),
    "r4b60a1": (4, 60, np.linspace(-10, 0, 101)),
}

# Brian2 code-generation target with which the stored files were computed (see --check)
TARGET = "cython"


def filename(tag):
    return f"impulse_results_fg_slow_{tag}_final.pkl"


def run_task(task):
    rate, b, EL_inh, seed, target = task
    impulse.set_target(target)
    fit = impulse.run_impulse_simulation(b_val=b, rate=rate, EL_inh=EL_inh, EL_exc=EL_EXC,
                                         A=A_VALUE, period=PERIOD, seed=seed)
    return impulse.impulse_record(b, rate, EL_inh, EL_EXC, A_VALUE, PERIOD, fit)


def run_tasks(tasks, workers):
    if workers <= 1:
        return [run_task(t) for t in tasks]
    with Pool(workers) as pool:
        return pool.map(run_task, tasks, chunksize=1)


def check(target):
    """Re-run the first stored run of the r2b0a1 file (Delta_EL = -10 mV, seed 12) and compare
    every stored field."""
    tag = "r2b0a1"
    rate, b, _ = PARAM_SETS[tag]
    stored = impulse.load_records(OUT_DIR / filename(tag))[0]
    print(f"{filename(tag)}, first record (seed {SEEDS[0]}), target {target}:")
    new = run_task((rate, b, stored["EL_inh"], SEEDS[0], target))
    return impulse.report_comparison(new, stored)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", nargs="+", default=list(PARAM_SETS), choices=list(PARAM_SETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--target", default=TARGET, choices=["cython", "numpy"],
                    help="Brian2 code-generation target (default: that of the stored files)")
    ap.add_argument("--check", action="store_true", help="re-run one stored run and compare")
    args = ap.parse_args()
    if args.check:
        check(args.target)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for tag in args.sets:
        rate, b, delta_ELs = PARAM_SETS[tag]
        tasks = [(rate, b, EL_EXC - dEL, s, args.target) for dEL in delta_ELs
                 for s in args.seeds]
        results = run_tasks(tasks, args.workers)
        path = args.out / filename(tag)
        with open(path, "wb") as f:
            pickle.dump(results, f)
        print(f"{tag}: {len(results)} runs -> {path}", flush=True)


if __name__ == "__main__":
    main()

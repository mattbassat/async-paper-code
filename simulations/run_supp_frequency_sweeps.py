"""Supplementary frequency-response sweeps of the AdEx network: high drive, larger modulation
amplitude and a different excitatory leak reversal potential (supplementary figures)."""
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

DELTA_EL = [-15, -10, -5, 0, 5, 10, 15]                    # mV
FREQUENCIES = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30,
               35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]  # Hz
FREQUENCIES_NO_01 = FREQUENCIES[1:]                        # Hz, the EL_exc sweeps
SEEDS = list(range(12, 22))
PARAM_SETS = [("r2b0", 2, 0), ("r4b0", 4, 0), ("r4b60", 4, 60)]   # tag, nu_drive (Hz), b (pA)


def _file(rate, b, A, EL_exc, delta_EL, freqs):
    return dict(rate=rate, b=b, A=A, EL_exc=EL_exc, delta_EL=delta_EL, freqs=freqs)


# data set -> (output folder, {file name: parameters})
DATASETS = {
    "high_drive": ("high_drive", {
        "impulse_results_r8b0a1_final.pkl": _file(8, 0, 1, -70, DELTA_EL, FREQUENCIES),
        "impulse_results_r8b0a1suppr8_final.pkl": _file(8, 0, 1, -70, [-25, -20], FREQUENCIES),
    }),
    "amplitude_2hz": ("amplitude_2hz", {
        f"impulse_results_{tag}a2_final.pkl": _file(rate, b, 2, -70, DELTA_EL, FREQUENCIES)
        for tag, rate, b in PARAM_SETS
    }),
    "el_exc": ("el_exc", {
        f"impulse_results_{tag}a1EL{-EL_exc}_final.pkl":
            _file(rate, b, 1, EL_exc, DELTA_EL, FREQUENCIES_NO_01)
        for tag, rate, b in PARAM_SETS for EL_exc in (-65, -75)
    }),
}

# Brian2 target and form of the modulation (adex_impulse.py) with which each data set was
# computed (see --check)
SETTINGS = {
    "high_drive": dict(target="cython", phase_from_onset=True),
    "amplitude_2hz": dict(target="numpy", phase_from_onset=True),
    "el_exc": dict(target="numpy", phase_from_onset=True),
}

# stored record re-run by --check: (file name, index); the first record of each check file is
# at the lowest input frequency of its grid, where the two forms of the modulation differ most
CHECK_RECORD = {
    "high_drive": ("impulse_results_r8b0a1_final.pkl", 0),
    "amplitude_2hz": ("impulse_results_r2b0a2_final.pkl", 0),
    "el_exc": ("impulse_results_r2b0a1EL65_final.pkl", 0),
}


def tasks_for(dataset, spec, seeds):
    """(dataset, rate, b, EL_inh, EL_exc, A, period, seed) for one file, in stored order."""
    periods = 1000 / np.array(spec["freqs"])
    return [(dataset, spec["rate"], spec["b"], spec["EL_exc"] - dEL, spec["EL_exc"], spec["A"],
             per, seed)
            for dEL in spec["delta_EL"] for per in periods for seed in seeds]


def run_task(task):
    dataset, rate, b, EL_inh, EL_exc, A, period, seed = task
    settings = SETTINGS[dataset]
    impulse.set_target(settings["target"])
    fit = impulse.run_impulse_simulation(b_val=b, rate=rate, EL_inh=EL_inh, EL_exc=EL_exc, A=A,
                                         period=period, seed=seed,
                                         phase_from_onset=settings["phase_from_onset"])
    return impulse.impulse_record(b, rate, EL_inh, EL_exc, A, period, fit)


def run_tasks(tasks, workers):
    if workers <= 1:
        return [run_task(t) for t in tasks]
    with Pool(workers) as pool:
        return pool.map(run_task, tasks, chunksize=1)


def check(dataset):
    """Re-run one stored record of the data set and compare every stored field."""
    folder, files = DATASETS[dataset]
    name, idx = CHECK_RECORD[dataset]
    spec = files[name]
    task = tasks_for(dataset, spec, SEEDS)[idx]
    stored = impulse.load_records(SUPP / folder / name)[idx]
    print(f"{name}, record {idx} (Delta_EL = {stored['EL_exc'] - stored['EL_inh']} mV, "
          f"f = {1000 / stored['period']:g} Hz, seed {task[-1]}), "
          f"target {SETTINGS[dataset]['target']}:")
    return impulse.report_comparison(run_task(task), stored)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset", choices=list(DATASETS))
    ap.add_argument("--files", nargs="+", default=None,
                    help="file names to compute (default: all files of the data set)")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default: data/supp/<folder of the data set>)")
    ap.add_argument("--check", action="store_true", help="re-run one stored run and compare")
    args = ap.parse_args()
    if args.check:
        check(args.dataset)
        return
    folder, files = DATASETS[args.dataset]
    names = args.files or list(files)
    unknown = [n for n in names if n not in files]
    if unknown:
        ap.error(f"unknown file(s) for {args.dataset}: {unknown}; choose from {list(files)}")
    out = args.out or SUPP / folder
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        results = run_tasks(tasks_for(args.dataset, files[name], args.seeds), args.workers)
        with open(out / name, "wb") as f:
            pickle.dump(results, f)
        print(f"{args.dataset}: {len(results)} runs -> {out / name}", flush=True)


if __name__ == "__main__":
    main()

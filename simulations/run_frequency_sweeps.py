"""Frequency-response sweeps of the AdEx network over Delta_EL (Fig. 1c, Fig. 2e, Fig. 3a).

For each of the three parameter sets (nu_drive, b) = (2 Hz, 0 pA), (4 Hz, 0 pA), (4 Hz, 60 pA),
with modulation amplitude A = 1 Hz and EL_exc = -70 mV, the network of adex_impulse.py is
driven at every input frequency f of the grid below for Delta_EL = EL_exc - EL_inh in
{-15, -10, ..., +15} mV, ten seeds each (12-21).  Each run lasts 5 s; the modulation starts
at 2 s and the response is fitted on the last 2 s.

Writes, for tag in r2b0a1, r4b0a1, r4b60a1:
  data/sweeps/impulse_results_{tag}_final.pkl   main grid, 7 Delta_EL x 31 f x 10 seeds
                                                 (Fig. 1c, Fig. 2e, Fig. 3a)
  data/sweeps/impulse_results_{tag}_extra.pkl   intermediate Delta_EL = +-2.5, +-7.5, +-12.5 mV
                                                 at the slow-input frequency f = 0.5 Hz only,
                                                 10 seeds, with the seed in `seed_used` (Fig. 3a)
Each file is a list of result records (dicts) in the order Delta_EL, f, seed.

The main grids were computed with Brian2's NumPy target and the modulation
sin(2 pi (t - t_onset) f), the extra points with the Cython target and sin(2 pi t f) (identical
at f = 0.5 Hz up to rounding); PART_SETTINGS reproduces both.

Runtime: about 1.5 min per run on one core, i.e. about 160 core-hours for the three main
grids (6510 runs) and 5 core-hours for the extra points (180 runs).

    python simulations/run_frequency_sweeps.py [--sets r2b0a1 ...] [--part final extra]
                                               [--seeds 12 13 ...] [--workers N] [--out DIR]
    python simulations/run_frequency_sweeps.py --check     # re-run two stored runs (~3 min)
"""
import os
os.environ.setdefault("OPENBLAS_CORETYPE", "Prescott")    # see adex_impulse.py, "Reproducibility"

import argparse                                            # noqa: E402
import pickle                                              # noqa: E402
import sys                                                 # noqa: E402
from multiprocessing import Pool                           # noqa: E402
from pathlib import Path                                   # noqa: E402

import numpy as np                                         # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adex_impulse as impulse                                  # noqa: E402
from asyncresp.paths import SWEEPS                         # noqa: E402

SETS = {"r2b0a1": dict(rate=2, b=0), "r4b0a1": dict(rate=4, b=0), "r4b60a1": dict(rate=4, b=60)}
A_VALUE = 1                                  # modulation amplitude (Hz)
EL_EXC = -70                                 # mV
DELTA_EL = [-15, -10, -5, 0, 5, 10, 15]      # mV
FREQUENCIES = [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30,
               35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]   # Hz
EXTRA_DELTA_EL = [-12.5, -7.5, -2.5, 2.5, 7.5, 12.5]                     # mV
SLOW_F = 0.5                                 # Hz
SEEDS = list(range(12, 22))


# Brian2 target and form of the modulation (adex_impulse.py) with which each part was computed
PART_SETTINGS = {"final": dict(target="numpy", phase_from_onset=True),
                 "extra": dict(target="cython", phase_from_onset=False)}


def main_grid_tasks(rate, b, seeds):
    """(rate, b, EL_inh, EL_exc, A, period, seed, part) for the main grid, in stored order."""
    periods = 1000 / np.array(FREQUENCIES)
    return [(rate, b, EL_EXC - dEL, EL_EXC, A_VALUE, per, seed, "final")
            for dEL in DELTA_EL for per in periods for seed in seeds]


def extra_tasks(rate, b, seeds):
    """The intermediate Delta_EL values at the slow-input frequency, in stored order."""
    return [(rate, b, EL_EXC - dEL, EL_EXC, A_VALUE, 1000 / SLOW_F, seed, "extra")
            for dEL in EXTRA_DELTA_EL for seed in seeds]


def run_task(task):
    rate, b, EL_inh, EL_exc, A, period, seed, part = task
    settings = PART_SETTINGS[part]
    impulse.set_target(settings["target"])
    fit = impulse.run_impulse_simulation(b_val=b, rate=rate, EL_inh=EL_inh, EL_exc=EL_exc, A=A,
                                    period=period, seed=seed,
                                    phase_from_onset=settings["phase_from_onset"])
    rec = impulse.impulse_record(b, rate, EL_inh, EL_exc, A, period, fit)
    if part == "extra":
        rec["seed_used"] = seed
    return rec


def run_tasks(tasks, workers):
    if workers <= 1:
        return [run_task(t) for t in tasks]
    with Pool(workers) as pool:
        return pool.map(run_task, tasks, chunksize=1)


def check():
    """Re-run the first stored run of the r2b0a1 main grid (Delta_EL = -15 mV, f = 0.1 Hz,
    seed 12) and of the r2b0a1 extra points (Delta_EL = -12.5 mV, f = 0.5 Hz, seed 12), and
    compare every stored field."""
    stored = impulse.load_records(SWEEPS / "impulse_results_r2b0a1_final.pkl")[0]
    print("impulse_results_r2b0a1_final.pkl, first record (seed 12):")
    impulse.report_comparison(run_task((stored["rate"], stored["b_val"], stored["EL_inh"],
                                   stored["EL_exc"], stored["A"], stored["period"], SEEDS[0],
                                   "final")), stored)
    stored = impulse.load_records(SWEEPS / "impulse_results_r2b0a1_extra.pkl")[0]
    print("impulse_results_r2b0a1_extra.pkl, first record:")
    impulse.report_comparison(run_task((stored["rate"], stored["b_val"], stored["EL_inh"],
                                   stored["EL_exc"], stored["A"], stored["period"],
                                   stored["seed_used"], "extra")), stored)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    ap.add_argument("--part", nargs="+", default=["final", "extra"], choices=["final", "extra"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=SWEEPS)
    ap.add_argument("--check", action="store_true", help="re-run two stored runs and compare")
    args = ap.parse_args()
    if args.check:
        check()
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for tag in args.sets:
        rate, b = SETS[tag]["rate"], SETS[tag]["b"]
        for part in args.part:
            tasks = (main_grid_tasks if part == "final" else extra_tasks)(rate, b, args.seeds)
            results = run_tasks(tasks, args.workers)
            path = args.out / f"impulse_results_{tag}_{part}.pkl"
            with open(path, "wb") as f:
                pickle.dump(results, f)
            print(f"{tag} {part}: {len(results)} runs -> {path}", flush=True)


if __name__ == "__main__":
    main()

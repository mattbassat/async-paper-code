"""Frequency-response sweep of the Hodgkin-Huxley network (Fig. 5a-b).

For each inhibitory resting potential El_inh (El_exc = -70 mV, Delta_EL = El_exc - El_inh from
-20 to +10 mV in 2.5-mV steps), each of 34 input frequencies (0.5-200 Hz) and 10 seeds
(12-21), the network of network.run_driven is simulated for 5 s: afferent Poisson drive
nu_drive = 4 Hz, modulated by a 1-Hz-amplitude sinusoid from t = 2.5 s on.  Each record holds
the sinusoid fits of the E and I population rates (amp_*, phase_*, offset_*), the baseline
rates (average_*_rate), EL_inh, El_exc (mV), period (ms) and frequency (Hz).

Output: data/sweeps/impulse_results_hh_grid_final.pkl, a list of dicts.  The stored file was
simulated in two passes, the 5-mV grid (El_inh = -80, -75, ..., -50 mV) and then the
intermediate values (-77.5, ..., -52.5 mV); the list is the first pass followed by the second,
each sorted by (EL_inh, frequency).  The stored file lacks 10 of the 4420 records (El_inh =
-67.5 mV: one seed at 6 Hz and nine seeds at 7 Hz), which were lost when an interrupted run was
resumed; a full re-run produces all 4420.  Within each (EL_inh, frequency) group the stored
records are in completion order (the seed was not recorded); this script orders them by seed.

Runtime: about 1.5-2 min per simulation on one core (C++ standalone build included), i.e.
roughly 120-150 CPU hours for the full sweep.

    python simulations/hh/run_hh_frequency_sweep.py --workers 8
    python simulations/hh/run_hh_frequency_sweep.py --check
"""
import argparse
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import time
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from asyncresp.paths import SWEEPS  # noqa: E402

OUT_FILE = SWEEPS / "impulse_results_hh_grid_final.pkl"

EL_EXC = -70
EL_INH_PASSES = [
    np.arange(-80, -45, 5),                                      # 5-mV grid
    np.array([-77.5, -72.5, -67.5, -62.5, -57.5, -52.5]),        # intermediate values
]
FREQUENCIES = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               15, 20, 25, 30, 35, 40, 45, 50,
               55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
               120, 140, 160, 180, 200]
PERIODS = 1000.0 / np.array(FREQUENCIES)                         # ms
SEEDS = list(range(12, 22))


def run_single_sim(job):
    """One simulation; returns the record stored in the output list."""
    EL_inh, per_value, seed = job
    from simulations.hh.network import run_driven

    with tempfile.TemporaryDirectory(prefix="hh_build_") as tmp:
        results = run_driven(El_inh=EL_inh, osc_period=per_value, SEED=seed,
                             build_dir=os.path.join(tmp, "build"), El_exc=EL_EXC)
    results["EL_inh"] = EL_inh
    results["El_exc"] = EL_EXC
    results["period"] = per_value
    results["frequency"] = 1000.0 / per_value
    return results


def run_sweep(workers, seeds, out):
    results = []
    t0 = time.time()
    for el_range in EL_INH_PASSES:
        jobs = list(product(el_range, PERIODS, seeds))
        print(f"{len(jobs)} simulations (El_inh = {list(el_range)}), {workers} workers",
              flush=True)
        with mp.Pool(processes=workers) as pool:
            block = []
            for k, res in enumerate(pool.imap(run_single_sim, jobs), 1):
                block.append(res)
                print(f"  {k}/{len(jobs)}  El_inh={res['EL_inh']} f={res['frequency']:.1f} Hz"
                      f"  amp_exc={res['amp_exc']:.3f} Hz  ({(time.time()-t0)/60:.1f} min)",
                      flush=True)
        block.sort(key=lambda r: (r["EL_inh"], r["frequency"], r["period"]))
        results.extend(block)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(results, f)
    print(f"saved {len(results)} records to {out}")


def check():
    """Re-run El_inh = -80 mV, 200 Hz, seed 12 and compare with the stored records of that
    condition (the seed is not stored, so one of the 10 stored records must match).

    The baseline rates (plain means of the binned population rates) must be identical.  The
    sinusoid-fit fields come from a least-squares solve whose last bits depend on the
    LAPACK/BLAS build, so they are compared to a relative tolerance of 1e-12.
    """
    with open(OUT_FILE, "rb") as f:
        stored = pickle.load(f)
    el, freq = EL_INH_PASSES[0][0], 200
    per = PERIODS[FREQUENCIES.index(freq)]
    rec = run_single_sim((el, per, SEEDS[0]))
    exact = ("average_exc_rate", "average_inh_rate", "EL_inh", "El_exc", "period", "frequency")
    group = [r for r in stored if r["EL_inh"] == el and r["period"] == per]
    match = [r for r in group if r.keys() == rec.keys()
             and all(r[k] == rec[k] for k in exact)
             and all(np.isclose(r[k], rec[k], rtol=1e-12, atol=0) for k in r)]
    print(f"El_inh={el} mV, f={freq} Hz, seed {SEEDS[0]}")
    if not match:
        print("check FAILED: no stored record of this condition matches")
        return False
    r = match[0]
    for k in r:
        rel = abs(r[k] - rec[k]) / max(abs(r[k]), 1e-300)
        print(f"  {k:18s} stored {r[k]!r:24}  re-run {rec[k]!r:24}  rel. diff {rel:.1e}")
    print("check PASSED")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    ap.add_argument("--check", action="store_true",
                    help="re-run one stored simulation and compare with data/")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check() else 1)
    run_sweep(a.workers, a.seeds, a.out)


if __name__ == "__main__":
    main()

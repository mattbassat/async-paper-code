"""Frequency-response sweep of the Hodgkin-Huxley network (Fig. 5a-b)."""
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

from asyncresp.paths import SWEEPS

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
    condition (the seed is not stored, so one of the 10 stored records must match)."""
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

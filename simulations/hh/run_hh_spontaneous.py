"""Spontaneous activity of the Hodgkin-Huxley network for the LFP spectra of Fig. 5c."""
import argparse
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from asyncresp.paths import SPONTANEOUS

OUT_DIR = SPONTANEOUS / "hh"

DELTA_EL = [-20.0, -10.0, 0.0, 10.0]     # mV; El_inh = -50, -60, -70, -80 mV
SEEDS = list(range(300, 310))
EL_EXC_MV = -70.0
EXT_DRIVE = 4.0                          # Hz
TOTAL_MS = 10000.0
MON_DT_MS = 0.5                          # LFP-proxy sampling step
N_REC_EXC = 500
V_REF_MV = -60.0


def out_name(delta_el, seed):
    tag = f"{delta_el:+.1f}".replace("+", "p").replace("-", "m")
    return f"hh_spont_{tag}_{seed}.pkl"


def simulate(delta_el, seed):
    """One run; returns the record written to disk."""
    from simulations.hh.network import run_spontaneous

    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="hh_build_") as tmp:
        lfp_inh, rate_exc, rate_inh, t_ms = run_spontaneous(
            delta_el, seed, os.path.join(tmp, "build"), tstop=TOTAL_MS, El_exc=EL_EXC_MV,
            ext_drive=EXT_DRIVE, mon_dt=MON_DT_MS, n_rec_exc=N_REC_EXC, V_ref=V_REF_MV)
    return dict(delta_EL=delta_el, seed=seed, EL_exc=EL_EXC_MV, EL_inh=EL_EXC_MV - delta_el,
                ext_drive=EXT_DRIVE, tstop=TOTAL_MS, dt_lfp_ms=MON_DT_MS,
                n_rec_exc=N_REC_EXC, V_ref_mV=V_REF_MV,
                lfp_inh=lfp_inh.astype(np.float32),
                rate_exc=rate_exc.astype(np.float32),
                rate_inh=rate_inh.astype(np.float32),
                t_ms=t_ms.astype(np.float32),
                sim_time_s=time.time() - t0)


def run_one(job):
    delta_el, seed, out_dir = job
    rec = simulate(delta_el, seed)
    path = Path(out_dir) / out_name(delta_el, seed)
    with open(path, "wb") as f:
        pickle.dump(rec, f)
    return path, rec["sim_time_s"]


def check():
    """Re-run Delta_EL = -20 mV, seed 300 and compare with the stored file."""
    with open(OUT_DIR / out_name(-20.0, 300), "rb") as f:
        stored = pickle.load(f)
    rec = simulate(-20.0, 300)
    ok = all(np.array_equal(np.asarray(stored[k]), np.asarray(rec[k])) for k in stored)
    print(f"compared fields {sorted(stored)}: "
          + ("check PASSED: identical" if ok else "check FAILED: arrays differ"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="output folder")
    ap.add_argument("--check", action="store_true",
                    help="re-run one stored simulation and compare with data/")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check() else 1)
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(d, s, str(a.out)) for d in DELTA_EL for s in a.seeds]
    print(f"{len(jobs)} runs, {a.workers} workers", flush=True)
    with mp.Pool(a.workers) as pool:
        for path, st in pool.imap_unordered(run_one, jobs):
            print(f"  {path.name}  ({st:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

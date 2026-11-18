"""The Delta_gL frequency-response sweep of Fig. 1d repeated at other g_L,exc values (SI).

Same tasks as run_alt_parameter_sweeps.py (Delta_gL = -5 ... +5 nS, 29 frequencies, seeds 12-21),
with only g_L,exc changed. Every run is saved on its own in --work as soon as it finishes, so the
sweep can be stopped and restarted (finished runs are skipped); once all runs of a g_L,exc value
are there they are collected into data/supp/gl_shift/.

    python simulations/run_gl_shift_sweeps.py --workers 8
    python simulations/run_gl_shift_sweeps.py --check
"""
import os
os.environ.setdefault("OPENBLAS_CORETYPE", "Prescott")    # bit-reproducible fits (simulations/README.md)

import argparse
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_alt_parameter_sweeps as R
from asyncresp.paths import SUPP, OUTPUT

OUT_DIR = SUPP / "gl_shift"
GL_EXC_VALUES = [8, 12]                     # nS (10 nS is the Fig. 1d sweep)


def tasks(gle):
    R.GL_EXC = gle
    return R.delta_gl_tasks(R.SEEDS)


def out_name(gle):
    return f"impulse_results_r{R.RATE}b{R.B}a{R.A_VALUE}gLexc{gle}_delta_gL_final.pkl"


def work(args):
    work_dir, gle, idx, task = args
    path = Path(work_dir) / f"gl{gle}" / f"{idx:04d}.pkl"
    if path.exists():
        return None
    t0 = time.time()
    rec = R.run_task(task)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(rec, f, protocol=4)
    os.replace(tmp, path)
    return f"gL_exc={gle} run {idx} ({time.time() - t0:.0f} s)"


def collect(work_dir, gle, out_dir):
    files = [Path(work_dir) / f"gl{gle}" / f"{i:04d}.pkl" for i in range(len(tasks(gle)))]
    missing = sum(not f.exists() for f in files)
    if missing:
        print(f"gL_exc={gle}: {missing} runs still missing")
        return
    recs = [pickle.load(open(f, "rb")) for f in files]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / out_name(gle), "wb") as f:
        pickle.dump(recs, f, protocol=4)
    print(f"wrote {out_dir / out_name(gle)} ({len(recs)} records)")


def check(gle=8, idx=0):
    """Re-run one stored record and compare it with the stored file."""
    stored = pickle.load(open(OUT_DIR / out_name(gle), "rb"))[idx]
    new = R.run_task(tasks(gle)[idx])
    for k, v in stored.items():
        same = np.isclose(v, new[k], rtol=1e-10, atol=0) if isinstance(v, float) else v == new[k]
        print(f"  {k:14s} {v!r:>24} {new[k]!r:>24} {'ok' if same else 'DIFFERENT'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gl", type=int, nargs="+", default=GL_EXC_VALUES)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--work", default=str(OUTPUT / "gl_shift_runs"))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        check()
        return
    jobs = []
    for gle in a.gl:
        (Path(a.work) / f"gl{gle}").mkdir(parents=True, exist_ok=True)
        jobs += [(a.work, gle, i, t) for i, t in enumerate(tasks(gle))]
    # seed-major, so that partial results already hold complete curves for the first seeds
    jobs.sort(key=lambda j: (j[3][1][-1], j[1], j[2]))
    jobs = [j for j in jobs if not (Path(a.work) / f"gl{j[1]}" / f"{j[2]:04d}.pkl").exists()]
    print(f"{len(jobs)} runs to do", flush=True)
    with Pool(a.workers, maxtasksperchild=50) as pool:
        for msg in pool.imap_unordered(work, jobs, chunksize=1):
            if msg:
                print(msg, flush=True)
    for gle in a.gl:
        collect(a.work, gle, Path(a.out))


if __name__ == "__main__":
    main()

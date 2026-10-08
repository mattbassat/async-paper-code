"""Simulated acetylcholine-like cue: RS adaptation b ramped from 75 to 10 pA (Fig. 6c).

AdEx network with nu_drive = 2 Hz and Delta E_L = +5 mV.  The RS spike-triggered adaptation
increment is b = 75 pA until the cue (2.5 s), decreases linearly to b = 10 pA over 1 s and
stays there; the trial is analysed up to 5 s (1.2 s of extra simulated time follow, so that
time-frequency estimates near the end of the trial use real data).  One run per seed; the
spectrogram of Fig. 6c averages seeds 300-362 (63 runs).

Output: data/driven/ach_cue/b5_new_pair_b75_b10_<seed>.pkl (inhibitory-current LFP proxy and
population rates).  Brian2 numpy target, as for the stored runs.

Runtime: about 1 min per run on one core (63 runs: ~35 min with 2 workers).

    python simulations/run_lfp_ach_cue.py --workers 2
    python simulations/run_lfp_ach_cue.py --check
"""
import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asyncresp.paths import DRIVEN               # noqa: E402
from simulations import adex_lfp                 # noqa: E402

OUT_DIR = DRIVEN / "ach_cue"
DELTA_EL, NU_DRIVE = 5.0, 2.0
B_PRE, B_POST = 75.0, 10.0
SEEDS = list(range(300, 363))


def file_name(seed):
    return f"b5_new_pair_b{B_PRE:g}_b{B_POST:g}_{seed}.pkl"


def run_one(seed):
    return adex_lfp.run_ach_cue_sim(seed=seed, delta_EL=DELTA_EL, nu_0=NU_DRIVE,
                                    b_pre=B_PRE, b_post=B_POST, target="numpy", verbose=True)


def _job(args):
    seed, out_dir = args
    out = run_one(seed)
    path = Path(out_dir) / file_name(seed)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(out, f, protocol=4)
    os.replace(tmp, path)
    return path


def check(seed=300):
    """Re-run one stored seed and compare the stored fields exactly."""
    ref_path = OUT_DIR / file_name(seed)
    with open(ref_path, "rb") as f:
        ref = pickle.load(f)
    t0 = time.time()
    out = run_one(seed)
    ok = True
    for k, v in ref.items():
        same = np.array_equal(v, out[k]) if isinstance(v, np.ndarray) else v == out[k]
        print(f"  {k:10s} identical: {same}")
        ok &= bool(same)
    print(f"{ref_path.name}: {'EXACT MATCH' if ok else 'MISMATCH'} ({time.time() - t0:.0f} s)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--check", action="store_true",
                    help="re-run stored seed 300 and compare with the stored file")
    a = ap.parse_args()

    if a.check:
        raise SystemExit(0 if check() else 1)

    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, str(a.out)) for s in a.seeds]
    print(f"{len(jobs)} runs -> {a.out}", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for n, path in enumerate(pool.imap_unordered(_job, jobs), 1):
            print(f"[{n}/{len(jobs)}] {path.name}  {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

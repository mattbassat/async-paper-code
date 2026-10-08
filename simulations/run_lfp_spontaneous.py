"""Spontaneous AdEx network activity and its inhibitory-current LFP proxy (Fig. 1e, 1f, 6c).

Three data sets, selected by a sub-command; each run is 10 s of spontaneous activity under a
constant Poisson drive, one file per network and seed (seeds 300-309):

  r2b0    Fig. 1e: nu_drive = 2 Hz, b = 0, Delta E_L = -15, -5, +5, +15 mV
          -> data/spontaneous/adex_r2b0/r2b0_spontWS_{m15,m5,p5,p15}_0_<seed>.pkl
  params  Fig. 1f: nu_drive = 2 Hz, Delta E_L = 0, one secondary parameter changed at a time
          (Delta g_L = g_L,inh - g_L,exc = -5/+5 nS, b = 0/150 pA, Q_ee = 1.25/1.60 nS,
          Q_ei = 1.40/2.00 nS)
          -> data/spontaneous/adex_params/spont_<param>_<value>_<seed>.pkl
  ach     Fig. 6c (pre/post spectra): nu_drive = 2 Hz, Delta E_L = +5 mV, b = 75 and 10 pA
          -> data/spontaneous/ach/b5_new_dEL5_nu2_b{75,10}_<seed>.pkl

Brian2 code-generation targets are those that produced the stored files (numpy for r2b0 and
ach, cython for params); the runs are deterministic given the seed and the target.

Runtime: about 1-1.5 min per run on one core (r2b0: 40 runs, params: 80 runs, ach: 20 runs).

    python simulations/run_lfp_spontaneous.py r2b0 --workers 2
    python simulations/run_lfp_spontaneous.py params --check
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

from asyncresp.paths import SPONTANEOUS          # noqa: E402
from simulations import adex_lfp                 # noqa: E402

SEEDS = list(range(300, 310))

# r2b0 (Fig. 1e)
R2B0_DELTA_EL = [-15.0, -5.0, 5.0, 15.0]

# secondary parameters (Fig. 1f): the two ends of each responsiveness sweep
PARAM_VALUES = {
    "delta_gL": [-5.0, 5.0],      # g_L,inh = 10 nS + delta_gL
    "b":        [0.0, 150.0],     # pA
    "Qee":      [1.25, 1.6],      # nS
    "Qei":      [1.4, 2.0],       # nS
}

# ACh-like conditions (Fig. 6c)
ACH_DELTA_EL, ACH_NU = 5.0, 2.0
ACH_B_VALUES = [75.0, 10.0]


def _signed(x, fmt):
    return format(x, fmt).replace("+", "p").replace("-", "m")


# ---------------------------------------------------------------- one run per data set
def r2b0_name(delta_EL, seed):
    return f"r2b0_spontWS_{_signed(delta_EL, '+.1f').replace('.', '_')}_{seed}.pkl"


def run_r2b0(delta_EL, seed):
    return adex_lfp.run_spontaneous_sim(delta_EL=delta_EL, seed=seed, nu_0=2.0, b_post=0.0,
                                        target="numpy", verbose=True)


def param_name(param, value, seed):
    return f"spont_{param}_{_signed(value, '+.2f')}_{seed}.pkl"


def run_param(param, value, seed):
    kw = {}
    if param == "delta_gL":
        kw["gL_inh"] = 10.0 + value
    elif param == "b":
        kw["b_post"] = value
    else:
        kw[param] = value
    out = adex_lfp.run_spontaneous_sim_params(seed=seed, target="cython", verbose=True, **kw)
    out["param"], out["value"] = param, float(value)
    return out


def ach_name(b_val, seed):
    return f"b5_new_dEL5_nu{ACH_NU:g}_b{b_val:g}_{seed}.pkl"


def run_ach(b_val, seed):
    return adex_lfp.run_spontaneous_sim(delta_EL=ACH_DELTA_EL, seed=seed, nu_0=ACH_NU, b_post=b_val,
                                        target="numpy", verbose=True)


DATASETS = {
    # name: (sub-folder, list of condition tuples, run function, file-name function, check condition)
    "r2b0":   ("adex_r2b0", [(d,) for d in R2B0_DELTA_EL], run_r2b0, r2b0_name, (-15.0,)),
    "params": ("adex_params", [(p, v) for p, vals in PARAM_VALUES.items() for v in vals],
               run_param, param_name, ("Qei", 2.0)),
    "ach":    ("ach", [(b,) for b in ACH_B_VALUES], run_ach, ach_name, (10.0,)),
}


# ---------------------------------------------------------------- runner
def _job(args):
    dataset, cond, seed, out_dir = args
    _, _, run_fn, name_fn, _ = DATASETS[dataset]
    out = run_fn(*cond, seed)
    path = Path(out_dir) / name_fn(*cond, seed)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(out, f, protocol=4)
    os.replace(tmp, path)
    return path


def check(dataset, seed=300):
    """Re-run one stored condition and compare lfp_inh with the stored file (exact equality)."""
    sub, _, run_fn, name_fn, cond = DATASETS[dataset]
    ref_path = SPONTANEOUS / sub / name_fn(*cond, seed)
    with open(ref_path, "rb") as f:
        ref = pickle.load(f)
    t0 = time.time()
    out = run_fn(*cond, seed)
    ok = True
    for k, v in ref.items():
        if isinstance(v, np.ndarray):
            same = np.array_equal(v, out[k])
        else:
            same = v == out[k]
        print(f"  {k:10s} identical: {same}")
        ok &= bool(same)
    print(f"{ref_path.name}: {'EXACT MATCH' if ok else 'MISMATCH'} ({time.time() - t0:.0f} s)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=list(DATASETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (default: the data/spontaneous/ folder of the data set)")
    ap.add_argument("--check", action="store_true",
                    help="re-run one stored condition (seed 300) and compare with the stored file")
    a = ap.parse_args()

    if a.check:
        raise SystemExit(0 if check(a.dataset) else 1)

    sub, conds, _, _, _ = DATASETS[a.dataset]
    out_dir = a.out if a.out is not None else SPONTANEOUS / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(a.dataset, c, s, str(out_dir)) for c in conds for s in a.seeds]
    print(f"{a.dataset}: {len(jobs)} runs -> {out_dir}", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for n, path in enumerate(pool.imap_unordered(_job, jobs), 1):
            print(f"[{n}/{len(jobs)}] {path.name}  {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

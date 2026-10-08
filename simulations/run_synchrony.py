"""Pairwise spike-count correlations of the spontaneous AdEx network vs Delta E_L (Fig. 1i).

For nu_drive = 2 Hz, b = 0 and Delta E_L = -15, -10, ..., +15 mV, 10 networks (seeds 100-109)
are simulated for 5 s of spontaneous activity.  For each run, 200 FS and 800 RS neurons are
sampled; their spikes in the window 500-5000 ms are counted in 5 ms bins (bins anchored at
500 ms), neurons with fewer than 3 spikes (or constant counts) are excluded, and CC is the mean
Pearson correlation over all pairs of the remaining neurons (Renart et al. 2010; Ecker et al.
2010).  Per Delta E_L the file stores the mean (CC), SD (CC_sd) and per-seed values (CC_all).

Output: data/synchrony/regimes_cc_r2b0.pkl, a dict {Delta E_L (int, mV): {CC, CC_sd, CC_all,
EL_inh, EL_exc, delta_EL, b_val, rate}}.

The simulations use the Brian2 numpy target and are deterministic given the seed.  The neuron
sample, however, is drawn from an unseeded generator, as in the computation of the stored values,
so the CC of a re-run differs from the stored value by the sampling error of the 1000-neuron
sample (of order 1e-4 per run), although the spike trains are identical.  Runtime: about 1 min per run on one core (70 runs).

    python simulations/run_synchrony.py --workers 2
    python simulations/run_synchrony.py --check
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

from asyncresp.paths import SYNCHRONY            # noqa: E402
from simulations import adex_lfp                 # noqa: E402

OUT_FILE = SYNCHRONY / "regimes_cc_r2b0.pkl"
DELTA_EL = [-15, -10, -5, 0, 5, 10, 15]
SEEDS = list(range(100, 110))
NU_DRIVE, B_VAL, EL_EXC = 2, 0, -70
T_START_MS, T_END_MS = 500.0, 5000.0


# ---------------------------------------------------------------- spike-count correlation
def _spike_times_ms(train):
    """Spike times in ms from a Brian2 quantity array (or a plain array already in ms)."""
    try:
        from brian2 import Quantity, ms as _ms
        if isinstance(train, Quantity):
            return np.asarray(train / _ms, dtype=float)
    except Exception:
        pass
    return np.asarray(train, dtype=float)


def compute_cc(inh_spike_trains, exc_spike_trains, bin_width_ms=5.0, t_start_ms=T_START_MS,
               t_end_ms=T_END_MS, n_sample_inh=200, n_sample_exc=800, min_spikes=3, rng=None):
    """Mean pairwise Pearson correlation of binned spike counts.

    n_sample_inh FS and n_sample_exc RS neurons are drawn without replacement (rng: a numpy
    Generator or seed; None = unseeded) and binned once on bins anchored at t_start_ms, so that
    all pairs share the same bins; neurons with fewer than min_spikes spikes in the window or
    constant counts are excluded; all pairs of the remaining neurons are averaged.
    """
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    bins = np.arange(t_start_ms, t_end_ms + bin_width_ms, bin_width_ms)
    counts = []
    for trains, n_sample in ((inh_spike_trains, n_sample_inh), (exc_spike_trains, n_sample_exc)):
        keys = list(trains.keys())
        if not keys:
            continue
        idx = rng.choice(len(keys), size=min(n_sample, len(keys)), replace=False)
        for j in idx:
            st = _spike_times_ms(trains[keys[j]])
            st = st[(st >= t_start_ms) & (st < t_end_ms)]
            if len(st) < min_spikes:
                continue
            c, _ = np.histogram(st, bins=bins)
            if c.std() <= 0:
                continue
            counts.append(c)
    if len(counts) < 2:
        return np.nan
    R = np.corrcoef(np.asarray(counts, dtype=float))
    r_all = R[np.triu_indices_from(R, k=1)]
    good = np.isfinite(r_all)
    return float(np.mean(r_all[good])) if good.any() else np.nan


# ---------------------------------------------------------------- runs
def run_one(args):
    delta_EL, seed = args
    sim = adex_lfp.run_spontaneous_spike_sim(delta_EL=delta_EL, seed=seed, nu_0=NU_DRIVE,
                                             b_val=B_VAL, total_ms=T_END_MS, t_trim_ms=T_START_MS)
    cc = compute_cc(sim["inh"], sim["exc"])
    print(f"  dEL={delta_EL:+d} seed={seed}: CC={cc:.5f}, rates {sim['mean_exc_rate']:.6f}/"
          f"{sim['mean_inh_rate']:.6f} Hz ({sim['sim_time_s']:.0f} s)", flush=True)
    return delta_EL, seed, cc


def entry(delta_EL, cc_all):
    cc_all = np.asarray(cc_all, dtype=float)
    return dict(CC=np.nanmean(cc_all), CC_sd=np.nanstd(cc_all), CC_all=cc_all,
                EL_inh=EL_EXC - delta_EL, EL_exc=EL_EXC, delta_EL=delta_EL, b_val=B_VAL, rate=NU_DRIVE)


def check(delta_EL=0, seed=SEEDS[0]):
    """Re-run one stored network and compare its CC with the stored per-seed value."""
    with open(OUT_FILE, "rb") as f:
        stored = pickle.load(f)[delta_EL]
    ref = stored["CC_all"][SEEDS.index(seed)]
    t0 = time.time()
    _, _, cc = run_one((delta_EL, seed))
    diff = abs(cc - ref)
    # the neuron sample is random: compare with the seed-to-seed spread of the stored values
    ok = diff < 3 * stored["CC_sd"]
    print(f"dEL={delta_EL:+d} seed={seed}: stored CC={ref:.6f}, re-run CC={cc:.6f}, |diff|={diff:.2e} "
          f"(seed-to-seed SD {stored['CC_sd']:.2e}) -> {'CONSISTENT' if ok else 'INCONSISTENT'} "
          f"({time.time() - t0:.0f} s)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    ap.add_argument("--check", action="store_true",
                    help="re-run Delta E_L = 0, seed 100 and compare its CC with the stored value")
    a = ap.parse_args()

    if a.check:
        raise SystemExit(0 if check() else 1)

    jobs = [(d, s) for d in DELTA_EL for s in a.seeds]
    print(f"{len(jobs)} runs -> {a.out}", flush=True)
    t0 = time.time()
    cc = {}
    with Pool(a.workers) as pool:
        for n, (d, s, c) in enumerate(pool.imap_unordered(run_one, jobs), 1):
            cc[(d, s)] = c
            print(f"[{n}/{len(jobs)}] {(time.time() - t0) / 60:.1f} min", flush=True)
    results = {d: entry(d, [cc[(d, s)] for s in a.seeds]) for d in DELTA_EL}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(results, f, protocol=4)
    os.replace(tmp, a.out)
    for d in DELTA_EL:
        print(f"  dEL={d:+3d}: CC = {results[d]['CC']:.5f} +/- {results[d]['CC_sd']:.5f}")


if __name__ == "__main__":
    main()

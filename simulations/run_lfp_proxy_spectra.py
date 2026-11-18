"""Spontaneous spectra of Fig. 1e-f for alternative LFP proxies and the population rates (SI).

Re-runs the spontaneous simulations of run_lfp_spontaneous.py (same networks, seeds 300-309 and
code-generation targets, so the inhibitory-current proxy is identical to the stored Fig. 1
traces) with all spikes recorded, and stores the Welch spectrum of each signal for every run.

    python simulations/run_lfp_proxy_spectra.py --workers 2
"""
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
import run_lfp_spontaneous as SP
from asyncresp.analysis import welch_psd
from asyncresp.paths import OUTPUT, SUPP
from kernel_lfp import kernel_lfp

OUT_FILE = SUPP / "lfp_proxies" / "spectra.pkl"
SEEDS = list(range(300, 310))
WINDOW_MS = (2000.0, 10000.0)
DT_MS = 0.5
SIGNALS = ["lfp_inh", "lfp_exc", "absmag", "kernel", "rate_exc", "rate_inh"]
# (data set, condition) as in run_lfp_spontaneous.py
CONDITIONS = [("r2b0", (d,)) for d in SP.R2B0_DELTA_EL] + \
             [("params", (p, v)) for p, vals in SP.PARAM_VALUES.items() for v in vals]


def simulate(dataset, cond, seed):
    import adex_lfp
    if dataset == "r2b0":
        return adex_lfp.run_spontaneous_sim(delta_EL=cond[0], seed=seed, nu_0=2.0, b_post=0.0,
                                            target="numpy", record_spikes=True)
    param, value = cond
    kw = {"delta_gL": dict(gL_inh=10.0 + value), "b": dict(b_post=value, nu_0=4.0)}.get(
        param, {param: value})
    return adex_lfp.run_spontaneous_sim_params(seed=seed, target="cython", record_spikes=True, **kw)


def signals(o):
    n = int(round(o["total_ms"] / DT_MS))
    def rate(sp, N):
        h = np.histogram(sp[1], bins=n, range=(0, o["total_ms"]))[0]
        return (h / N / (DT_MS * 1e-3)).astype(np.float32)
    return dict(lfp_inh=o["lfp_inh"], lfp_exc=o["lfp_exc"], absmag=o["lfp_absmag"],
                kernel=kernel_lfp(o["spikes_exc"], o["spikes_inh"], 8000, 2000, o["total_ms"], DT_MS),
                rate_exc=rate(o["spikes_exc"], 8000), rate_inh=rate(o["spikes_inh"], 2000))


def job(args):
    work, dataset, cond, seed = args
    path = Path(work) / f"{dataset}_{'_'.join(map(str, cond))}_{seed}.pkl"
    if not path.exists():
        o = simulate(dataset, cond, seed)
        spec = {}
        for k, x in signals(o).items():
            f, p = welch_psd(x, DT_MS, window=WINDOW_MS)
            spec[k] = p
        spec["f"] = f
        with open(path, "wb") as fh:
            pickle.dump(spec, fh, protocol=4)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--work", default=str(OUTPUT / "lfp_proxy_runs"))
    a = ap.parse_args()
    Path(a.work).mkdir(parents=True, exist_ok=True)
    jobs = [(a.work, d, c, s) for s in SEEDS for d, c in CONDITIONS]
    with Pool(a.workers) as pool:
        paths = pool.map(job, jobs, chunksize=1)
    out = {}
    for (_, d, c, s), p in zip(jobs, paths):
        spec = pickle.load(open(p, "rb"))
        entry = out.setdefault((d, c), {"f": spec["f"], **{k: [] for k in SIGNALS}})
        for k in SIGNALS:
            entry[k].append(spec[k].astype(np.float32))
    for entry in out.values():
        for k in SIGNALS:
            entry[k] = np.array(entry[k])                 # (seed, frequency)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "wb") as fh:
        pickle.dump(dict(seeds=SEEDS, signals=SIGNALS, spectra=out), fh, protocol=4)
    print(f"wrote {OUT_FILE}")


if __name__ == "__main__":
    main()

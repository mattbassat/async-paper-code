"""Single driven runs of the AdEx network with full time series (Fig. 1a, Fig. 2b, Fig. 3c)."""
import os
os.environ.setdefault("OPENBLAS_CORETYPE", "Prescott")    # bit-reproducible fits (simulations/README.md)

import argparse
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adex_impulse as impulse
from asyncresp.paths import DATA

TARGET = "cython"

FIG1 = dict(b_val=0, rate=4, EL_inh=-60, EL_exc=-70, A=1, period=500.0, oscillation_onset=2000,
            seed=12)
FIG1_FILE = "driven/fig1_example_sim.pkl"

SLOW_F = 0.5                                               # Hz
FIG3 = dict(b_val=0, rate=2, EL_exc=-70.0, A=1, period=1000.0 / SLOW_F, oscillation_onset=2000,
            tot_time=7000)
FIG3_NETS = {"m15": -15.0, "m5": -5.0, "p5": 5.0}          # file tag -> Delta_EL (mV)
FIG3_SEEDS = list(range(12, 22))
FIG3_DROP = ("spk_exc_t", "spk_exc_i", "spk_inh_t", "spk_inh_i", "t_lfp_ms", "lfp_inh", "lfp_exc")
FIG3_FILE = "driven/slow_input_examples/fig3_example_{tag}_{seed}.pkl"

BURST = dict(b_val=60, rate=4, EL_inh=-70, EL_exc=-70, A=1.5, period=500.0,
             oscillation_onset=1500, oscillation_end=2500, tot_time=4000, seed=12)
BURST_FILE = "mean_field/mf_burst_sim.pkl"


def traced_run(p):
    """Run with traces; the record starts with the parameters and fit of a sweep record."""
    impulse.set_target(TARGET)
    out = impulse.run_impulse_simulation(return_traces=True, **p)
    rec = impulse.impulse_record(p["b_val"], p["rate"], p["EL_inh"], p["EL_exc"], p["A"], p["period"],
                            out)
    rec.update({k: v for k, v in out.items() if k not in rec})
    return rec


def fig1_example():
    return traced_run(FIG1)


def fig3_example(tag, seed):
    delta_EL = FIG3_NETS[tag]
    p = dict(FIG3, EL_inh=FIG3["EL_exc"] - delta_EL, seed=seed)
    rec = traced_run(p)
    for k in FIG3_DROP:
        rec.pop(k)
    rec["delta_EL"] = delta_EL
    rec["tag"], rec["seed"] = tag, seed
    return rec


def burst_example():
    rec = traced_run(BURST)
    rec["delta_EL"] = BURST["EL_exc"] - BURST["EL_inh"]
    return rec


def _job(job):
    kind, args = job
    if kind == "fig1":
        return fig1_example()
    if kind == "burst":
        return burst_example()
    return fig3_example(*args)


def _path(out, job):
    kind, args = job
    if kind == "fig1":
        return out / FIG1_FILE
    if kind == "burst":
        return out / BURST_FILE
    return out / FIG3_FILE.format(tag=args[0], seed=args[1])


def check():
    """Re-run one Fig. 3c example (Delta_EL = -5 mV, seed 12) and compare all stored fields."""
    stored = impulse.load_records(DATA / FIG3_FILE.format(tag="m5", seed=12))
    impulse.report_comparison(fig3_example("m5", 12), stored)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--which", nargs="+", default=["fig1", "fig3", "burst"],
                    choices=["fig1", "fig3", "burst"])
    ap.add_argument("--seeds", nargs="+", type=int, default=FIG3_SEEDS, help="Fig. 3c seeds")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=DATA, help="data folder")
    ap.add_argument("--check", action="store_true", help="re-run one stored example and compare")
    args = ap.parse_args()
    if args.check:
        check()
        return
    jobs = []
    for kind in args.which:
        if kind == "fig3":
            jobs += [("fig3", (tag, seed)) for tag in FIG3_NETS for seed in args.seeds]
        else:
            jobs.append((kind, ()))
    def save(job, rec):
        path = _path(args.out, job)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(rec, f)
        print(f"{path}: amp_exc = {rec['amp_exc']:.3f} Hz", flush=True)

    if args.workers <= 1:
        for job in jobs:
            save(job, _job(job))
    else:
        with Pool(args.workers) as pool:
            for job, rec in zip(jobs, pool.imap(_job, jobs)):
                save(job, rec)


if __name__ == "__main__":
    main()

"""Frequency-response sweeps over other neuromodulated parameters (Fig. 1d).

The network of adex_impulse.py (A = 1 Hz, seeds 12-21, 5 s runs, modulation from 2 s, fit on
the last 2 s) is driven at every input frequency f while one parameter is varied away from
the reference network (nu_drive = 2 Hz, b = 0 pA, EL_exc = EL_inh = -70 mV):

  changing_b  spike-triggered adaptation b of the RS cells, 0 ... 150 pA in steps of 25 pA
              -> data/sweeps/impulse_results_changing_b_final.pkl
  delta_gL    leak conductance of the FS cells, g_L,inh = g_L,exc - Delta g_L with
              g_L,exc = 10 nS and Delta g_L = -5 ... +5 nS (field delta_gL = gL_exc - gL_inh)
              -> data/sweeps/impulse_results_r2b0a1gLexc10_delta_gL_final.pkl
  Qee, Qei    excitatory quantal conductance onto RS cells (Qee, also used for the afferent
              synapses onto RS cells) or onto FS cells (Qei, also used for the afferent synapses
              onto FS cells), in nS.  These sweeps were run in two phases of input frequency,
              the slow input f = 0.5 Hz ("slow") and the remaining 28 frequencies ("rest"),
              with later additions of parameter values ("*_extra"):
              -> data/sweeps/impulse_results_r2b0a1_Qee_{slow,slow_extra,rest,rest_extra}.pkl
              -> data/sweeps/impulse_results_r2b0a1_Qei_{slow,slow_extra,rest}.pkl

The b and Delta g_L sweeps use the modulation sin(2 pi (t - t_onset) f) (see adex_impulse.py).
The b sweep was computed with Brian2's Cython target, the other sweeps with the NumPy target
(TARGETS below).
Records are written in loop order (b or Delta g_L, then f, then seed; for the Q sweeps f,
then the parameter value, then seed).  The stored "rest" files of the Q sweeps were collected
from parallel workers in completion order, so their record order differs from the loop order.

Runtime: about 1.5 min per run on one core: changing_b 2030 runs, delta_gL 3190 runs,
Qee 2320 runs, Qei 2030 runs (roughly 50-80 core-hours per family).

    python simulations/run_alt_parameter_sweeps.py [--sweeps changing_b delta_gL Qee Qei]
                                                   [--seeds 12 13 ...] [--workers N] [--out DIR]
    python simulations/run_alt_parameter_sweeps.py --check  # re-run one stored run (~2 min)
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

RATE, B, A_VALUE, EL_EXC = 2, 0, 1, -70
FREQ_FULL = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30,
             35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]    # Hz
FREQ_SLOW = [0.5]
FREQ_REST = [f for f in FREQ_FULL if f not in FREQ_SLOW]
SEEDS = list(range(12, 22))

B_RANGE = np.arange(0, 175, 25)                 # pA
GL_EXC = 10                                     # nS
DELTA_GL_RANGE = np.arange(-5, 6, 1)            # nS
SYN_DEFAULTS = dict(Qee=1.5, Qei=1.5, Qie=5.0, Qii=5.0, Tee=5.0, Tei=5.0, Tie=5.0, Tii=5.0)
# (file suffix, parameter values, frequencies) of each phase of the quantal-conductance sweeps
SYN_PHASES = {
    "Qee": [("slow", [1.25, 1.30, 1.35, 1.40, 1.45, 1.50], FREQ_SLOW),
            ("slow_extra", [1.55, 1.60], FREQ_SLOW),
            ("rest", [1.25, 1.30, 1.35, 1.40, 1.45, 1.50], FREQ_REST),
            ("rest_extra", [1.55, 1.60], FREQ_REST)],
    "Qei": [("slow", [1.5, 1.6, 1.7, 1.8, 1.9, 2.0], FREQ_SLOW),
            ("slow_extra", [1.4], FREQ_SLOW),
            ("rest", [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], FREQ_REST)],
}
SWEEPS_ALL = ["changing_b", "delta_gL", "Qee", "Qei"]
TARGETS = {"b": "cython", "gL": "numpy", "syn": "numpy"}      # Brian2 target per sweep kind


# ---- task lists: (kind, arguments) in the loop order of each sweep -------------------------

def changing_b_tasks(seeds):
    periods = 1000 / np.array(FREQ_FULL)
    EL_inh = EL_EXC - 0                         # Delta_EL = 0
    return [("b", (b, RATE, EL_inh, EL_EXC, A_VALUE, per, seed))
            for b in B_RANGE for per in periods for seed in seeds]


def delta_gl_tasks(seeds):
    periods = 1000 / np.array(FREQ_FULL)
    return [("gL", (B, RATE, EL_EXC, EL_EXC, GL_EXC - dgl, GL_EXC, A_VALUE, per, seed))
            for dgl in DELTA_GL_RANGE for per in periods for seed in seeds]


def syn_tasks(param, values, freqs, seeds):
    freqs = sorted(freqs)                       # slowest first
    periods = 1000 / np.array(freqs)
    tasks = []
    for per in periods:
        for v in values:
            syn = dict(SYN_DEFAULTS)
            syn[param] = v
            tasks += [("syn", (B, RATE, syn, A_VALUE, per, seed)) for seed in seeds]
    return tasks


def run_task(task):
    kind, args = task
    impulse.set_target(TARGETS[kind])
    if kind == "b":
        b, rate, EL_inh, EL_exc, A, per, seed = args
        fit = impulse.run_impulse_simulation(b, rate, EL_inh, EL_exc, A=A, period=per, seed=seed,
                                        phase_from_onset=True)
        return impulse.impulse_record(b, rate, EL_inh, EL_exc, A, per, fit)
    if kind == "gL":
        b, rate, EL_inh, EL_exc, gL_inh, gL_exc, A, per, seed = args
        fit = impulse.run_impulse_simulation(b, rate, EL_inh, EL_exc, A=A, period=per, seed=seed,
                                        gL_inh=gL_inh, gL_exc=gL_exc, phase_from_onset=True)
        return impulse.gl_record(b, rate, EL_inh, EL_exc, gL_inh, gL_exc, A, per, fit)
    b, rate, syn, A, per, seed = args
    fit = impulse.run_impulse_simulation(b, rate, EL_EXC, EL_EXC, A=A, period=per, seed=seed, **syn)
    return impulse.synapse_record(b, rate, syn, A, per, fit)


def run_tasks(tasks, workers):
    if workers <= 1:
        return [run_task(t) for t in tasks]
    with Pool(workers) as pool:
        return pool.map(run_task, tasks, chunksize=1)


def outputs(sweep, seeds):
    """(file name, task list) of every output file of one sweep family."""
    if sweep == "changing_b":
        return [("impulse_results_changing_b_final.pkl", changing_b_tasks(seeds))]
    if sweep == "delta_gL":
        return [(f"impulse_results_r{RATE}b{B}a{A_VALUE}gLexc{GL_EXC}_delta_gL_final.pkl",
                 delta_gl_tasks(seeds))]
    return [(f"impulse_results_r{RATE}b{B}a{A_VALUE}_{sweep}_{phase}.pkl",
             syn_tasks(sweep, values, freqs, seeds))
            for phase, values, freqs in SYN_PHASES[sweep]]


def check():
    """Re-run the first stored record of the Qee slow-input sweep and compare all fields."""
    stored = impulse.load_records(SWEEPS / "impulse_results_r2b0a1_Qee_slow.pkl")[0]
    syn = {k: stored[k] for k in SYN_DEFAULTS}
    rec = run_task(("syn", (stored["b_val"], stored["rate"], syn, stored["A"],
                            stored["period"], SEEDS[0])))
    impulse.report_comparison(rec, stored)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweeps", nargs="+", default=SWEEPS_ALL, choices=SWEEPS_ALL)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=SWEEPS)
    ap.add_argument("--check", action="store_true", help="re-run one stored run and compare")
    args = ap.parse_args()
    if args.check:
        check()
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for sweep in args.sweeps:
        for name, tasks in outputs(sweep, args.seeds):
            results = run_tasks(tasks, args.workers)
            with open(args.out / name, "wb") as f:
                pickle.dump(results, f)
            print(f"{name}: {len(results)} runs", flush=True)


if __name__ == "__main__":
    main()

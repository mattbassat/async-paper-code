"""Baseline (spontaneous) activity metrics of the AdEx network vs Delta_EL (supplementary
figure)."""
import os
os.environ.setdefault("OPENBLAS_CORETYPE", "Prescott")    # bit-reproducible fits (simulations/README.md)

import argparse
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path
from random import sample

import numpy as np
import pandas as pd
from brian2 import (NeuronGroup, PoissonGroup, PopulationRateMonitor, SpikeMonitor,
                    StateMonitor, Synapses, defaultclock, prefs, run, start_scope)
from brian2 import seed as brian_seed
from brian2.units import Hz, amp, mV, ms, nA, nS, pA, pF, second, siemens, volt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asyncresp.paths import SUPP

OUT_DIR = SUPP / "baseline"
METRICS_CSV, CURRENTS_CSV = "baseline_metrics.csv", "current_ratios.csv"
PARAM_SETS = {"r2b0": dict(rate=2, b_val=0),
              "r4b0": dict(rate=4, b_val=0),
              "r4b60": dict(rate=4, b_val=60)}
EL_EXC = -70                                # mV
BIN_SIZE = 5                                # population-rate bin (ms)
DELTA_EL = {"metrics": np.arange(-15, 20, 5), "currents": np.arange(-15, 16, 5)}   # mV
SEEDS = {"metrics": list(range(12, 22)), "currents": list(range(10))}

# Brian2 code-generation target with which the stored files of each part were computed
# (see --check)
TARGETS = {"metrics": "numpy", "currents": "cython"}


# --- spike-train statistics ---
def compute_cv(inh_spike_trains, exc_spike_trains, inh_samples=200, exc_samples=800):
    """Mean ISI coefficient of variation over randomly sampled FS and RS cells with more than
    two spikes."""
    sampled_inh = sample(list(inh_spike_trains.keys()), inh_samples)
    sampled_exc = sample(list(exc_spike_trains.keys()), exc_samples)
    cvs_inh = []
    for neuron in sampled_inh:
        t = inh_spike_trains[neuron]
        if len(t) > 2:
            isi = np.diff(t) / ms
            cvs_inh.append(np.std(isi) / np.mean(isi))
    cvs_inh = [cv for cv in cvs_inh if not np.isnan(cv)]
    cvs_exc = []
    for neuron in sampled_exc:
        t = exc_spike_trains[neuron]
        if len(t) > 2:
            isi = np.diff(t) / ms
            cvs_exc.append(np.std(isi) / np.mean(isi))
    cvs_exc = [cv for cv in cvs_exc if not np.isnan(cv)]
    return np.mean(cvs_inh + cvs_exc)


def compute_cc(inh_spike_trains, exc_spike_trains):
    """Mean Pearson correlation of 5 ms spike counts over 1000 random pairs of cells (both
    populations pooled; pairs with fewer than three spikes in either cell are skipped)."""
    ccs = []
    trains = {**inh_spike_trains, **exc_spike_trains}
    idx = list(trains.keys())
    for _ in range(1000):
        n1, n2 = sample(idx, 2)
        s1, s2 = trains[n1], trains[n2]
        if len(s1) > 2 and len(s2) > 2:
            max_time = np.max([s1[-1], s2[-1]]) * second
            bin_width = 5 * ms
            bins = np.arange(0, max_time + bin_width, bin_width)
            c1, _ = np.histogram(s1, bins=bins)
            c2, _ = np.histogram(s2, bins=bins)
            if len(c1) == len(c2):
                m1, m2 = np.mean(c1), np.mean(c2)
                cov = np.mean((c1 - m1) * (c2 - m2))
                sd1, sd2 = np.std(c1), np.std(c2)
                if sd1 > 0 and sd2 > 0:
                    ccs.append(cov / (sd1 * sd2))
    return np.mean(ccs) if ccs else np.nan


def bin_array(array, BIN, time_array):
    """Average `array` (sampled at `time_array`, ms) in consecutive bins of BIN ms."""
    N0 = int(BIN / (time_array[1] - time_array[0]))
    N1 = int((time_array[-1] - time_array[0]) / BIN)
    return array[:N0 * N1].reshape((N1, N0)).mean(axis=1)


# --- network ---
EQS = {
    "metrics": """
    dvm/dt=(gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT)-GsynE*(vm-Ee)-GsynI*(vm-Ei)+I-w)/C : volt (unless refractory)
    dw/dt=(a*(vm-EL)-w)/tauw : amp
    dGsynI/dt = -GsynI/TsynI : siemens
    dGsynE/dt = -GsynE/TsynE : siemens
    TsynI:second
    TsynE:second
    Vr:volt
    b:amp
    DeltaT:volt
    Vcut:volt
    VT:volt
    EL:volt
    """,
    "currents": """
    dvm/dt=(gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT)-(GsynE_aff+GsynE_rec)*(vm-Ee)-GsynI*(vm-Ei)+I-w)/C : volt (unless refractory)
    dw/dt=(a*(vm-EL)-w)/tauw : amp
    dGsynI/dt    = -GsynI/TsynI         : siemens
    dGsynE_aff/dt = -GsynE_aff/TsynE   : siemens
    dGsynE_rec/dt = -GsynE_rec/TsynE   : siemens
    TsynI:second
    TsynE:second
    Vr:volt
    b:amp
    DeltaT:volt
    Vcut:volt
    VT:volt
    EL:volt
    """,
}


def run_baseline_simulation(part, b_val, rate, EL_inh, EL_exc, bin_size=BIN_SIZE, seed=12):
    """One 5 s run of the undriven network; returns the record of `part` ('metrics' or
    'currents'). b_val: adaptation increment of the RS cells (pA); rate: nu_drive (Hz);
    EL_inh, EL_exc: leak reversal potentials (mV); bin_size: population-rate bin (ms)."""
    split = part == "currents"
    start_scope()
    np.random.seed(seed)
    brian_seed(seed)

    DT = 0.1                                # ms
    defaultclock.dt = DT * ms
    N1 = 2000                               # FS cells
    N2 = 8000                               # RS cells
    TotTime = 5000                          # ms
    duration = TotTime * ms
    C = 200 * pF
    gL = 10 * nS
    tauw = 500 * ms
    a = 0.0 * nS
    I = 0. * nA
    Ee = 0 * mV
    Ei = -80 * mV
    eqs = EQS[part]
    e_vars = ("GsynE_aff", "GsynE_rec") if split else ("GsynE",)
    e_rec, e_aff = ("GsynE_rec", "GsynE_aff") if split else ("GsynE", "GsynE")

    # FS population
    G_inh = NeuronGroup(N1, model=eqs, threshold='vm > Vcut',
                        reset="vm = Vr; w += b", refractory='5*ms', method='heun')
    G_inh.vm = -60 * mV
    G_inh.w = a * (G_inh.vm - G_inh.EL)
    G_inh.Vr = -65 * mV
    G_inh.GsynI = 0.0 * nS
    for v in e_vars:
        setattr(G_inh, v, 0.0 * nS)
    G_inh.TsynI = 5.0 * ms
    G_inh.TsynE = 5.0 * ms
    G_inh.b = 0 * pA
    G_inh.DeltaT = 0.5 * mV
    G_inh.VT = -50. * mV
    G_inh.Vcut = G_inh.VT + 5 * G_inh.DeltaT
    G_inh.EL = float(EL_inh) * mV

    # RS population
    G_exc = NeuronGroup(N2, model=eqs, threshold='vm > Vcut',
                        reset="vm = Vr; w += b", refractory='5*ms', method='heun')
    G_exc.vm = -60 * mV
    G_exc.w = a * (G_exc.vm - G_exc.EL)
    G_exc.Vr = -65 * mV
    G_exc.GsynI = 0.0 * nS
    for v in e_vars:
        setattr(G_exc, v, 0.0 * nS)
    G_exc.TsynI = 5.0 * ms
    G_exc.TsynE = 5.0 * ms
    G_exc.b = float(b_val) * pA
    G_exc.DeltaT = 2 * mV
    G_exc.VT = -50. * mV
    G_exc.Vcut = G_exc.VT + 5 * G_exc.DeltaT
    G_exc.EL = float(EL_exc) * mV

    # constant afferent drive
    P_ed = PoissonGroup(N2, rates=float(rate) * Hz)

    # connections (the order of creation fixes the order of the random draws)
    Qi = 5.0 * nS
    Qe = 1.5 * nS
    Qe_ext = 1.5 * nS
    prbC = 0.05
    S_12 = Synapses(G_inh, G_exc, on_pre='GsynI_post+=Qi')
    S_12.connect('i!=j', p=prbC)
    S_11 = Synapses(G_inh, G_inh, on_pre='GsynI_post+=Qi')
    S_11.connect('i!=j', p=prbC)
    S_21 = Synapses(G_exc, G_inh, on_pre=f'{e_rec}_post+=Qe')
    S_21.connect('i!=j', p=prbC)
    S_22 = Synapses(G_exc, G_exc, on_pre=f'{e_rec}_post+=Qe')
    S_22.connect('i!=j', p=prbC)
    S_ed_in = Synapses(P_ed, G_inh, on_pre=f'{e_aff}_post+=Qe_ext')
    S_ed_in.connect(p=prbC)
    S_ed_ex = Synapses(P_ed, G_exc, on_pre=f'{e_aff}_post+=Qe_ext')
    S_ed_ex.connect(p=prbC)

    # recorded subsets: 10 % of each population
    rec1 = np.random.choice(N1, size=int(0.1 * N1), replace=False)
    rec2 = np.random.choice(N2, size=int(0.1 * N2), replace=False)

    M1G_inh = SpikeMonitor(G_inh)
    FRG_inh = PopulationRateMonitor(G_inh)
    M1G_exc = SpikeMonitor(G_exc)
    FRG_exc = PopulationRateMonitor(G_exc)
    M2G1 = StateMonitor(G_inh, 'vm', record=rec1)
    M3G1 = StateMonitor(G_inh, 'w', record=rec1)                            # noqa: F841
    # monitors are bound to local names: run() collects the objects it simulates from them
    M4G1 = StateMonitor(G_inh, e_vars[0], record=rec1)
    M4G1_rec = StateMonitor(G_inh, e_vars[1], record=rec1) if split else None
    M5G1 = StateMonitor(G_inh, 'GsynI', record=rec1)
    M2G2 = StateMonitor(G_exc, 'vm', record=rec2)
    M3G2 = StateMonitor(G_exc, 'w', record=rec2)
    M4G2 = StateMonitor(G_exc, e_vars[0], record=rec2)
    M4G2_rec = StateMonitor(G_exc, e_vars[1], record=rec2) if split else None
    M5G2 = StateMonitor(G_exc, 'GsynI', record=rec2)

    run(duration)

    # population rates
    BIN = bin_size
    time_array = np.array(np.arange(int(TotTime / DT))) * DT
    popRateG_exc = bin_array(np.array(FRG_exc.rate / Hz), BIN, time_array)
    popRateG_inh = bin_array(np.array(FRG_inh.rate / Hz), BIN, time_array)
    transient_end = 500                     # ms
    transient_end_idx = int(transient_end / BIN)
    average_exc_rate = np.mean(popRateG_exc[transient_end_idx:])
    average_inh_rate = np.mean(popRateG_inh[transient_end_idx:])
    std_exc_rate = np.std(popRateG_exc[transient_end_idx:])
    std_inh_rate = np.std(popRateG_inh[transient_end_idx:])

    # conductances, weighted by population size
    t_idx = int(transient_end / DT)
    if split:
        M4G1_trimmed = M4G1.GsynE_aff[:, t_idx:] + M4G1_rec.GsynE_rec[:, t_idx:]
        G2_aff_trimmed = M4G2.GsynE_aff[:, t_idx:]
        G2_rec_trimmed = M4G2_rec.GsynE_rec[:, t_idx:]
        M4G2_trimmed = G2_aff_trimmed + G2_rec_trimmed
    else:
        M4G1_trimmed = M4G1.GsynE[:, t_idx:]
        M4G2_trimmed = M4G2.GsynE[:, t_idx:]
    avg_GsynE_inh = np.mean(M4G1_trimmed)
    avg_GsynE_exc = np.mean(M4G2_trimmed)
    muGe = (avg_GsynE_inh * N1 + avg_GsynE_exc * N2) / (N1 + N2)
    var_GsynE_inh = np.mean(np.var(M4G1_trimmed, axis=1))
    var_GsynE_exc = np.mean(np.var(M4G2_trimmed, axis=1))
    var_muGe = (var_GsynE_inh * N1 + var_GsynE_exc * N2) / (N1 + N2)
    del M4G1_trimmed, M4G2_trimmed

    M5G1_trimmed = M5G1.GsynI[:, t_idx:]
    M5G2_trimmed = M5G2.GsynI[:, t_idx:]
    avg_GsynI_inh = np.mean(M5G1_trimmed)
    avg_GsynI_exc = np.mean(M5G2_trimmed)
    muGi = (avg_GsynI_inh * N1 + avg_GsynI_exc * N2) / (N1 + N2)
    var_GsynI_inh = np.mean(np.var(M5G1_trimmed, axis=1))
    var_GsynI_exc = np.mean(np.var(M5G2_trimmed, axis=1))
    var_muGi = (var_GsynI_inh * N1 + var_GsynI_exc * N2) / (N1 + N2)

    g_ratio = (muGe + muGi + gL) / gL       # total conductance relative to leak
    tau_eff = C / (gL + muGe + muGi)        # effective membrane time constant
    e_i_ratio = muGe / muGi

    # membrane potential and adaptation
    M2G1_trimmed = M2G1.vm[:, t_idx:]
    M2G2_trimmed = M2G2.vm[:, t_idx:]
    muVi = np.mean(M2G1_trimmed)
    muVe = np.mean(M2G2_trimmed)
    sVi = np.std(M2G1_trimmed)
    sVe = np.std(M2G2_trimmed)
    M3G2_trimmed = M3G2.w[:, t_idx:]
    w_stationary = np.mean(M3G2_trimmed)

    M1G_inh_trimmed = {i: t[t >= transient_end * ms] for i, t in M1G_inh.spike_trains().items()}
    M1G_exc_trimmed = {i: t[t >= transient_end * ms] for i, t in M1G_exc.spike_trains().items()}

    out = {
        "b_val": b_val, "rate": rate, "EL_inh": EL_inh, "EL_exc": EL_exc,
        "average_exc_rate": average_exc_rate, "average_inh_rate": average_inh_rate,
        "std_exc_rate": std_exc_rate, "std_inh_rate": std_inh_rate,
        "muGe": muGe, "muGi": muGi, "var_muGe": var_muGe, "var_muGi": var_muGi,
        "e_i_ratio": e_i_ratio, "g_ratio": g_ratio, "tau_eff": tau_eff,
        "muVe": muVe, "muVi": muVi, "sVe": sVe, "sVi": sVi,
        "w_stationary": w_stationary,
        "CV": compute_cv(M1G_inh_trimmed, M1G_exc_trimmed),
        "CC": compute_cc(M1G_inh_trimmed, M1G_exc_trimmed),
        "recurrence": average_exc_rate / (average_exc_rate + rate),
    }
    if not split:
        return out

    # synaptic currents onto the sampled RS cells.  With this sign convention I_e < 0 and
    # I_i > 0, so I_e + I_i is the residual after excitation and inhibition cancel.
    # (temporaries are released as soon as they are used: the recorded traces alone take
    # several GB; the values computed are unaffected)
    vm_exc = M2G2_trimmed / volt
    Ee_val = float(Ee / volt)
    Ei_val = float(Ei / volt)
    driving_exc = vm_exc - Ee_val
    I_e_aff = (G2_aff_trimmed / siemens) * driving_exc
    mean_I_e_aff = np.mean(I_e_aff)
    I_e_rec = (G2_rec_trimmed / siemens) * driving_exc
    del driving_exc
    I_e_total = I_e_aff + I_e_rec
    del I_e_aff, I_e_rec
    mean_I_e_total = np.mean(I_e_total)
    del I_e_total
    driving_inh = vm_exc - Ei_val
    del vm_exc
    I_i_syn = (M5G2_trimmed / siemens) * driving_inh
    del driving_inh
    mean_I_i_syn = np.mean(I_i_syn)
    del I_i_syn
    aff_fraction = np.abs(mean_I_e_aff) / np.abs(mean_I_e_total)
    abs_inh_exc_ratio = np.abs(mean_I_i_syn) / np.abs(mean_I_e_total)

    # balance index (Ahmadian & Miller, 2021): net synaptic current relative to the total
    # excitatory current; the variant with w adds the (hyperpolarising) adaptation current
    mean_I_net = mean_I_e_total + mean_I_i_syn
    balance_index = np.abs(mean_I_net) / np.abs(mean_I_e_total)
    mean_w = np.mean(M3G2_trimmed / amp)
    balance_index_with_w = np.abs(mean_I_net + mean_w) / np.abs(mean_I_e_total)

    out.update({
        "aff_fraction": aff_fraction, "abs_inh_exc_ratio": abs_inh_exc_ratio,
        "mean_I_e_total": mean_I_e_total, "mean_I_i_syn": mean_I_i_syn,
        "mean_I_net": mean_I_net, "balance_index": balance_index,
        "balance_index_with_w": balance_index_with_w,
    })
    return out


# --- sweeps ---
def tasks_for(part, sets, seeds=None):
    """(part, set, EL_inh, seed) of every run, in stored order (Delta_EL, then seed)."""
    seeds = SEEDS[part] if seeds is None else seeds
    return [(part, tag, EL_EXC - d, s) for tag in sets for d in DELTA_EL[part] for s in seeds]


def run_task(task):
    part, tag, EL_inh, seed, target = task
    prefs.codegen.target = target
    p = PARAM_SETS[tag]
    rec = run_baseline_simulation(part, p["b_val"], p["rate"], EL_inh, EL_EXC, BIN_SIZE, seed)
    if part == "currents":
        rec["param_set"] = tag
        rec["delta_EL"] = float(EL_EXC - EL_inh)
        rec["EL_inh"] = float(EL_inh)
        rec["seed"] = seed
    return rec


def run_tasks(tasks, workers):
    """Run the tasks, each in a fresh worker process (releases the recorded traces)."""
    with Pool(max(workers, 1), maxtasksperchild=1) as pool:
        return pool.map(run_task, tasks, chunksize=1)


def reduce_metrics(records_by_set, seeds):
    """Figure table of part 'metrics' from the full records ({set: list of records}, records in
    the order Delta_EL, seed)."""
    rows = []
    for tag, recs in records_by_set.items():
        for k, r in enumerate(recs):
            rows.append({"param_set": tag, "delta_EL": float(r["EL_exc"] - r["EL_inh"]),
                         "seed": seeds[k % len(seeds)],
                         "average_exc_rate": float(r["average_exc_rate"]),
                         "average_inh_rate": float(r["average_inh_rate"]),
                         "muVe_mV": float(r["muVe"] / mV), "muVi_mV": float(r["muVi"] / mV),
                         "g_ratio": float(r["g_ratio"])})
    return pd.DataFrame(rows)


def reduce_currents(df):
    """Figure table of part 'currents' from the full table."""
    return df[["param_set", "delta_EL", "seed", "aff_fraction", "abs_inh_exc_ratio",
               "balance_index"]].reset_index(drop=True)


def check(target, workers):
    """Re-run the first stored network of each part (r2b0, Delta_EL = -15 mV; seed 12 for
    'metrics', seed 0 for 'currents') and compare with the stored figure tables."""
    tasks = [("metrics", "r2b0", EL_EXC + 15, 12, target or TARGETS["metrics"]),
             ("currents", "r2b0", EL_EXC + 15, 0, target or TARGETS["currents"])]
    new_m, new_c = run_tasks(tasks, workers)
    new = {"metrics": reduce_metrics({"r2b0": [new_m]}, [12]).iloc[0],
           "currents": reduce_currents(pd.DataFrame([new_c])).iloc[0]}
    same = True
    for part, name in (("metrics", METRICS_CSV), ("currents", CURRENTS_CSV)):
        stored = pd.read_csv(OUT_DIR / name, float_precision="round_trip")
        row = stored[(stored.param_set == "r2b0") & (stored.delta_EL == -15)
                     & (stored.seed == new[part]["seed"])].iloc[0]
        print(f"{name}, r2b0, Delta_EL = -15 mV, seed {row.seed}, "
              f"target {target or TARGETS[part]}:")
        for k in stored.columns[3:]:
            a, b = float(new[part][k]), float(row[k])
            same &= a == b
            print(f"  {k:18s} new {a:.12g}  stored {b:.12g}  "
                  + ("identical" if a == b else f"rel. diff {abs(a - b) / abs(b):.1e}"))
    print("all stored fields reproduced exactly" if same
          else "NOT all stored fields reproduced exactly")
    return same


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parts", nargs="+", default=["metrics", "currents"],
                    choices=["metrics", "currents"])
    ap.add_argument("--sets", nargs="+", default=list(PARAM_SETS), choices=list(PARAM_SETS))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--target", default=None, choices=["numpy", "cython"],
                    help="Brian2 code-generation target (default: that of the stored files of "
                    "each part)")
    ap.add_argument("--check", action="store_true", help="re-run one stored network per part "
                    "and compare")
    args = ap.parse_args()
    if args.check:
        check(args.target, args.workers)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for part in args.parts:
        tasks = [t + (args.target or TARGETS[part],) for t in tasks_for(part, args.sets)]
        results = run_tasks(tasks, args.workers)
        if part == "metrics":
            n = len(DELTA_EL[part]) * len(SEEDS[part])
            by_set = {tag: results[i * n:(i + 1) * n] for i, tag in enumerate(args.sets)}
            for tag, recs in by_set.items():
                with open(args.out / f"baseline_results_new_{tag}_final.pkl", "wb") as f:
                    pickle.dump(recs, f)
            reduce_metrics(by_set, SEEDS[part]).to_csv(args.out / METRICS_CSV, index=False)
        else:
            df = pd.DataFrame(results)
            df.to_csv(args.out / "sweep_delta_EL_balance_results.csv", index=False)
            reduce_currents(df).to_csv(args.out / CURRENTS_CSV, index=False)
        print(f"part {part}: {len(results)} runs -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

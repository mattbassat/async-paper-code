"""Responses of the AdEx network to single sinusoidal pulses (supplementary pulse-input
figures)."""
import argparse
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from brian2 import (NeuronGroup, PoissonGroup, PopulationRateMonitor, Synapses, TimedArray,
                    defaultclock, prefs, run, start_scope)
from brian2 import seed as brian_seed
from brian2.units import Hz, mV, ms, nA, nS, pA, pF  # noqa: F401  (units used in the model)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asyncresp.paths import SUPP

OUT_DIR = SUPP / "pulse_inputs"
RATE, B, A_VALUE, EL_EXC = 2, 0, 2, -70            # Hz, pA, Hz, mV
T_CENTER, BINSIZE, TOT_TIME = 2500, 1, 5000        # ms
PERIODS = [2000, 200, 20]                          # ms
SEEDS = list(range(10))
DELTA_EL = {"half_cycle": [-15, -5, 5], "two_cycle": [5]}     # mV (Delta_EL = EL_exc - EL_inh)

# Brian2 code-generation target of the original analysis
TARGET = "cython"


# --- pulse waveforms ---
def half_cycle_impulse(t, A=5, period=500, offset=0, t_center=2500):
    """One positive half-cycle A sin(.) of duration period / 2 centred at t_center (ms), on top
    of `offset`."""
    half_width = period / 4
    t_start = t_center - half_width
    t_end = t_center + half_width
    waveform = np.zeros_like(t) + offset
    mask = (t >= t_start) & (t <= t_end)
    phase = (t[mask] - t_start) / (t_end - t_start) * np.pi
    waveform[mask] += A * np.sin(phase)
    return waveform


def two_cycle_impulse(t, A=5, period=500, offset=0, t_center=2500):
    """Two full cycles A sin(.) (total duration 2 period) centred at t_center (ms), on top of
    `offset`, clipped at zero."""
    total_duration = 2 * period
    t_start = t_center - total_duration / 2
    t_end = t_center + total_duration / 2
    waveform = np.zeros_like(t) + offset
    mask = (t >= t_start) & (t <= t_end)
    phase = (t[mask] - t_start) / (t_end - t_start) * (4 * np.pi)
    waveform[mask] += A * np.sin(phase)
    return np.clip(waveform, 0, None)


SHAPES = {"half_cycle": half_cycle_impulse, "two_cycle": two_cycle_impulse}


def default_window(shape, period, t_center=T_CENTER):
    """Plotted time window (ms): 0-5000 ms for P = 2000 ms, otherwise t_center +- 3 P / 2
    (half_cycle) or t_center +- 2 P (two_cycle)."""
    if period == 2000:
        return (0.0, float(TOT_TIME))
    hw = 3 * period / 2 if shape == "half_cycle" else 2 * period
    return (t_center - hw, t_center + hw)


def bin_array(array, BIN, time_array):
    """Mean over consecutive bins of BIN ms (the incomplete last bin is dropped)."""
    N0 = int(BIN / (time_array[1] - time_array[0]))
    N1 = int((time_array[-1] - time_array[0]) / BIN)
    return array[:N0 * N1].reshape((N1, N0)).mean(axis=1)


# --- network ---
def simulate_pulse(b_val, rate, EL_inh, EL_exc, A=1, period=500, t_center=2500, binsize=5,
                   seed=12, shape="half_cycle"):
    """One 5 s run of the network under a single pulse (`shape`: 'half_cycle' or 'two_cycle').
    rate: nu_drive (Hz); b_val: adaptation of RS cells (pA); EL_inh, EL_exc (mV); A (Hz);
    period, t_center, binsize (ms); seed: seed of NumPy and Brian2."""
    start_scope()
    np.random.seed(seed)
    brian_seed(seed)

    DT = 0.1
    defaultclock.dt = DT * ms

    N1 = 2000
    N2 = 8000
    TotTime = TOT_TIME
    duration = TotTime * ms

    C = 200 * pF          # noqa: F841
    gL = 10 * nS          # noqa: F841
    tauw = 500 * ms       # noqa: F841
    a = 0.0 * nS
    I = 0. * nA           # noqa: F841,E741
    Ee = 0 * mV           # noqa: F841
    Ei = -80 * mV         # noqa: F841

    eqs = """
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
    """

    G_inh = NeuronGroup(N1, model=eqs, threshold='vm > Vcut',
                        reset="vm = Vr; w += b", refractory='5*ms', method='heun')
    G_exc = NeuronGroup(N2, model=eqs, threshold='vm > Vcut',
                        reset="vm = Vr; w += b", refractory='5*ms', method='heun')

    for G, EL_val, b_val_local in [(G_inh, EL_inh, 0), (G_exc, EL_exc, b_val)]:
        G.vm = -60 * mV
        G.w = a * (G.vm - G.EL)
        G.Vr = -65 * mV
        G.GsynI = 0.0 * nS
        G.GsynE = 0.0 * nS
        G.TsynI = 5.0 * ms
        G.TsynE = 5.0 * ms
        G.b = float(b_val_local) * pA
        G.DeltaT = 2 * mV if G is G_exc else 0.5 * mV
        G.VT = -50. * mV
        G.Vcut = G.VT + 5 * G.DeltaT
        G.EL = float(EL_val) * mV

    Qi = 5.0 * nS         # noqa: F841
    Qe = 1.5 * nS         # noqa: F841
    Qe_ext = 1.5 * nS     # noqa: F841
    prbC = 0.05

    # recurrent connections (the order of creation fixes the order of the random draws)
    S_ie = Synapses(G_inh, G_exc, on_pre='GsynI_post+=Qi')
    S_ie.connect('i!=j', p=prbC)
    S_ii = Synapses(G_inh, G_inh, on_pre='GsynI_post+=Qi')
    S_ii.connect('i!=j', p=prbC)
    S_ei = Synapses(G_exc, G_inh, on_pre='GsynE_post+=Qe')
    S_ei.connect('i!=j', p=prbC)
    S_ee = Synapses(G_exc, G_exc, on_pre='GsynE_post+=Qe')
    S_ee.connect('i!=j', p=prbC)

    # afferent drive: Poisson neurons whose common rate follows the pulse waveform
    DT_aff = 0.1
    time_array_aff = np.arange(0, TotTime, DT_aff)
    afferent_input = SHAPES[shape](time_array_aff, A, period, rate, t_center) * Hz
    rate_array = TimedArray(afferent_input, dt=DT * ms)                     # noqa: F841
    G_aff = PoissonGroup(N2, rates='rate_array(t)')
    S_aff_e = Synapses(G_aff, G_exc, on_pre='GsynE_post+=Qe_ext')
    S_aff_e.connect(p=prbC)
    S_aff_i = Synapses(G_aff, G_inh, on_pre='GsynE_post+=Qe_ext')
    S_aff_i.connect(p=prbC)

    FRG_inh = PopulationRateMonitor(G_inh)
    FRG_exc = PopulationRateMonitor(G_exc)

    run(duration)

    time_array = np.asarray(FRG_exc.t / ms)
    t_binned = bin_array(time_array, binsize, time_array)
    exc = bin_array(np.array(FRG_exc.rate / Hz), binsize, time_array)
    inh = bin_array(np.array(FRG_inh.rate / Hz), binsize, time_array)
    aff = bin_array(np.asarray(afferent_input / Hz), binsize, time_array)
    return t_binned, exc, inh, aff


# --- runs ---
def tasks_for(shape, seeds):
    """(shape, delta_EL, period, seed) of every run, conditions in figure order."""
    return [(shape, d, p, s) for d in DELTA_EL[shape] for p in PERIODS for s in seeds]


def run_task(task):
    shape, delta_EL, period, seed, target = task
    prefs.codegen.target = target
    t0 = time.time()
    t, exc, inh, aff = simulate_pulse(B, RATE, EL_EXC - delta_EL, EL_EXC, A=A_VALUE,
                                      period=period, t_center=T_CENTER, binsize=BINSIZE,
                                      seed=int(seed), shape=shape)
    lo, hi = default_window(shape, period)
    m = (t >= lo) & (t <= hi)
    print(f"{shape} Delta_EL {delta_EL:+d} mV, P {period} ms, seed {seed}: "
          f"{time.time() - t0:.0f} s", flush=True)
    return t[m], exc[m], inh[m], aff[m]


def collect(shape, seeds, results, target):
    """Group the per-run traces into one record per condition."""
    conds, k = [], 0
    for d in DELTA_EL[shape]:
        for p in PERIODS:
            runs = results[k:k + len(seeds)]
            k += len(seeds)
            conds.append(dict(
                delta_EL=d, EL_inh=EL_EXC - d, period=p,
                pulse_ms=p / 2 if shape == "half_cycle" else 2 * p,
                window=default_window(shape, p),
                t_ms=runs[0][0].astype(np.float32),
                exc=np.array([r[1] for r in runs], dtype=np.float32),
                inh=np.array([r[2] for r in runs], dtype=np.float32),
                aff=np.array([r[3] for r in runs], dtype=np.float32)))
    return dict(shape=shape, A=A_VALUE, rate=RATE, b=B, EL_exc=EL_EXC, t_center=T_CENTER,
                binsize=BINSIZE, seeds=list(seeds), target=target,
                conditions=conds)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("shapes", nargs="*", default=list(SHAPES), choices=list(SHAPES))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--target", default=TARGET, choices=["cython", "numpy"],
                    help="Brian2 code-generation target (default: that of the original analysis)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for shape in args.shapes:
        tasks = [t + (args.target,) for t in tasks_for(shape, args.seeds)]
        t0 = time.time()
        if args.workers <= 1:
            results = [run_task(t) for t in tasks]
        else:
            with Pool(args.workers) as pool:
                results = pool.map(run_task, tasks, chunksize=1)
        out = collect(shape, args.seeds, results, args.target)
        path = args.out / f"{shape}.pkl"
        with open(path, "wb") as f:
            pickle.dump(out, f)
        print(f"{shape}: {len(tasks)} runs in {(time.time() - t0) / 60:.1f} min -> {path}",
              flush=True)


if __name__ == "__main__":
    main()

"""Membrane-potential traces of single AdEx and HH cells under network-like synaptic
bombardment."""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from brian2 import (NeuronGroup, PoissonGroup, SpikeMonitor, StateMonitor, Synapses,
                    defaultclock, prefs, run, seed, start_scope)
from brian2 import Hz, mV, ms, nS, pA, pF  # noqa: F401  (units used in the model strings)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asyncresp.paths import SUPP
from simulations.hh.cells import get_membrane_equation

OUT = SUPP / "single_cell_traces" / "traces.pkl"

# operating point of the AdEx network (nu_drive = 4 Hz, b = 0 pA, Delta_EL = 0 mV)
AVERAGE_EXC_RATE = 7.6097969966629595   # Hz
AVERAGE_INH_RATE = 24.53349276974416    # Hz
DRIVE_RATE = 4.0                        # Hz
N_EXC_IN, N_INH_IN = 400, 100           # expected numbers of presynaptic partners
SEED = 10
TSTOP, TRANSIENT, SHOW_MS = 2000, 1000, 1000   # ms


def _window(t, vm, spike_times, transient, show_ms):
    mask = (t >= transient) & (t < transient + show_ms)
    spk = spike_times[(spike_times >= transient) & (spike_times < transient + show_ms)]
    return t[mask] - transient, vm[mask], spk - transient


def simulate_adex(EL, DeltaT, b_val, nu_e, nu_i, tstop=TSTOP, transient=TRANSIENT,
                  show_ms=SHOW_MS, DT=0.1, Qe=1.5, Qi=5.0, Tsyn_e=5.0, Tsyn_i=5.0,
                  Ee=0., Ei=-80., seed_val=SEED):
    """Single AdEx cell driven by one excitatory and one inhibitory Poisson source."""
    start_scope()
    defaultclock.dt = DT * ms
    seed(seed_val)
    np.random.seed(seed_val)

    C = 200*pF; gL = 10*nS; tauw = 500*ms; a = 0*nS   # noqa: F841,E702
    Vr = -65*mV                                         # noqa: F841

    eqs = f"""
    dvm/dt = (gL*(EL_var - vm) + gL*DeltaT_var*exp((vm - VT_var)/DeltaT_var)
              - GsynE*(vm - {Ee}*mV) - GsynI*(vm - {Ei}*mV)
              - w) / C : volt (unless refractory)
    dw/dt = (a*(vm - EL_var) - w) / tauw : amp
    dGsynE/dt = -GsynE / ({Tsyn_e}*ms) : siemens
    dGsynI/dt = -GsynI / ({Tsyn_i}*ms) : siemens
    EL_var     : volt
    VT_var     : volt
    DeltaT_var : volt
    """

    G = NeuronGroup(1, eqs,
                    threshold='vm > VT_var + 5*DeltaT_var',
                    reset='vm = Vr; w += b_val*pA',
                    refractory=5*ms,
                    method='heun')

    G.vm = EL
    G.w = 0 * pA
    G.GsynE = 0 * nS
    G.GsynI = 0 * nS
    G.EL_var = EL
    G.VT_var = -50 * mV
    G.DeltaT_var = DeltaT

    P_exc = PoissonGroup(1, rates=nu_e * Hz)
    P_inh = PoissonGroup(1, rates=nu_i * Hz)

    S_exc = Synapses(P_exc, G, on_pre='GsynE_post += Qe*nS')
    S_exc.connect()
    S_inh = Synapses(P_inh, G, on_pre='GsynI_post += Qi*nS')
    S_inh.connect()

    VM = StateMonitor(G, 'vm', record=True)
    SM = SpikeMonitor(G)

    run(tstop * ms)

    return _window(np.asarray(VM.t / ms), np.asarray(VM.vm[0] / mV), np.asarray(SM.t / ms),
                   transient, show_ms)


def simulate_hh(GNa_density, GK_density, VT, nu_e, nu_i, tstop=TSTOP, transient=TRANSIENT,
                show_ms=SHOW_MS, DT=0.1, Qe=1.5, Qi=5.0, Tsyn_e=5.0, Tsyn_i=5.0,
                Ee=0., Ei=-80., membrane_size=200e-6, cell_name='HH', seed_val=SEED):
    """Single HH cell (densities in mS/cm^2, VT in mV) driven like simulate_adex."""
    start_scope()
    defaultclock.dt = DT * ms
    seed(seed_val)
    np.random.seed(seed_val)

    params = {
        'name': cell_name,
        'N': 1,
        'Cm': 1.0e6 * membrane_size,
        'Gl': 0.05e6 * membrane_size,
        'GNa': GNa_density * 1e-3 * 1e9 * membrane_size,
        'GK': GK_density * 1e-3 * 1e9 * membrane_size,
        'GM': 0.,
        'El': -70.,
        'EK': -90.,
        'ENa': 50.,
        'VT': VT,
        'tau_max': 4000.,
    }

    synaptic_array = [
        {'name': 'e', 'Erev': Ee, 'Tsyn': Tsyn_e},
        {'name': 'i', 'Erev': Ei, 'Tsyn': Tsyn_i},
    ]

    neuron = get_membrane_equation(params, synaptic_array)
    neuron.VT = VT * mV
    neuron.El = -70 * mV
    neuron.V = -65 * mV
    neuron.m = 0.05
    neuron.h = 0.60
    neuron.n = 0.30
    neuron.p = 0.0
    neuron.I0 = 0 * pA

    P_exc = PoissonGroup(1, rates=nu_e * Hz)
    P_inh = PoissonGroup(1, rates=nu_i * Hz)

    S_exc = Synapses(P_exc, neuron, on_pre='Ge_post += Qe*nS')
    S_exc.connect()
    S_inh = Synapses(P_inh, neuron, on_pre='Gi_post += Qi*nS')
    S_inh.connect()

    VM = StateMonitor(neuron, 'V', record=True)
    SM = SpikeMonitor(neuron)

    run(tstop * ms)

    return _window(np.asarray(VM.t / ms), np.asarray(VM.V[0] / mV), np.asarray(SM.t / ms),
                   transient, show_ms)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    prefs.codegen.target = "numpy"
    nu_e = N_EXC_IN * (AVERAGE_EXC_RATE + DRIVE_RATE)
    nu_i = N_INH_IN * AVERAGE_INH_RATE

    # same order of runs as the original analysis (each run re-seeds both generators)
    cells = [
        ("AdEx_RS", simulate_adex, dict(EL=-70*mV, DeltaT=2*mV, b_val=0)),
        ("HH_RS", simulate_hh, dict(GNa_density=90., GK_density=3., VT=-53., cell_name='HH_RS')),
        ("AdEx_FS", simulate_adex, dict(EL=-70*mV, DeltaT=0.5*mV, b_val=0)),
        ("HH_FS", simulate_hh, dict(GNa_density=110., GK_density=5., VT=-55.,
                                    cell_name='HH_FS')),
    ]
    out = dict(input=dict(nu_e_in_hz=nu_e, nu_i_in_hz=nu_i, exc_rate_hz=AVERAGE_EXC_RATE,
                          inh_rate_hz=AVERAGE_INH_RATE, drive_rate_hz=DRIVE_RATE, seed=SEED,
                          tstop_ms=TSTOP, transient_ms=TRANSIENT, dt_ms=0.1))
    for name, fn, kw in cells:
        t, vm, spk = fn(nu_e=nu_e, nu_i=nu_i, **kw)
        par = {k: (float(v / mV) if hasattr(v, "dim") else v) for k, v in kw.items()}
        out[name] = dict(t_ms=t.astype(np.float32), vm_mV=vm.astype(np.float32),
                         spike_ms=spk, rate_hz=len(spk) / (SHOW_MS / 1000.), params=par)
        print(f"{name}: {out[name]['rate_hz']:.2f} Hz")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

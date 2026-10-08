"""Hodgkin-Huxley E-I network driven by afferent Poisson input (Fig. 5).

Network adapted from Carlu et al. (2020, J. Neurophysiol.): 8000 regular-spiking excitatory
and 2000 fast-spiking inhibitory HH cells (cells.py) with conductance-based synapses and
random connectivity (connectivity.py, 'CONFIG1').  Each cell receives afferent excitation
from a population of 8000 Poisson units (connection probability 0.05, Qe = 1.5 nS).
Integration: exponential Euler, dt = 0.1 ms, Brian2 C++ standalone device (one build
directory per simulation).

The excitatory resting potential is fixed at El_exc = -70 mV and the inhibitory one is varied;
Delta_EL = El_exc - El_inh.

  run_driven      sinusoidally modulated drive; sinusoid fits of the population rates
                  (frequency-response sweep, Fig. 5a-b)
  run_spontaneous constant drive; inhibitory LFP proxy (spontaneous spectra, Fig. 5c)
"""
import os
import shutil

import numpy as np
from brian2 import (PoissonGroup, PopulationRateMonitor, StateMonitor, Synapses, TimedArray,
                    Hz, defaultclock, device, ms, mV, nS, run, seed, set_device, start_scope)

from .cells import get_membrane_equation, get_neuron_params
from .connectivity import (build_up_recurrent_connections_for_2_pop,
                           get_connectivity_and_synapses_matrix)

NRN_EXC, NRN_INH, NTWK = 'HH_RS_grid', 'HH_FS_grid', 'CONFIG1'


def bin_array(array, BIN, time_array):
    """Average `array` (sampled on `time_array`, ms) in consecutive bins of BIN ms."""
    dt = time_array[1] - time_array[0]
    N0 = int(BIN/dt)
    N1 = int(len(array)/N0)
    return array[:N0*N1].reshape((N1, N0)).mean(axis=1)


def fit_sinusoid(time_ms, rate_hz, period_ms):
    """Least-squares fit rate = offset + amp*cos(2*pi*t/period + phase) at a known period."""
    t = np.asarray(time_ms)
    r = np.asarray(rate_hz)
    omega = 2*np.pi/period_ms
    X = np.column_stack([np.cos(omega*t), np.sin(omega*t), np.ones_like(t)])
    coeffs, _, _, _ = np.linalg.lstsq(X, r, rcond=None)
    A_cos, A_sin, offset = coeffs
    amp = np.sqrt(A_cos**2 + A_sin**2)
    phase = np.arctan2(-A_sin, A_cos)
    return amp, phase, offset


def _activate_device(build_dir):
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    device.reinit()
    device.activate()
    set_device('cpp_standalone', build_on_run=True, directory=build_dir)


def run_driven(El_inh, osc_period, SEED, build_dir, El_exc=-70, DT=0.1, tstop=5000,
               ext_drive=4.0, Qe_ext=1.5, osc_amplitude=1.0, osc_onset=2500):
    """Network driven at ext_drive (Hz per afferent unit), modulated from osc_onset (ms) on by
    osc_amplitude*sin(2*pi*(t - osc_onset)/osc_period).

    Returns a dict with the sinusoid fits (amp, phase, offset; Hz, rad, Hz) of the excitatory
    and inhibitory population rates (1-ms bins, fitted over t >= 3000 ms) and the baseline
    rates average_exc_rate / average_inh_rate (mean over 500 ms <= t < osc_onset).
    """
    _activate_device(build_dir)
    start_scope()
    defaultclock.dt = DT * ms
    seed(SEED)
    np.random.seed(SEED)

    M = get_connectivity_and_synapses_matrix(NTWK, number=2)
    Ne = int(M[0, 0]['Ntot'] * (1 - M[0, 0]['gei']))
    Ni = int(M[0, 0]['Ntot'] * M[0, 0]['gei'])
    exc_params = get_neuron_params(NRN_EXC, number=Ne)
    inh_params = get_neuron_params(NRN_INH, number=Ni)
    exc_neurons = get_membrane_equation(exc_params, M[:, 0])
    inh_neurons = get_membrane_equation(inh_params, M[:, 1])
    exc_neurons.El = El_exc * mV
    inh_neurons.El = El_inh * mV
    exc_neurons.VT = exc_params['VT'] * mV
    inh_neurons.VT = inh_params['VT'] * mV

    # initial conditions
    exc_neurons.V = '(-90+30*rand())*mV'
    inh_neurons.V = '(-90+30*rand())*mV'
    exc_neurons.p = 0.2

    # recurrent connections (kept as named locals: run() collects the objects of this frame)
    exc_exc, exc_inh, inh_exc, inh_inh = \
        build_up_recurrent_connections_for_2_pop([exc_neurons, inh_neurons], M)

    PRe = PopulationRateMonitor(exc_neurons)
    PRi = PopulationRateMonitor(inh_neurons)

    # afferent Poisson rate, oscillating about ext_drive after osc_onset
    time_array = np.arange(int(tstop/DT)) * DT
    osc_drive = np.zeros_like(time_array)
    mask = time_array >= osc_onset
    osc_drive[mask] = osc_amplitude * np.sin(
        2 * np.pi * (time_array[mask] - osc_onset) / osc_period)
    total_drive = ext_drive + osc_drive
    drive_array = TimedArray(total_drive * Hz, dt=DT*ms)

    G_aff = PoissonGroup(Ne, rates='drive_array(t)')
    Qe_ext = Qe_ext * nS
    S_aff_ex = Synapses(G_aff, exc_neurons, on_pre='Gee_post += Qe_ext')
    S_aff_ex.connect(p=0.05)
    S_aff_in = Synapses(G_aff, inh_neurons, on_pre='Gei_post += Qe_ext')
    S_aff_in.connect(p=0.05)

    run(tstop * ms)

    rate_exc = np.array(PRe.rate / Hz)
    rate_inh = np.array(PRi.rate / Hz)

    # 1-ms bins
    BIN = 1
    time_monitor = np.asarray(PRe.t / ms)
    rate_exc_binned = bin_array(rate_exc, BIN, time_monitor)
    rate_inh_binned = bin_array(rate_inh, BIN, time_monitor)
    time_binned = bin_array(time_monitor, BIN, time_monitor)

    # baseline rates: after a 500-ms transient, before the modulation starts
    baseline_mask = (time_binned >= 500) & (time_binned < osc_onset)
    average_exc_rate = np.mean(rate_exc_binned[baseline_mask])
    average_inh_rate = np.mean(rate_inh_binned[baseline_mask])

    # sinusoid fits over the last 2 s (the first 500 ms of modulation are discarded)
    mask = time_binned >= 3000
    amp_exc, phase_exc, offset_exc = fit_sinusoid(
        time_binned[mask], rate_exc_binned[mask], osc_period)
    amp_inh, phase_inh, offset_inh = fit_sinusoid(
        time_binned[mask], rate_inh_binned[mask], osc_period)

    return {
        "amp_exc": amp_exc,
        "phase_exc": phase_exc,
        "offset_exc": offset_exc,
        "amp_inh": amp_inh,
        "phase_inh": phase_inh,
        "offset_inh": offset_inh,
        "average_exc_rate": average_exc_rate,
        "average_inh_rate": average_inh_rate,
    }


def run_spontaneous(delta_el, SEED, build_dir, tstop=10000.0, El_exc=-70.0, DT=0.1,
                    ext_drive=4.0, Qe_ext=1.5, mon_dt=0.5, n_rec_exc=500, V_ref=-60.0,
                    E_I=-80.0):
    """Network under constant drive (ext_drive Hz per afferent unit, no modulation) for
    tstop ms, with El_inh = El_exc - delta_el.

    The inhibitory LFP proxy is lfp_inh(t) = sum_i |G_ie,i(t) (E_I - V_ref)| over the first
    n_rec_exc excitatory cells (nS*mV), with the I->E conductance G_ie sampled every mon_dt ms
    and a fixed driving-force reference V_ref -- the same definition as for the AdEx networks.
    Returns (lfp_inh, rate_exc, rate_inh, t_ms); rates in Hz at every time step.
    """
    _activate_device(build_dir)
    start_scope()
    defaultclock.dt = DT * ms
    seed(SEED)
    np.random.seed(SEED)

    M = get_connectivity_and_synapses_matrix(NTWK, number=2)
    Ne = int(M[0, 0]["Ntot"] * (1 - M[0, 0]["gei"]))
    Ni = int(M[0, 0]["Ntot"] * M[0, 0]["gei"])
    exc_params = get_neuron_params(NRN_EXC, number=Ne)
    inh_params = get_neuron_params(NRN_INH, number=Ni)
    exc = get_membrane_equation(exc_params, M[:, 0])
    inh = get_membrane_equation(inh_params, M[:, 1])
    exc.El, inh.El = El_exc * mV, (El_exc - delta_el) * mV
    exc.VT, inh.VT = exc_params["VT"] * mV, inh_params["VT"] * mV
    exc.V = "(-90+30*rand())*mV"
    inh.V = "(-90+30*rand())*mV"
    exc.p = 0.2
    # recurrent connections (kept as named locals: run() collects the objects of this frame)
    exc_exc, exc_inh, inh_exc, inh_inh = \
        build_up_recurrent_connections_for_2_pop([exc, inh], M)

    # constant afferent drive
    n_steps = int(tstop / DT)
    drive = TimedArray(np.full(n_steps, ext_drive) * Hz, dt=DT * ms)
    G_aff = PoissonGroup(Ne, rates="drive(t)")
    S_e = Synapses(G_aff, exc, on_pre="Gee_post += %f*nS" % Qe_ext)
    S_e.connect(p=0.05)
    S_i = Synapses(G_aff, inh, on_pre="Gei_post += %f*nS" % Qe_ext)
    S_i.connect(p=0.05)

    PRe, PRi = PopulationRateMonitor(exc), PopulationRateMonitor(inh)
    M_gI = StateMonitor(exc, "Gie", record=np.arange(n_rec_exc), dt=mon_dt * ms)

    run(tstop * ms)

    gI = np.asarray(M_gI.Gie / nS)                       # (n_rec, n_samples), nS
    lfp_inh = np.sum(np.abs(gI * (E_I - V_ref)), axis=0)
    return (lfp_inh, np.asarray(PRe.rate / Hz), np.asarray(PRi.rate / Hz),
            np.asarray(PRe.t / ms))

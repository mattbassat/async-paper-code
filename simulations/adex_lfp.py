"""AdEx network simulations that record the synaptic-current LFP proxy (Mazzoni et al., 2015).

lfp_inh = sum over 500 RS cells of |G_I (E_I - V_ref)|, V_ref = -60 mV, sampled at 0.5 ms.
"""
import time

import numpy as np

# --- network constants ---
N_EXC, N_INH = 8000, 2000
DT_MS = 0.1                 # integration step (ms)
MON_DT_MS = 0.5             # sampling of the recorded conductances (2 kHz)
EL_EXC_MV = -70.0           # RS resting potential; E_L,inh = E_L,exc - delta_EL
V_REF_MV = -60.0            # fixed reference potential of the LFP proxy
N_REC_EXC = 500             # RS cells contributing to the LFP proxy
EE_MV, EI_MV = 0.0, -80.0   # synaptic reversal potentials

EQS = """
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
# same model with a per-neuron leak conductance (population-specific g_L)
EQS_GL = EQS + "gL:siemens\n"


# --- helpers ---
def lfp_proxy(gE_nS, gI_nS, V_ref_mV=V_REF_MV):
    """LFP proxies (lfp_exc, lfp_inh, lfp_absmag) from recorded conductances (cells x time, nS)."""
    I_E = gE_nS * (EE_MV - V_ref_mV)
    I_I = gI_nS * (EI_MV - V_ref_mV)
    lfp_exc = np.sum(np.abs(I_E), axis=0)
    lfp_inh = np.sum(np.abs(I_I), axis=0)
    return lfp_exc, lfp_inh, lfp_exc + lfp_inh


def modulated_drive(t, A, period, offset, t0, t_off):
    """Afferent rate (Hz): `offset`, plus A*sin(2 pi (t - t0)/period) for t0 <= t <= t_off."""
    H = lambda x: np.heaviside(x, 1)
    window = H(t - t0) * H(t_off - t)
    waveform = offset + A * np.sin(2 * np.pi * (t - t0) / period) * window
    if np.any(waveform < 0):
        raise ValueError("The drive waveform contains negative rates.")
    return waveform


def bin_array(array, BIN, time_array):
    """Average `array` (sampled on `time_array`) in consecutive bins of width BIN (ms)."""
    N0 = int(BIN / (time_array[1] - time_array[0]))
    N1 = int((time_array[-1] - time_array[0]) / BIN)
    return array[:N0 * N1].reshape((N1, N0)).mean(axis=1)


def clip_spikes(vm_mat, dt_ms, VT_mV=-50.0, pad_ms=3.0):
    """Remove spike waveforms from membrane-potential traces (cells x time, mV)."""
    pad_samples = max(1, int(round(pad_ms / dt_ms)))
    v = vm_mat.copy()
    mask = v >= VT_mV
    if pad_samples > 0 and mask.any():
        for s in range(1, pad_samples + 1):
            mask[:, s:] |= mask[:, :-s]
    v[mask] = np.nan
    idx = np.where(~np.isnan(v), np.arange(v.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    v = v[np.arange(v.shape[0])[:, None], idx]
    still = np.isnan(v)
    if still.any():
        rowmean = np.nanmean(v, axis=1)
        v[still] = np.take(rowmean, np.where(still)[0])
    return v


def _window_mean(x, t, t0, t1):
    m = (t >= t0) & (t < t1)
    return float(np.mean(x[m])) if m.any() else np.nan


def _setup(seed, target, dt_ms):
    """Fresh Brian2 scope with all random generators seeded by `seed`."""
    from brian2 import start_scope, defaultclock, set_device, prefs, ms
    from brian2 import seed as brian_seed
    prefs.codegen.target = target
    set_device('runtime')
    start_scope()
    np.random.seed(seed)
    brian_seed(seed)
    defaultclock.dt = dt_ms * ms
    return np.random.default_rng(seed)


def _populations(eqs, EL_inh, EL_exc, b_exc_pA, exc_reset="vm = Vr; w += b"):
    """FS (inhibitory) and RS (excitatory) AdEx populations with their standard parameters."""
    from brian2 import NeuronGroup, ms, mV, nS, pA
    a = 0.0 * nS
    G_inh = NeuronGroup(N_INH, model=eqs, threshold='vm > Vcut',
                        reset="vm = Vr; w += b", refractory='5*ms', method='heun')
    G_inh.vm = -60 * mV
    G_inh.w = a * (G_inh.vm - G_inh.EL)
    G_inh.Vr = -65 * mV
    G_inh.GsynI = 0.0 * nS
    G_inh.GsynE = 0.0 * nS
    G_inh.TsynI = 5.0 * ms
    G_inh.TsynE = 5.0 * ms
    G_inh.b = 0 * pA
    G_inh.DeltaT = 0.5 * mV
    G_inh.VT = -50. * mV
    G_inh.Vcut = G_inh.VT + 5 * G_inh.DeltaT
    G_inh.EL = float(EL_inh) * mV

    G_exc = NeuronGroup(N_EXC, model=eqs, threshold='vm > Vcut',
                        reset=exc_reset, refractory='5*ms', method='heun')
    G_exc.vm = -60 * mV
    G_exc.w = a * (G_exc.vm - G_exc.EL)
    G_exc.Vr = -65 * mV
    G_exc.GsynI = 0.0 * nS
    G_exc.GsynE = 0.0 * nS
    G_exc.TsynI = 5.0 * ms
    G_exc.TsynE = 5.0 * ms
    if b_exc_pA is not None:
        G_exc.b = float(b_exc_pA) * pA
    G_exc.DeltaT = 2 * mV
    G_exc.VT = -50. * mV
    G_exc.Vcut = G_exc.VT + 5 * G_exc.DeltaT
    G_exc.EL = float(EL_exc) * mV
    return G_inh, G_exc


# Constants of the model equations (C, tau_w, a, I, reversal potentials).  Brian2 resolves
# the names in the equations from the scope that calls run(), so each simulation function binds
# them (and gL, the quantal conductances, TimedArrays) as local variables.
def _model_constants():
    from brian2 import ms, mV, nS, pF, nA
    return dict(C=200 * pF, tauw=500 * ms, a=0.0 * nS, I=0. * nA, Ee=0 * mV, Ei=-80 * mV)


# --- spontaneous activity ---
def run_spontaneous_sim(delta_EL, seed, nu_0=2.0, b_post=0.0, total_ms=10000.0,
                        analysis_window=(2000.0, 10000.0), delay_ms=0.0, target="numpy",
                        record_spikes=False, verbose=False):
    """Spontaneous activity under a constant Poisson drive nu_0 (Hz) onto both populations."""
    from brian2 import (PoissonGroup, Synapses, PopulationRateMonitor, StateMonitor,
                        SpikeMonitor, run, ms, nS, Hz)
    rng = _setup(seed, target, DT_MS)
    ns = _model_constants()
    C, tauw, a, I, Ee, Ei = (ns[k] for k in ("C", "tauw", "a", "I", "Ee", "Ei"))
    gL = 10 * nS
    G_inh, G_exc = _populations(EQS, EL_EXC_MV - delta_EL, EL_EXC_MV, b_post)

    P_ed = PoissonGroup(N_EXC, rates=float(nu_0) * Hz)
    Qi, Qe, Qe_ext, prbC = 5.0 * nS, 1.5 * nS, 1.5 * nS, 0.05
    S_12 = Synapses(G_inh, G_exc, on_pre='GsynI_post+=Qi'); S_12.connect('i!=j', p=prbC)   # I->E
    S_11 = Synapses(G_inh, G_inh, on_pre='GsynI_post+=Qi'); S_11.connect('i!=j', p=prbC)   # I->I
    S_21 = Synapses(G_exc, G_inh, on_pre='GsynE_post+=Qe'); S_21.connect('i!=j', p=prbC)   # E->I
    S_22 = Synapses(G_exc, G_exc, on_pre='GsynE_post+=Qe'); S_22.connect('i!=j', p=prbC)   # E->E
    S_ed_in = Synapses(P_ed, G_inh, on_pre='GsynE_post+=Qe_ext'); S_ed_in.connect(p=prbC)  # drive->I
    S_ed_ex = Synapses(P_ed, G_exc, on_pre='GsynE_post+=Qe_ext'); S_ed_ex.connect(p=prbC)  # drive->E
    if delay_ms > 0:
        for S in (S_12, S_11, S_21, S_22, S_ed_in, S_ed_ex):
            S.delay = delay_ms * ms

    rec_exc = np.sort(rng.choice(N_EXC, N_REC_EXC, replace=False))
    FRG_inh = PopulationRateMonitor(G_inh)
    FRG_exc = PopulationRateMonitor(G_exc)
    M_gE = StateMonitor(G_exc, 'GsynE', record=rec_exc, dt=MON_DT_MS * ms)
    M_gI = StateMonitor(G_exc, 'GsynI', record=rec_exc, dt=MON_DT_MS * ms)
    if record_spikes:                     # monitors draw no random numbers: same network activity
        S_exc, S_inh = SpikeMonitor(G_exc), SpikeMonitor(G_inh)

    t0 = time.time()
    run(total_ms * ms)
    sim_time = time.time() - t0

    fr_exc = np.asarray(FRG_exc.rate / Hz)
    fr_inh = np.asarray(FRG_inh.rate / Hz)
    t_fr = np.arange(len(fr_exc)) * DT_MS
    lfp_exc, lfp_inh, lfp_absmag = lfp_proxy(np.asarray(M_gE.GsynE / nS), np.asarray(M_gI.GsynI / nS))
    out = dict(
        delta_EL=float(delta_EL), seed=int(seed), nu_0=float(nu_0), b_post=float(b_post),
        delay_ms=float(delay_ms), total_ms=float(total_ms), dt_fr_ms=float(DT_MS),
        dt_lfp_ms=float(MON_DT_MS),
        fr_exc=fr_exc.astype(np.float32), fr_inh=fr_inh.astype(np.float32),
        lfp_absmag=lfp_absmag.astype(np.float32), lfp_exc=lfp_exc.astype(np.float32),
        lfp_inh=lfp_inh.astype(np.float32),
        mean_exc_rate=_window_mean(fr_exc, t_fr, *analysis_window),
        mean_inh_rate=_window_mean(fr_inh, t_fr, *analysis_window),
        sim_time_s=sim_time)
    if record_spikes:
        out.update(spikes_exc=(np.asarray(S_exc.i), np.asarray(S_exc.t / ms)),
                   spikes_inh=(np.asarray(S_inh.i), np.asarray(S_inh.t / ms)))
    if verbose:
        print(f"  spontaneous dEL={delta_EL:+g} nu0={nu_0:g} b={b_post:g} seed={seed}: "
              f"rates {out['mean_exc_rate']:.2f}/{out['mean_inh_rate']:.2f} Hz ({sim_time:.0f} s)",
              flush=True)
    return out


def run_spontaneous_sim_params(seed, delta_EL=0.0, nu_0=2.0, b_post=0.0,
                               gL_exc=10.0, gL_inh=10.0, Qee=1.5, Qei=1.5, Qie=5.0, Qii=5.0,
                               total_ms=10000.0, target="cython", record_spikes=False,
                               verbose=False):
    """Spontaneous activity with population-specific leak and quantal conductances."""
    from brian2 import (PoissonGroup, Synapses, PopulationRateMonitor, StateMonitor,
                        SpikeMonitor, run, ms, nS, Hz)
    rng = _setup(seed, target, DT_MS)
    ns = _model_constants()
    C, tauw, a, I, Ee, Ei = (ns[k] for k in ("C", "tauw", "a", "I", "Ee", "Ei"))
    G_inh, G_exc = _populations(EQS_GL, EL_EXC_MV - delta_EL, EL_EXC_MV, b_post)
    G_inh.gL = float(gL_inh) * nS
    G_exc.gL = float(gL_exc) * nS

    P_ed = PoissonGroup(N_EXC, rates=float(nu_0) * Hz)
    prbC = 0.05
    Qee_, Qei_, Qie_, Qii_ = Qee * nS, Qei * nS, Qie * nS, Qii * nS
    S_12 = Synapses(G_inh, G_exc, on_pre='GsynI_post+=Qie_'); S_12.connect('i!=j', p=prbC)   # FS->RS
    S_11 = Synapses(G_inh, G_inh, on_pre='GsynI_post+=Qii_'); S_11.connect('i!=j', p=prbC)   # FS->FS
    S_21 = Synapses(G_exc, G_inh, on_pre='GsynE_post+=Qei_'); S_21.connect('i!=j', p=prbC)   # RS->FS
    S_22 = Synapses(G_exc, G_exc, on_pre='GsynE_post+=Qee_'); S_22.connect('i!=j', p=prbC)   # RS->RS
    S_ed_in = Synapses(P_ed, G_inh, on_pre='GsynE_post+=Qei_'); S_ed_in.connect(p=prbC)      # drive->FS
    S_ed_ex = Synapses(P_ed, G_exc, on_pre='GsynE_post+=Qee_'); S_ed_ex.connect(p=prbC)      # drive->RS

    rec_exc = np.sort(rng.choice(N_EXC, N_REC_EXC, replace=False))
    FRG_inh = PopulationRateMonitor(G_inh)
    FRG_exc = PopulationRateMonitor(G_exc)
    M_gE = StateMonitor(G_exc, 'GsynE', record=rec_exc, dt=MON_DT_MS * ms)
    M_gI = StateMonitor(G_exc, 'GsynI', record=rec_exc, dt=MON_DT_MS * ms)
    if record_spikes:                     # monitors draw no random numbers: same network activity
        S_exc, S_inh = SpikeMonitor(G_exc), SpikeMonitor(G_inh)

    t0 = time.time()
    run(total_ms * ms)
    sim_time = time.time() - t0

    lfp_exc, lfp_inh, lfp_absmag = lfp_proxy(np.asarray(M_gE.GsynE / nS), np.asarray(M_gI.GsynI / nS))
    out = dict(
        seed=int(seed), delta_EL=float(delta_EL), nu_0=float(nu_0), b_post=float(b_post),
        gL_exc=float(gL_exc), gL_inh=float(gL_inh), Qee=float(Qee), Qei=float(Qei),
        Qie=float(Qie), Qii=float(Qii), total_ms=float(total_ms), dt_lfp_ms=float(MON_DT_MS),
        mean_rate_exc=float(np.mean(FRG_exc.rate / Hz)), mean_rate_inh=float(np.mean(FRG_inh.rate / Hz)),
        lfp_absmag=lfp_absmag.astype(np.float32), lfp_exc=lfp_exc.astype(np.float32),
        lfp_inh=lfp_inh.astype(np.float32), sim_time_s=sim_time)
    if record_spikes:
        out.update(spikes_exc=(np.asarray(S_exc.i), np.asarray(S_exc.t / ms)),
                   spikes_inh=(np.asarray(S_inh.i), np.asarray(S_inh.t / ms)))
    if verbose:
        print(f"  spontaneous seed={seed}: rates {out['mean_rate_exc']:.2f}/{out['mean_rate_inh']:.2f} Hz "
              f"({sim_time:.0f} s)", flush=True)
    return out


# --- driven, then switched off ---
def run_offset_persistence_sim(delta_EL, seed, freq_hz, A, nu_0=2.0, b_post=0.0,
                               oscillation_onset=2000.0, oscillation_offset=5000.0,
                               total_ms=8000.0, delay_ms=0.0, target="numpy", verbose=False):
    """Drive nu_0 + A*sin(2 pi f (t - onset)) between onset and offset (ms), then nu_0 again."""
    from brian2 import (PoissonGroup, Synapses, PopulationRateMonitor, StateMonitor,
                        TimedArray, run, ms, mV, nS, Hz)
    rng = _setup(seed, target, DT_MS)
    ns = _model_constants()
    C, tauw, a, I, Ee, Ei = (ns[k] for k in ("C", "tauw", "a", "I", "Ee", "Ei"))
    gL = 10 * nS
    EL_inh = EL_EXC_MV - delta_EL
    period_ms = 1000.0 / freq_hz
    G_inh, G_exc = _populations(EQS, EL_inh, EL_EXC_MV, b_post)

    Qi, Qe, Qe_ext, prbC = 5.0 * nS, 1.5 * nS, 1.5 * nS, 0.05
    S_12 = Synapses(G_inh, G_exc, on_pre='GsynI_post+=Qi'); S_12.connect('i!=j', p=prbC)   # I->E
    S_11 = Synapses(G_inh, G_inh, on_pre='GsynI_post+=Qi'); S_11.connect('i!=j', p=prbC)   # I->I
    S_21 = Synapses(G_exc, G_inh, on_pre='GsynE_post+=Qe'); S_21.connect('i!=j', p=prbC)   # E->I
    S_22 = Synapses(G_exc, G_exc, on_pre='GsynE_post+=Qe'); S_22.connect('i!=j', p=prbC)   # E->E

    t_aff = np.arange(0, total_ms, DT_MS)
    rate_array = TimedArray(modulated_drive(t_aff, A=A, period=period_ms, offset=nu_0,
                                            t0=oscillation_onset, t_off=oscillation_offset) * Hz,
                            dt=DT_MS * ms)
    G_aff = PoissonGroup(N_EXC, rates='rate_array(t)')
    S_aff_ex = Synapses(G_aff, G_exc, on_pre='GsynE_post+=Qe_ext'); S_aff_ex.connect(p=prbC)  # drive->E
    S_aff_in = Synapses(G_aff, G_inh, on_pre='GsynE_post+=Qe_ext'); S_aff_in.connect(p=prbC)  # drive->I
    if delay_ms > 0:
        for S in (S_12, S_11, S_21, S_22, S_aff_in, S_aff_ex):
            S.delay = delay_ms * ms

    rec_exc = np.sort(rng.choice(N_EXC, N_REC_EXC, replace=False))
    FRG_exc = PopulationRateMonitor(G_exc)
    FRG_inh = PopulationRateMonitor(G_inh)
    M_vm = StateMonitor(G_exc, 'vm', record=rec_exc, dt=MON_DT_MS * ms)
    M_gE = StateMonitor(G_exc, 'GsynE', record=rec_exc, dt=MON_DT_MS * ms)
    M_gI = StateMonitor(G_exc, 'GsynI', record=rec_exc, dt=MON_DT_MS * ms)

    t0 = time.time()
    run(total_ms * ms)
    sim_time = time.time() - t0

    fr_exc_full = np.asarray(FRG_exc.rate / Hz)
    fr_inh_full = np.asarray(FRG_inh.rate / Hz)
    t_full = np.arange(len(fr_exc_full)) * DT_MS
    BIN = 1.0
    t_fr = bin_array(t_full, BIN, t_full)
    fr_exc = bin_array(fr_exc_full, BIN, t_full)
    fr_inh = bin_array(fr_inh_full, BIN, t_full)

    t_state = np.asarray(M_vm.t / ms)
    vm_exc_pop = clip_spikes(np.asarray(M_vm.vm / mV), MON_DT_MS).mean(axis=0)
    lfp_exc, lfp_inh, lfp_absmag = lfp_proxy(np.asarray(M_gE.GsynE / nS), np.asarray(M_gI.GsynI / nS))
    if verbose:
        print(f"  driven dEL={delta_EL:+g} f={freq_hz:g} Hz seed={seed} ({sim_time:.0f} s)", flush=True)

    return dict(
        nu_0=float(nu_0), b_post=float(b_post), delay_ms=float(delay_ms),
        EL_exc=float(EL_EXC_MV), EL_inh=float(EL_inh), delta_EL=float(delta_EL),
        seed=int(seed), freq_hz=float(freq_hz), period_ms=float(period_ms), A=float(A),
        oscillation_onset=float(oscillation_onset), oscillation_offset=float(oscillation_offset),
        total_ms=float(total_ms), dt_fr_ms=float(BIN), dt_state_ms=float(MON_DT_MS),
        t_fr=t_fr.astype(np.float32), fr_exc=fr_exc.astype(np.float32), fr_inh=fr_inh.astype(np.float32),
        t_state=t_state.astype(np.float32), vm_exc_pop=vm_exc_pop.astype(np.float32),
        lfp_absmag=lfp_absmag.astype(np.float32), lfp_exc=lfp_exc.astype(np.float32),
        lfp_inh=lfp_inh.astype(np.float32), sim_time_s=sim_time)


def run_spontaneous_spike_sim(delta_EL, seed, nu_0=2.0, b_val=0.0, total_ms=5000.0,
                              t_trim_ms=500.0, target="numpy"):
    """Spontaneous activity (5 s) recording all spike times, for spike-count correlations."""
    from brian2 import (start_scope, defaultclock, set_device, prefs, PoissonGroup, Synapses,
                        SpikeMonitor, PopulationRateMonitor, run, ms, nS, Hz)
    from brian2 import seed as brian_seed
    prefs.codegen.target = target
    set_device('runtime')
    start_scope()
    np.random.seed(seed)
    brian_seed(seed)
    defaultclock.dt = DT_MS * ms
    ns = _model_constants()
    C, tauw, a, I, Ee, Ei = (ns[k] for k in ("C", "tauw", "a", "I", "Ee", "Ei"))
    gL = 10 * nS
    G_inh, G_exc = _populations(EQS, EL_EXC_MV - delta_EL, EL_EXC_MV, b_val)

    P_ed = PoissonGroup(N_EXC, rates=float(nu_0) * Hz)
    Qi, Qe, Qe_ext, prbC = 5.0 * nS, 1.5 * nS, 1.5 * nS, 0.05
    S_12 = Synapses(G_inh, G_exc, on_pre='GsynI_post+=Qi'); S_12.connect('i!=j', p=prbC)   # I->E
    S_11 = Synapses(G_inh, G_inh, on_pre='GsynI_post+=Qi'); S_11.connect('i!=j', p=prbC)   # I->I
    S_21 = Synapses(G_exc, G_inh, on_pre='GsynE_post+=Qe'); S_21.connect('i!=j', p=prbC)   # E->I
    S_22 = Synapses(G_exc, G_exc, on_pre='GsynE_post+=Qe'); S_22.connect('i!=j', p=prbC)   # E->E
    S_ed_in = Synapses(P_ed, G_inh, on_pre='GsynE_post+=Qe_ext'); S_ed_in.connect(p=prbC)  # drive->I
    S_ed_ex = Synapses(P_ed, G_exc, on_pre='GsynE_post+=Qe_ext'); S_ed_ex.connect(p=prbC)  # drive->E

    M_inh = SpikeMonitor(G_inh)
    M_exc = SpikeMonitor(G_exc)
    FRG_inh = PopulationRateMonitor(G_inh)
    FRG_exc = PopulationRateMonitor(G_exc)

    t0 = time.time()
    run(total_ms * ms)
    sim_time = time.time() - t0

    # mean population rates: 5 ms bins, from t_trim_ms on
    time_array = np.arange(int(total_ms / DT_MS)) * DT_MS
    i0 = int(t_trim_ms / 5)
    return dict(
        inh={i: t[t >= t_trim_ms * ms] for i, t in M_inh.spike_trains().items()},
        exc={i: t[t >= t_trim_ms * ms] for i, t in M_exc.spike_trains().items()},
        mean_exc_rate=float(np.mean(bin_array(np.asarray(FRG_exc.rate / Hz), 5, time_array)[i0:])),
        mean_inh_rate=float(np.mean(bin_array(np.asarray(FRG_inh.rate / Hz), 5, time_array)[i0:])),
        sim_time_s=sim_time)

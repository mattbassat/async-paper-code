"""AdEx network driven by a sinusoidally modulated afferent input (Brian2).

A network of 8000 excitatory regular-spiking (RS) and 2000 inhibitory fast-spiking (FS) AdEx
neurons (di Volo et al., 2019), randomly connected with probability 0.05 through
conductance-based exponential synapses, receives Poisson input from 8000 afferent neurons
whose common rate is

    nu_aff(t) = nu_drive + A sin(2 pi t / P)     for t >= t_onset   (nu_drive before),

with nu_drive the background drive, A the modulation amplitude and P = 1000 / f the period
(ms).  The population rates (1 ms bins) are fitted, after a 3 s transient, with a sinusoid of
the input frequency by linear least squares; the fitted amplitudes are the evoked E and I
responses.  The amplitudes are also expressed relative to the SD of the rates during the
spontaneous baseline (1 s to t_onset).

`run_impulse_simulation` returns the fitted response of one run.  With `return_traces=True`
it also returns the binned rates, the drive waveform, the spike trains and an LFP proxy; the
extra recording does not change the simulated network (the monitor draws no random numbers),
so the fitted values are identical with and without traces.

Optional arguments reproduce the parameter variations of the paper:
  * gL_exc / gL_inh  (nS): population-specific leak conductances (Delta g_L sweep);
  * Qee, Qei, Qie, Qii (nS) and Tee, Tei, Tie, Tii (ms): pathway-specific quantal
    conductances and synaptic decay times, Q_{source}{target} (e = RS, i = FS); the afferent
    synapses onto RS and FS cells use Qee and Qei respectively;
  * phase_from_onset: the modulation is sin(2 pi (t - t_onset) / P) instead of
    sin(2 pi t / P).  For every input frequency that is a multiple of 0.5 Hz the two forms
    coincide (t_onset = 2 s) up to floating-point rounding; both forms are kept because each
    was used for some of the stored sweeps.

Reproducibility.  Brian2's random-number streams depend on the code-generation target, so a
stored result is reproduced only with the target it was computed with (set_target; each run
script uses the right one).  The simulated network is then identical, but the least-squares
fit can still differ in the last bits between BLAS/LAPACK builds (OpenBLAS, for instance,
selects CPU-specific kernels).  The run scripts select OpenBLAS's generic x86-64 kernels
(OPENBLAS_CORETYPE=Prescott), with which the stored results computed on x86-64 Linux (the
Cython-target runs) are reproduced bit for bit; for the others the fitted values agree to a
relative precision of about 1e-13.
"""
import pickle

import numpy as np
from brian2 import (NeuronGroup, PoissonGroup, PopulationRateMonitor, SpikeMonitor,
                    StateMonitor, Synapses, TimedArray, defaultclock, prefs, run, start_scope)
from brian2 import seed as brian_seed
from brian2.units import Hz, mV, ms, nA, nS, pA, pF

N_INH, N_EXC = 2000, 8000                  # FS and RS cells
DT = 0.1                                   # integration and drive time step (ms)
BIN = 1                                    # population-rate bin (ms)
BASELINE_START = 1000                      # start of the spontaneous baseline (ms)
FIT_CUTOFF = 3000                          # the sinusoid is fitted to t >= FIT_CUTOFF (ms)

# LFP proxy recorded with return_traces=True: sum over the first N_REC_EXC excitatory cells of
# the absolute synaptic currents the conductances would drive at a fixed reference potential
# (Mazzoni et al., 2015).
N_REC_EXC, V_REF, DT_LFP = 500, -60.0, 0.5           # cells, mV, ms


def set_target(target):
    """Brian2 code-generation target ('numpy' or 'cython'); random-number streams are
    target-specific, so a stored result is reproduced only with the target it was computed with."""
    prefs.codegen.target = target


def modulating_input(t, A, period, offset, t0, tf, phase_from_onset=False):
    """Afferent rate (Hz) at times t (ms): offset + A sin(2 pi t / period) for t0 <= t <= tf."""
    H = lambda x: np.heaviside(x, 1)
    phase_t = (t - t0) if phase_from_onset else t
    waveform = (A * np.sin(1 / period * 2 * np.pi * phase_t)) * H(t - t0) * H(tf - t) + offset
    if np.any(waveform < 0):
        raise ValueError("The afferent rate must be non-negative.")
    return waveform


def bin_array(array, bin_ms, time_array):
    """Average `array` (sampled at the times `time_array`, ms) in consecutive bins of bin_ms."""
    n0 = int(bin_ms / (time_array[1] - time_array[0]))
    n1 = int((time_array[-1] - time_array[0]) / bin_ms)
    return array[:n0 * n1].reshape((n1, n0)).mean(axis=1)


def fit_sinusoid(time_ms, rate_hz, period_ms):
    """Least-squares fit r(t) = a cos(wt) + b sin(wt) + c; returns amplitude, phase (rad), offset."""
    t = np.asarray(time_ms) * 1.0
    r = np.asarray(rate_hz) * 1.0
    mask = np.isfinite(r)
    t, r = t[mask], r[mask]
    omega = 2 * np.pi / period_ms
    X = np.column_stack([np.cos(omega * t), np.sin(omega * t), np.ones_like(t)])
    coeffs, _, _, _ = np.linalg.lstsq(X, r, rcond=None)
    A_cos, A_sin, offset = coeffs
    return np.sqrt(A_cos ** 2 + A_sin ** 2), np.arctan2(-A_sin, A_cos), offset


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


def run_impulse_simulation(b_val, rate, EL_inh, EL_exc, A=1, period=500, oscillation_onset=2000,
                           seed=12, tot_time=5000, oscillation_end=None, phase_from_onset=False,
                           gL_inh=None, gL_exc=None,
                           Qee=1.5, Qei=1.5, Qie=5.0, Qii=5.0,
                           Tee=5.0, Tei=5.0, Tie=5.0, Tii=5.0,
                           return_traces=False):
    """One run of the sinusoidally driven network.

    b_val: spike-triggered adaptation of the RS cells (pA); rate: background drive nu_drive
    (Hz); EL_inh, EL_exc: leak reversal potentials (mV); A: modulation amplitude (Hz); period:
    modulation period (ms); oscillation_onset / oscillation_end: start / end of the modulation
    (ms; the end defaults to the end of the run); seed: seed of NumPy and Brian2; tot_time:
    duration (ms); gL_inh, gL_exc: leak conductances (nS; None = 10 nS for both).
    """
    start_scope()
    np.random.seed(seed)
    brian_seed(seed)

    defaultclock.dt = DT * ms
    TotTime = tot_time
    duration = TotTime * ms
    C = 200 * pF
    tauw = 500 * ms
    a = 0.0 * nS
    I = 0. * nA
    Ee = 0 * mV
    Ei = -80 * mV
    per = period

    pop_gL = gL_inh is not None or gL_exc is not None
    if pop_gL:
        gL_inh = 10 if gL_inh is None else gL_inh
        gL_exc = 10 if gL_exc is None else gL_exc
        eqs = EQS + "gL : siemens\n"
    else:
        gL = 10 * nS
        eqs = EQS

    # synaptic parameters, Q_{source}{target}
    Qee_s, Qei_s, Qie_s, Qii_s = float(Qee) * nS, float(Qei) * nS, float(Qie) * nS, float(Qii) * nS

    G_inh = NeuronGroup(N_INH, model=eqs, threshold='vm > Vcut',
                        reset="vm = Vr; w += b", refractory='5*ms', method='heun')
    G_inh.vm = -60 * mV
    G_inh.w = a * (G_inh.vm - G_inh.EL)
    G_inh.Vr = -65 * mV
    G_inh.GsynI = 0.0 * nS
    G_inh.GsynE = 0.0 * nS
    G_inh.TsynI = float(Tii) * ms
    G_inh.TsynE = float(Tei) * ms
    G_inh.b = 0 * pA
    G_inh.DeltaT = 0.5 * mV
    G_inh.VT = -50. * mV
    G_inh.Vcut = G_inh.VT + 5 * G_inh.DeltaT
    G_inh.EL = float(EL_inh) * mV
    if pop_gL:
        G_inh.gL = gL_inh * nS

    G_exc = NeuronGroup(N_EXC, model=eqs, threshold='vm > Vcut',
                        reset="vm = Vr; w += b", refractory='5*ms', method='heun')
    G_exc.vm = -60 * mV
    G_exc.w = a * (G_exc.vm - G_exc.EL)
    G_exc.Vr = -65 * mV
    G_exc.GsynI = 0.0 * nS
    G_exc.GsynE = 0.0 * nS
    G_exc.TsynI = float(Tie) * ms
    G_exc.TsynE = float(Tee) * ms
    G_exc.b = float(b_val) * pA
    G_exc.DeltaT = 2 * mV
    G_exc.VT = -50. * mV
    G_exc.Vcut = G_exc.VT + 5 * G_exc.DeltaT
    G_exc.EL = float(EL_exc) * mV
    if pop_gL:
        G_exc.gL = gL_exc * nS

    # recurrent connections (the order of creation fixes the order of the random draws)
    prbC = 0.05
    S_12 = Synapses(G_inh, G_exc, on_pre='GsynI_post+=Qie_s'); S_12.connect('i!=j', p=prbC)
    S_11 = Synapses(G_inh, G_inh, on_pre='GsynI_post+=Qii_s'); S_11.connect('i!=j', p=prbC)
    S_21 = Synapses(G_exc, G_inh, on_pre='GsynE_post+=Qei_s'); S_21.connect('i!=j', p=prbC)
    S_22 = Synapses(G_exc, G_exc, on_pre='GsynE_post+=Qee_s'); S_22.connect('i!=j', p=prbC)

    # afferent Poisson population carrying the background drive and its modulation
    time_array_aff = np.arange(0, TotTime, DT)
    t_off = TotTime if oscillation_end is None else float(oscillation_end)
    afferent_input = modulating_input(time_array_aff, A, per, rate, oscillation_onset, t_off,
                                      phase_from_onset) * Hz
    rate_array = TimedArray(afferent_input, dt=DT * ms)
    G_aff = PoissonGroup(N_EXC, rates='rate_array(t)')
    S_aff_ex = Synapses(G_aff, G_exc, on_pre='GsynE_post+=Qee_s'); S_aff_ex.connect(p=prbC)
    S_aff_in = Synapses(G_aff, G_inh, on_pre='GsynE_post+=Qei_s'); S_aff_in.connect(p=prbC)

    M1G_inh = SpikeMonitor(G_inh)
    FRG_inh = PopulationRateMonitor(G_inh)
    M1G_exc = SpikeMonitor(G_exc)
    FRG_exc = PopulationRateMonitor(G_exc)
    if return_traces:
        SM = StateMonitor(G_exc, ['GsynI', 'GsynE'], record=np.arange(N_REC_EXC), dt=DT_LFP * ms)

    run(duration)

    # population rates in 1 ms bins
    time_array = np.asarray(FRG_exc.t / ms)
    TimBinned = bin_array(time_array, BIN, time_array)
    popRateG_exc = bin_array(np.array(FRG_exc.rate / Hz), BIN, time_array)
    popRateG_inh = bin_array(np.array(FRG_inh.rate / Hz), BIN, time_array)

    # spontaneous baseline (1 s to onset)
    if oscillation_onset < 1500:
        raise ValueError("Baseline period too short. Ensure oscillation_onset >= 1500 ms.")
    i0, i1 = int(BASELINE_START / BIN), int(oscillation_onset / BIN)
    std_exc_rate = np.std(popRateG_exc[i0:i1])
    std_inh_rate = np.std(popRateG_inh[i0:i1])

    # sinusoid fits after the transient
    valid = TimBinned >= FIT_CUTOFF
    t_fit = TimBinned[valid]
    amp_exc, phase_exc_raw, offset_exc = fit_sinusoid(t_fit, popRateG_exc[valid], period)
    amp_inh, phase_inh_raw, offset_inh = fit_sinusoid(t_fit, popRateG_inh[valid], period)
    amp_exc_norm = amp_exc / std_exc_rate if std_exc_rate > 0 else np.nan
    amp_inh_norm = amp_inh / std_inh_rate if std_inh_rate > 0 else np.nan

    # phases relative to the fitted afferent waveform (degrees)
    afferent_binned = bin_array(afferent_input, BIN, time_array)
    _, phase_aff_raw, _ = fit_sinusoid(t_fit, np.asarray(afferent_binned[valid]) * 1.0, period)
    phase_exc = np.degrees(np.angle(np.exp(1j * (phase_exc_raw - phase_aff_raw))))
    phase_inh = np.degrees(np.angle(np.exp(1j * (phase_inh_raw - phase_aff_raw))))

    out = {
        "amp_exc": amp_exc, "phase_exc_deg": phase_exc, "offset_exc": offset_exc,
        "amp_inh": amp_inh, "phase_inh_deg": phase_inh, "offset_inh": offset_inh,
        "amp_exc_norm": amp_exc_norm, "amp_inh_norm": amp_inh_norm,
    }
    if return_traces:
        gI = np.asarray(SM.GsynI / nS)                       # (N_REC_EXC, n_t)
        gE = np.asarray(SM.GsynE / nS)
        out.update({
            "t_ms": np.asarray(TimBinned), "fr_exc": np.asarray(popRateG_exc),
            "fr_inh": np.asarray(popRateG_inh),
            "t_drive_ms": np.asarray(time_array_aff), "drive_hz": np.asarray(afferent_input / Hz),
            "spk_exc_t": np.asarray(M1G_exc.t / ms), "spk_exc_i": np.asarray(M1G_exc.i),
            "spk_inh_t": np.asarray(M1G_inh.t / ms), "spk_inh_i": np.asarray(M1G_inh.i),
            "t_lfp_ms": np.asarray(SM.t / ms),
            "lfp_inh": np.abs(gI * (Ei / mV - V_REF)).sum(axis=0),
            "lfp_exc": np.abs(gE * (Ee / mV - V_REF)).sum(axis=0),
            "dt_lfp_ms": DT_LFP, "n_rec_exc": N_REC_EXC, "v_ref_mV": V_REF,
            "bin_ms": BIN, "seed": seed, "oscillation_onset": float(oscillation_onset),
            "phase_exc_raw": float(phase_exc_raw), "phase_inh_raw": float(phase_inh_raw),
            "phase_aff_raw": float(phase_aff_raw), "fit_cutoff_ms": float(FIT_CUTOFF),
        })
    return out


# ---- result records, with the fields (and field order) of the stored sweep files ----------

FIT_KEYS = ("amp_exc", "phase_exc_deg", "offset_exc", "amp_inh", "phase_inh_deg", "offset_inh",
            "amp_exc_norm", "amp_inh_norm")


def impulse_record(b_val, rate, EL_inh, EL_exc, A, period, fit):
    """Record of the Delta_EL and b sweeps."""
    rec = {"b_val": b_val, "rate": rate, "EL_inh": EL_inh, "EL_exc": EL_exc, "A": A,
           "period": period}
    rec.update({k: fit[k] for k in FIT_KEYS})
    return rec


def gl_record(b_val, rate, EL_inh, EL_exc, gL_inh, gL_exc, A, period, fit):
    """Record of the Delta g_L sweep (delta_gL = gL_exc - gL_inh, nS)."""
    rec = {"b_val": b_val, "rate": rate, "EL_inh": EL_inh, "EL_exc": EL_exc,
           "gL_inh": gL_inh, "gL_exc": gL_exc, "delta_gL": gL_exc - gL_inh, "A": A,
           "period": period}
    rec.update({k: fit[k] for k in FIT_KEYS})
    return rec


def synapse_record(b_val, rate, syn, A, period, fit):
    """Record of the quantal-conductance sweeps; `syn` holds Qee..Qii (nS) and Tee..Tii (ms)."""
    rec = {"b_val": b_val, "rate": rate}
    rec.update({k: (float(syn[k]) * nS) / nS for k in ("Qee", "Qei", "Qie", "Qii")})
    rec.update({k: (float(syn[k]) * ms) / ms for k in ("Tee", "Tei", "Tie", "Tii")})
    rec.update({"A": A, "period": period})
    rec.update({k: fit[k] for k in FIT_KEYS})
    return rec


def load_records(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _equal(v, w):
    if isinstance(v, np.ndarray) or isinstance(w, np.ndarray):
        return np.shape(v) == np.shape(w) and np.array_equal(np.asarray(v), np.asarray(w),
                                                             equal_nan=True)
    if isinstance(v, float) and np.isnan(v):
        return isinstance(w, float) and np.isnan(w)
    return v == w


def _rel_diff(v, w):
    try:
        v, w = np.asarray(v, float), np.asarray(w, float)
        if v.shape != w.shape:
            return np.inf
        d = np.abs(v - w)[np.isfinite(v) | np.isfinite(w)]
        return float(np.max(d / np.maximum(np.abs(w[np.isfinite(v) | np.isfinite(w)]), 1e-300),
                            initial=0.0))
    except (TypeError, ValueError):
        return np.nan


def report_comparison(new, stored):
    """Compare every field of a stored record with a recomputed one; returns True if all are
    exactly equal."""
    same = True
    for k, v in stored.items():
        eq = k in new and _equal(v, new[k])
        same &= eq
        print(f"  {k:16s} {'identical' if eq else 'max rel. diff %.1e' % _rel_diff(new.get(k), v)}")
    print("all stored fields reproduced exactly" if same
          else "NOT all stored fields reproduced exactly")
    return same

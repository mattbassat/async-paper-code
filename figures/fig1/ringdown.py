"""Fit of the LFP response around the offset of a sinusoidal drive (Fig. 1h).

During the drive the signal is fitted by a sinusoid at the drive frequency (steady state, from
1 s after the drive onset to its offset).  After the offset it is fitted by a damped sinusoid
that starts from the steady-state amplitude and phase at the offset time, with the decay time
tau and the ringing frequency omega_d as the free parameters (nested bounded 1-D searches,
omega_d within 0.5-1.5 times the drive frequency).  Times are in ms, angular frequencies in
rad/ms.
"""
import numpy as np
from scipy.optimize import minimize_scalar


def fit_sinusoid(t, x, period):
    """Least-squares fit of baseline + amp*cos(omega*t + phase) to (t, x).
    The phase is in the same absolute-time convention as t."""
    omega = 2 * np.pi / period
    X = np.column_stack([np.cos(omega * t), np.sin(omega * t), np.ones_like(t)])
    coeffs, _, _, _ = np.linalg.lstsq(X, x, rcond=None)
    A_cos, A_sin, offset = coeffs
    amp = np.sqrt(A_cos ** 2 + A_sin ** 2)
    phase = np.arctan2(-A_sin, A_cos)
    return float(amp), float(phase), float(offset)


def damped_sinusoid(t, tau, omega_d, amp, phase0, baseline, t_off):
    return baseline + amp * np.exp(-(t - t_off) / tau) * np.cos(omega_d * (t - t_off) + phase0)


def fit_tau_and_omega_d(t, x, amp, phase0, baseline, t_off, tau_bounds, omega_d_bounds):
    """Outer bounded search over omega_d; for each omega_d, inner bounded search for the best tau
    (least squares)."""
    def sse_given_omega_d(omega_d):
        def sse(tau):
            pred = damped_sinusoid(t, tau, omega_d, amp, phase0, baseline, t_off)
            return np.sum((x - pred) ** 2)
        res = minimize_scalar(sse, bounds=tau_bounds, method="bounded")
        return res.x, sse(res.x)

    omega_d_fit = minimize_scalar(lambda w: sse_given_omega_d(w)[1], bounds=omega_d_bounds,
                                  method="bounded").x
    tau_fit, _ = sse_given_omega_d(omega_d_fit)
    return float(tau_fit), float(omega_d_fit)


def analyze_ringdown(t_ms, x, period_ms, t_onset_ms, t_offset_ms, t_end_ms,
                     omega_d_frac_bounds=(0.5, 1.5)):
    """Steady-state sinusoid fit during the drive, then the damped-sinusoid fit after its offset."""
    t_ms = np.asarray(t_ms, dtype=float)
    x = np.asarray(x, dtype=float)

    ss_mask = (t_ms >= t_onset_ms + 1000.0) & (t_ms <= t_offset_ms)
    amp, phase_raw, baseline = fit_sinusoid(t_ms[ss_mask], x[ss_mask], period_ms)

    omega_drive = 2 * np.pi / period_ms
    post_mask = (t_ms >= t_offset_ms) & (t_ms <= t_end_ms)
    tau_bounds = (period_ms / 20, 10 * (t_end_ms - t_offset_ms))
    omega_d_bounds = (omega_d_frac_bounds[0] * omega_drive, omega_d_frac_bounds[1] * omega_drive)

    phase0 = omega_drive * t_offset_ms + phase_raw
    tau_ms, omega_d = fit_tau_and_omega_d(t_ms[post_mask], x[post_mask], amp, phase0, baseline,
                                          t_offset_ms, tau_bounds, omega_d_bounds)
    return dict(amp=amp, phase_rad=phase_raw, baseline=baseline, tau_ms=tau_ms, omega_d=omega_d,
                phase0_rad=phase0, omega_drive=omega_drive)


def ringdown_curve(t_ms, fit, t_offset_ms):
    """The fitted curve: the steady-state sinusoid before the offset, the damped sinusoid after."""
    t_ms = np.asarray(t_ms, dtype=float)
    pre = t_ms < t_offset_ms
    y = np.empty_like(t_ms)
    y[pre] = fit["baseline"] + fit["amp"] * np.cos(fit["omega_drive"] * t_ms[pre] + fit["phase_rad"])
    y[~pre] = damped_sinusoid(t_ms[~pre], fit["tau_ms"], fit["omega_d"], fit["amp"],
                              fit["phase0_rad"], fit["baseline"], t_offset_ms)
    return y

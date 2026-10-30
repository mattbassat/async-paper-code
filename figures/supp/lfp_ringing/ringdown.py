"""Ringing of the LFP after the offset of a sinusoidal drive, and its number of significant
cycles (after Pesnot Lerousseau et al., 2021)."""
import numpy as np
from scipy.optimize import minimize_scalar

BASELINE_START_MS = 1000.0     # pre-onset window used for the noise level: [1 s, drive onset)
TRANSIENT_MS = 1000.0          # steady-state fit starts this long after the drive onset
OMEGA_D_FRAC_BOUNDS = (0.5, 1.5)
BOUND_TOL = 1e-3               # relative distance to a search bound counted as "on the bound"


def fit_sinusoid(t, x, period):
    """Least-squares fit of baseline + amp*cos(omega*t + phase) to (t, x). The phase is in the
    same absolute-time convention as t."""
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
    """Outer bounded search over omega_d; for each omega_d, inner bounded search for the best
    tau."""
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


def significant_cycles(amp, tau, omega_d_hz, baseline_std, k_sigma):
    """Ringing periods before amp * exp(-t / tau) drops below k_sigma * baseline_std."""
    if (not np.isfinite(baseline_std) or baseline_std <= 0
            or amp <= k_sigma * baseline_std or not np.isfinite(amp) or not np.isfinite(tau)):
        return 0.0
    period_d_ms = 1000.0 / omega_d_hz
    t_thresh_ms = tau * np.log(amp / (k_sigma * baseline_std))
    return float(max(0.0, t_thresh_ms / period_d_ms))


def analyze_ringdown(t_ms, x, period_ms, t_onset_ms, t_offset_ms, t_end_ms, k_sigma):
    """Steady-state and ringdown fits of one trace, and its number of significant cycles."""
    t_ms = np.asarray(t_ms, dtype=float)
    x = np.asarray(x, dtype=float)

    baseline_mask = (t_ms >= BASELINE_START_MS) & (t_ms < t_onset_ms)
    baseline_std = float(np.std(x[baseline_mask]))

    ss_mask = (t_ms >= t_onset_ms + TRANSIENT_MS) & (t_ms <= t_offset_ms)
    amp, phase_raw, baseline = fit_sinusoid(t_ms[ss_mask], x[ss_mask], period_ms)

    omega_drive = 2 * np.pi / period_ms
    post_mask = (t_ms >= t_offset_ms) & (t_ms <= t_end_ms)
    tau_bounds = (period_ms / 20, 10 * (t_end_ms - t_offset_ms))
    omega_d_bounds = (OMEGA_D_FRAC_BOUNDS[0] * omega_drive, OMEGA_D_FRAC_BOUNDS[1] * omega_drive)

    phase0 = omega_drive * t_offset_ms + phase_raw
    tau_ms, omega_d = fit_tau_and_omega_d(t_ms[post_mask], x[post_mask], amp, phase0, baseline,
                                          t_offset_ms, tau_bounds, omega_d_bounds)
    omega_d_hz = omega_d * 1000 / (2 * np.pi)

    lo, hi = omega_d_bounds
    at_bound = (abs(omega_d - lo) <= BOUND_TOL * abs(lo)) or (abs(omega_d - hi) <= BOUND_TOL * abs(hi))
    n_sig = 0.0 if at_bound else significant_cycles(amp, tau_ms, omega_d_hz, baseline_std, k_sigma)

    return dict(amp=amp, phase_deg=float(np.degrees(phase_raw)), baseline=baseline,
                baseline_std=baseline_std, tau_ms=tau_ms, omega_d_hz=omega_d_hz,
                n_sig_cycles=n_sig, phase0_rad=phase0, omega_drive_rad_ms=omega_drive)


def ringdown_curve(t_ms, fit, t_offset_ms):
    """The fitted curve on t_ms: the steady-state sinusoid before the offset, the damped
    sinusoid from the offset on."""
    t_ms = np.asarray(t_ms, dtype=float)
    pre = t_ms < t_offset_ms
    y = np.empty_like(t_ms)
    y[pre] = fit["baseline"] + fit["amp"] * np.cos(fit["omega_drive_rad_ms"] * t_ms[pre]
                                                   + np.radians(fit["phase_deg"]))
    omega_d = fit["omega_d_hz"] * 2 * np.pi / 1000.0
    y[~pre] = damped_sinusoid(t_ms[~pre], fit["tau_ms"], omega_d, fit["amp"],
                              fit["phase0_rad"], fit["baseline"], t_offset_ms)
    return t_ms, y


def significant_cycles_summary(run, k_sigma):
    """Mean and SEM over seeds of the number of significant cycles, per Delta_EL."""
    means, sems = [], []
    for i in range(len(run["delta_EL"])):
        vals = np.array([analyze_ringdown(run["t_state"], lfp, run["period_ms"],
                                          run["oscillation_onset"], run["oscillation_offset"],
                                          run["total_ms"], k_sigma)["n_sig_cycles"]
                         for lfp in run["lfp_inh"][i]], dtype=float)
        n_valid = max(1, np.sum(np.isfinite(vals)))
        means.append(np.nanmean(vals))
        sems.append(np.nanstd(vals) / np.sqrt(n_valid))
    return np.asarray(run["delta_EL"]), np.array(means), np.array(sems)

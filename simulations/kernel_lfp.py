"""Kernel-method LFP (Telenczuk, Telenczuk & Destexhe, 2020, J Neurosci Methods 344:108871).

Every spike adds a Gaussian unitary LFP at a virtual electrode, with an amplitude that decays
exponentially with the cell's distance from the electrode and a peak delayed by a constant
synaptic delay plus axonal propagation ("soma" layer amplitudes).
"""
import numpy as np
from scipy.signal import fftconvolve

AXONAL_VELOCITY_MM_PER_S = 200.0
SPACE_CONSTANT_MM = 0.2              # amplitude decay with distance
DELAY_CONST_MS = 10.4                # constant delay of the unitary LFP peak
SIGMA_I_MS = 2.1                     # width of the inhibitory unitary LFP
SIGMA_E_MS = 1.5 * SIGMA_I_MS        # excitatory unitary LFPs are 1.5 times slower
# (inhibitory, excitatory) unitary-LFP peak amplitudes (uV) by depth, Telenczuk et al. Table 1
LAYER_AMPLITUDES_UV = {"deep": (-0.2, -0.16), "soma": (3.0, 0.48),
                       "superficial": (-1.2, 0.24), "surface": (0.3, -0.08)}
PLANE_SIZE_MM = 0.2                  # cells placed uniformly in a 0.2 x 0.2 mm square
POSITION_SEED = 20240601             # one fixed layout for every run


def positions(n_exc, n_inh, plane_size_mm=PLANE_SIZE_MM, seed=POSITION_SEED):
    """(x, y) positions (mm) of the excitatory and inhibitory cells, drawn with a fixed seed."""
    rng = np.random.default_rng(seed)
    return rng.random((n_exc, 2)) * plane_size_mm, rng.random((n_inh, 2)) * plane_size_mm


def population_lfp(spike_i, spike_t_ms, pos_mm, amp0_uV, sigma_ms, n_bins, dt_ms):
    """Sum of the delayed, distance-weighted Gaussians of one population's spikes."""
    if len(spike_t_ms) == 0:
        return np.zeros(n_bins)
    c = PLANE_SIZE_MM / 2.0
    dist = np.hypot(pos_mm[:, 0] - c, pos_mm[:, 1] - c)
    amp = amp0_uV * np.exp(-dist / SPACE_CONSTANT_MM)
    delay = DELAY_CONST_MS + dist / AXONAL_VELOCITY_MM_PER_S * 1000.0
    spike_i = np.asarray(spike_i, dtype=int)
    peak_t = np.asarray(spike_t_ms, dtype=float) + delay[spike_i]
    half = max(1, int(np.ceil(6.0 * sigma_ms / dt_ms)))
    kernel = np.exp(-(np.arange(-half, half + 1) * dt_ms) ** 2 / (2.0 * sigma_ms ** 2))
    idx = np.round(peak_t / dt_ms).astype(int) + half
    ok = (idx >= 0) & (idx < n_bins + 2 * half)
    train = np.zeros(n_bins + 2 * half)
    np.add.at(train, idx[ok], amp[spike_i][ok])
    return fftconvolve(train, kernel, mode="same")[half:half + n_bins]


def kernel_lfp(spikes_exc, spikes_inh, n_exc, n_inh, total_ms, dt_ms, layer="soma"):
    """Kernel LFP (uV) over [0, total_ms) at dt_ms from (indices, times in ms) of each population."""
    amp_i, amp_e = LAYER_AMPLITUDES_UV[layer]
    n_bins = int(round(total_ms / dt_ms))
    pos_exc, pos_inh = positions(n_exc, n_inh)
    lfp_e = population_lfp(*spikes_exc, pos_exc, amp_e, SIGMA_E_MS, n_bins, dt_ms)
    lfp_i = population_lfp(*spikes_inh, pos_inh, amp_i, SIGMA_I_MS, n_bins, dt_ms)
    return (lfp_e + lfp_i).astype(np.float32)

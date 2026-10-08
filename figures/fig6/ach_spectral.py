"""Spectral analysis of the simulated acetylcholine-like transient (Fig. 6c).

The network's adaptation b ramps from 75 pA to 10 pA over a 1 s "cue" (2500-3500 ms), a model
of cue-evoked cholinergic release.  The analysis follows the protocol of Howe et al. (2017):

Spectrogram and inset show one quantity: the power at each frequency, z-scored against its own
mean and s.d. over the reference period (the 1000 ms immediately before cue onset), separately
for each run, then averaged over runs.  The spectrogram shows it as colour; the inset is the
same z-scored spectrogram averaged over the time axis across the inset window, drawn as the
mean over runs with a 95% confidence interval (1.96 x s.e.m.).

The time-frequency estimate is the multitaper method of Howe et al. (Chronux mtspecgramc):
a 500 ms sliding window with 20% overlap (400 ms step), DPSS tapers with
NW = T W = 0.5 s x 3 Hz = 1.5 (a +/-3 Hz frequency smear) and K = 2 NW - 1 = 2 tapers, the K
periodograms averaged.  Each cell has only 2K = 4 degrees of freedom per run, so 63 runs are
averaged, matching the 63 trials of Howe et al.  At a 400 ms step only two windows fall inside
the 1 s reference period, so the reference mean and s.d. are estimated from the same estimator
re-stepped at 50 ms over the same 1 s of data; the displayed spectrogram is unaffected.

The inset window is the post-cue period (3500-5000 ms): the modelled cholinergic transient
(the b ramp) completes only at cue offset, so this is the epoch that corresponds
physiologically to the late-cue window of Howe et al.

The pre-/post-cue spectra are Welch PSDs of dedicated spontaneous runs at b = 75 and b = 10 pA.

Every function that reads an LFP takes a `signal` argument, a function returning the LFP proxy
from one stored run, so the same protocol can be applied to alternative LFP proxies.
"""
import numpy as np
from scipy.signal.windows import dpss

from asyncresp.analysis import welch_psd
from asyncresp.io import load_pickle
from asyncresp.paths import DRIVEN, SPONTANEOUS

CUE_DIR = DRIVEN / "ach_cue"
SPONT_DIR = SPONTANEOUS / "ach"

CUE_SEEDS = list(range(300, 363))                       # 63 cue runs
SEEDS = list(range(300, 310))                           # 10 spontaneous runs per b
B_PRE, B_POST = 75.0, 10.0                              # adaptation b before / after the cue (pA)
T_CUE_START, T_CUE_END, TOTAL_MS = 2500.0, 3500.0, 5000.0
BASELINE = (T_CUE_START - 1000.0, T_CUE_START)          # z-score reference period (ms)
INSET_WINDOW = (T_CUE_END, TOTAL_MS)                    # post-cue window of the inset (ms)
SPONT_WINDOW = (2000.0, 10000.0)                        # Welch window of the spontaneous runs (ms)
FMAX = 100.0                                            # highest frequency analysed (Hz)

WINDOW_MS, OVERLAP = 500.0, 0.20
STEP_MS = WINDOW_MS * (1 - OVERLAP)
NW = (WINDOW_MS / 1000.0) * 3.0                         # +/- 3 Hz frequency smear
K_TAPERS = int(np.floor(2 * NW - 1))
BASE_STEP_MS = 50.0                                     # window step for the reference statistics


def lfp_inh(o):
    """The main LFP proxy: sum |G_I (E_I - V_ref)| over 500 RS cells."""
    return o["lfp_inh"]


def multitaper_spectrogram(sig, dt_ms, window_ms=WINDOW_MS, step_ms=STEP_MS, fmax=FMAX):
    """Multitaper spectrogram equivalent to Chronux mtspecgramc: in each window the
    mean-removed segment is multiplied by K DPSS tapers, each tapered segment is Fourier
    transformed and squared, and the K periodograms are averaged.  No zero padding.

    Returns frequencies (Hz, 0 < f <= fmax), window centres (ms) and power (freq x time)."""
    sig = np.asarray(sig, float)
    n_win, n_step = int(round(window_ms / dt_ms)), int(round(step_ms / dt_ms))
    tap, _ = dpss(n_win, NW, Kmax=K_TAPERS, sym=False, return_ratios=True)
    starts = range(0, len(sig) - n_win + 1, n_step)
    freqs = np.fft.rfftfreq(n_win, d=dt_ms / 1000.0)
    fm = (freqs <= fmax) & (freqs > 0)     # the DC bin of a mean-removed segment is zero
    freqs = freqs[fm]
    power, t_ms = [], []
    for s0 in starts:
        seg = sig[s0:s0 + n_win]
        seg = seg - seg.mean()
        spec = np.fft.rfft(tap * seg[None, :], axis=1)[:, fm]
        power.append((np.abs(spec) ** 2).mean(axis=0))
        t_ms.append((s0 + n_win / 2.0) * dt_ms)
    return freqs, np.asarray(t_ms), np.asarray(power).T


def seed_spectra(o, fmax=FMAX, signal=lfp_inh):
    """Spectrogram of one cue run, with the mean and s.d. of the power over its 1 s reference
    period (one value per frequency)."""
    dt = o["dt_lfp_ms"]
    sig = np.asarray(signal(o), float)[:int(round(TOTAL_MS / dt))]
    freqs, t_ms, power = multitaper_spectrogram(sig, dt, fmax=fmax)
    b0, b1 = int(round(BASELINE[0] / dt)), int(round(BASELINE[1] / dt))
    _, _, base = multitaper_spectrogram(sig[b0:b1], dt, step_ms=BASE_STEP_MS, fmax=fmax)
    mu, sd = base.mean(axis=1, keepdims=True), base.std(axis=1, ddof=1, keepdims=True)
    return freqs, t_ms, power, mu, np.maximum(sd, 1e-12)


def driven_raws(seeds=CUE_SEEDS):
    """The stored cue runs (b ramped from 75 to 10 pA during the cue)."""
    return [load_pickle(CUE_DIR / f"b5_new_pair_b75_b10_{s}.pkl") for s in seeds]


def group_spectrogram(raws, window=INSET_WINDOW, signal=lfp_inh):
    """Per-run z-scored spectrograms averaged over runs (the spectrogram's colour), and the
    same z-scored spectrogram averaged over time across `window`, one spectrum per run (the
    inset).  Returns freqs, t_ms, mean z (freq x time), per-run inset spectra (run x freq) and
    the number of windows averaged for the inset."""
    zs, ins, freqs, t_ms = [], [], None, None
    for o in raws:
        freqs, t_ms, power, mu, sd = seed_spectra(o, signal=signal)
        z = (power - mu) / sd
        zs.append(z)
        m = (t_ms >= window[0]) & (t_ms < window[1])
        ins.append(z[:, m].mean(axis=1))
    return freqs, t_ms, np.stack(zs).mean(axis=0), np.stack(ins), int(m.sum())


def inset_stats(ins):
    """Mean over runs and the half-width of its 95% confidence interval (1.96 x s.e.m.)."""
    return ins.mean(axis=0), 1.96 * ins.std(axis=0, ddof=1) / np.sqrt(ins.shape[0])


def spont_paths(b_val, seeds=SEEDS):
    """The stored spontaneous runs at adaptation b = b_val (pA)."""
    return [SPONT_DIR / f"b5_new_dEL5_nu2_b{b_val:g}_{s}.pkl" for s in seeds]


def spont_psd(b_val, signal=lfp_inh, paths=None):
    """Welch PSD (density scaling; 4096-sample segments, 50% overlap) of the spontaneous LFP
    over SPONT_WINDOW, averaged over runs.  `paths` overrides the default runs at b_val."""
    psds, freqs = [], None
    for path in (paths or spont_paths(b_val)):
        o = load_pickle(path)
        freqs, p = welch_psd(signal(o), o["dt_lfp_ms"], SPONT_WINDOW)
        psds.append(p)
    return freqs, np.mean(psds, axis=0)

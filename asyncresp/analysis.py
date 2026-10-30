"""Analysis helpers shared by several figures."""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import welch

RESONANCE_FMIN = 10.0     # Hz; a response maximum at or below this frequency is not a resonance


def aggregate_impulse_results(impulse_results, param_name):
    """Mean and SD over seeds of the E and I response amplitudes, per (parameter, frequency)."""
    df = pd.DataFrame(impulse_results)
    if param_name == "delta_EL":
        df["delta_EL"] = df["EL_exc"] - df["EL_inh"]
    if param_name == "delta_gL":
        df["delta_gL"] = df["gL_inh"] - df["gL_exc"]
    df["frequency"] = 1000.0 / df["period"]
    return (
        df.groupby([param_name, "frequency"])
        .agg(amp_exc_mean=("amp_exc", "mean"), amp_exc_std=("amp_exc", "std"),
             amp_inh_mean=("amp_inh", "mean"), amp_inh_std=("amp_inh", "std"))
        .reset_index()
    )


def sampled_resonance_peaks(data, param_name, fmin=RESONANCE_FMIN):
    """Empirical resonance frequency of each network of a seed-averaged sweep: the sampled drive
    frequency with the largest mean E response, counted as a resonance only if it lies above
    `fmin` (None otherwise)."""
    out = {}
    for value, sub in data.groupby(param_name):
        sub = sub.sort_values("frequency")
        f_peak = float(sub["frequency"].values[int(sub["amp_exc_mean"].values.argmax())])
        out[value] = f_peak if f_peak > fmin else None
    return out


def welch_psd(x, dt_ms, window=None):
    """Welch power spectral density (density scaling) of a trace sampled every `dt_ms`:
    4096-sample segments (about 2 s at 2 kHz) with 50 % overlap and a Hann window. `window` =
    (start, end) in ms restricts the trace before the estimate."""
    if window is not None:
        i0, i1 = int(round(window[0] / dt_ms)), int(round(window[1] / dt_ms))
        x = x[i0:i1]
    return welch(x, fs=1000.0 / dt_ms, nperseg=min(4096, len(x)),
                 noverlap=min(2048, len(x) // 2), scaling="density")


def rebin(t, x, bin_ms):
    """Average a uniformly sampled trace (t in ms) over consecutive bins of `bin_ms`."""
    n = int(round(bin_ms / (t[1] - t[0])))
    m = (len(t) // n) * n
    return t[:m].reshape(-1, n).mean(1), x[:m].reshape(-1, n).mean(1)


def rocket_colors(values, reverse=False):
    """Discrete seaborn "rocket" palette keyed by the sorted parameter values (dark -> light).
    reverse=True flips it, so that light is always the resonant (UD) side of a sweep."""
    values = np.sort(np.unique(values))
    palette = sns.color_palette("rocket", n_colors=len(values))
    if reverse:
        palette = palette[::-1]
    return dict(zip(values, palette))

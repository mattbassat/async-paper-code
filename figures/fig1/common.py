"""Small helpers used by several Fig. 1 panels."""
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter
from asyncresp.analysis import rebin, rocket_colors, welch_psd  # noqa: F401  (re-exported)


def welch_psd_lfp(lfp, dt_lfp_ms):
    """Welch power spectral density of an LFP trace (see asyncresp.analysis.welch_psd)."""
    return welch_psd(lfp, dt_lfp_ms)


def plain_log_ticks(axis):
    """Label the decades of a logarithmic axis as plain numbers (1, 10, 100), minor ticks
    unlabelled."""
    axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    axis.set_minor_formatter(NullFormatter())

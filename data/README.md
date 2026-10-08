# Stored results

All files are Python pickles (`asyncresp.io.load_pickle`).  To keep the repository small, the
stored copies of the time-series simulations contain only the fields the figures use (spike
trains and unused LFP proxies were removed); the scripts in `simulations/` regenerate the
complete records.

| Folder | Contents | Used in |
|---|---|---|
| `sweeps/` | Frequency-response sweeps (`impulse_results_*`, lists of single-run records with the fitted E/I response amplitudes) and mean-field linear-stability tables (`eigen_df_*`, pandas DataFrames) | Figs. 1c-d, 2e, 3a, 5a-b |
| `synchrony/` | Mean pairwise spike-count correlation per Delta E_L | Fig. 1i |
| `spontaneous/adex_r2b0/` | Spontaneous inhibitory-current LFP proxy, Delta E_L = -15, -5, +5, +15 mV | Fig. 1e |
| `spontaneous/adex_params/` | The same for the extreme values of the other parameters | Fig. 1f |
| `spontaneous/hh/` | The same for the Hodgkin-Huxley network | Fig. 5c |
| `spontaneous/ach/` | The same at adaptation b = 75 and 10 pA | Fig. 6c |
| `driven/fig1_example_sim.pkl` | Example driven simulation (rates, spikes, LFP) | Fig. 1a |
| `driven/od_ud_55hz/` | OD/UD networks driven at 55 Hz (`g_*`: rates, 10 seeds; `h_*`: LFP around stimulus offset) | Figs. 1g-h |
| `driven/slow_input_examples/` | Responses to a 0.5 Hz input, Delta E_L = -15, -5, +5 mV | Fig. 3c |
| `driven/ach_cue/` | LFP proxy during the acetylcholine-like adaptation ramp, 63 runs | Fig. 6c |
| `mean_field/mf_burst_sim.pkl` | Spiking-network rates under a transient drive, for comparison with the mean field | Fig. 2b |

File-name conventions: `r2b0a1` = nu_drive 2 Hz, b 0 pA, drive amplitude 1 Hz; `m15`/`p5` =
Delta E_L of -15/+5 mV; the trailing number is the random seed.

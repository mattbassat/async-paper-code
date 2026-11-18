# Stored results

All files are Python pickles (`asyncresp.io.load_pickle`), except the CSV tables of
`supp/baseline/`. To keep the repository small, the
stored copies of the time-series simulations contain only the fields the figures use (spike
trains and unused LFP proxies were removed); the scripts in `simulations/` regenerate the
complete records.

Figures with letters (A-V) are in S1 Appendix.

| Folder | Contents | Used in |
|---|---|---|
| `sweeps/` | Frequency-response sweeps (`impulse_results_*`, lists of single-run records with the fitted E/I response amplitudes) and mean-field linear-stability tables (`eigen_df_*`, pandas DataFrames) | Figs. 1c-d, 2e, 3a, 5a-b, D, I, N-S |
| `synchrony/` | Mean pairwise spike-count correlation per Delta E_L | Fig. 1i |
| `spontaneous/adex_r2b0/` | Spontaneous inhibitory-current LFP proxy, Delta E_L = -15, -5, +5, +15 mV | Fig. 1e |
| `spontaneous/adex_params/` | The same for the extreme values of the other parameters | Fig. 1f |
| `spontaneous/hh/` | The same for the Hodgkin-Huxley network | Fig. 5c |
| `driven/fig1_example_sim.pkl` | Example driven simulation (rates, spikes, LFP) | Fig. 1a |
| `driven/od_ud_55hz/` | OD/UD networks driven at 55 Hz (`g_*`: rates, 10 seeds; `h_*`: LFP around stimulus offset) | Figs. 1g-h |
| `driven/slow_input_examples/` | Responses to a 0.5 Hz input, Delta E_L = -15, -5, +5 mV | Fig. 3c |
| `mean_field/mf_burst_sim.pkl` | Spiking-network rates under a transient drive, for comparison with the mean field | Fig. 2b |
| `supp/lfp_proxies/` | Welch spectra (per seed) of the spontaneous runs of Fig. 1e-f for six signals: inhibitory and excitatory current proxies, their absolute-magnitude sum, the kernel-method LFP and the E and I population rates | Fig. K |
| `supp/lfp_ringing/` | Inhibitory-current LFP proxy of networks driven at their resonance and switched off (no delay: 7 Delta E_L; 2 ms delay: 12 Delta E_L; 10 seeds each), and spontaneous LFP of the 2 ms-delay network; arrays of shape (Delta E_L, seed, time) | Figs. L, M |
| `supp/high_drive/` | Frequency-response sweep (records as in `sweeps/`) at nu_drive 8 Hz, b 0 pA, Delta E_L = -25 ... +15 mV, and its mean-field table | Fig. F |
| `supp/amplitude_2hz/` | Frequency-response sweeps of the three parameter sets at A = 2 Hz | Figs. E, V |
| `supp/el_exc/` | Frequency-response sweeps of the three parameter sets at E_L,exc = -65 and -75 mV | Figs. G, H |
| `supp/gl_shift/` | The Delta g_L frequency-response sweep of `sweeps/` (records as there) repeated at g_L,exc = 8 and 12 nS (g_L,inh = g_L,exc - 5 ... + 5 nS) | Fig. I |
| `supp/delays/` | Frequency-response sweeps (records as in `sweeps/`) of the nu_drive 4 Hz, b 0 pA network with a uniform 1 or 2 ms delay on all recurrent and afferent synapses; f = 0.5-200 Hz, 10 seeds | Fig. U |
| `supp/fine_grained/` | Slow-input (0.5 Hz) responses (records as in `sweeps/`) of the three parameter sets (`r2b0a1`, `r4b0a1`, `r4b60a1`) for Delta E_L in 0.1 mV steps around the responsiveness peak (-10 ... 0, -15 ... -5 and -10 ... 0 mV), 10 seeds | Fig. T |
| `supp/pulse_inputs/` | E, I and afferent rates (1 ms bins, float32, 10 seeds) of the nu_drive 2 Hz, b 0 pA network under a single half-cycle (`half_cycle.pkl`: Delta E_L = -15, -5, +5 mV) or two-cycle pulse (`two_cycle.pkl`: Delta E_L = +5 mV), A = 2 Hz, P = 2000, 200, 20 ms; arrays of shape (seed, time) within the plotted window | Figs. A, B |
| `supp/pulse_amplitude/` | Evoked E and I amplitudes (mean and SD over 10 seeds) for a 1000 ms half-cycle pulse vs Delta E_L = -15 ... +15 mV (1 mV steps; 0 mV listed twice, from the two parts of the sweep), one dict per A = 1, 5, 10, 25, 50 Hz | Fig. C |
| `supp/baseline/` | Baseline (undriven) metrics per run (3 parameter sets x 7 Delta E_L x 10 seeds; CSV): `baseline_metrics.csv` (E/I firing rates, E/I mean membrane potential, conductance ratio; seeds 12-21) and `current_ratios.csv` (afferent/total excitatory and inhibitory/excitatory current ratios, balance index beta, in E cells; seeds 0-9) | Fig. J |

File-name conventions: `r2b0a1` = nu_drive 2 Hz, b 0 pA, drive amplitude 1 Hz; `m15`/`p5` =
Delta E_L of -15/+5 mV; the trailing number is the random seed.

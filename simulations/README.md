# Simulations

These scripts regenerate the data files in `data/`. Running them is optional: the results are
already stored in the repository. Each script writes to the
same path and file name that the figure scripts read (pass `--out` to write elsewhere), and
most provide `--check`, which re-runs one stored simulation and compares it with the stored file.

| Script | Writes | Used in | Approximate cost (one core) |
|---|---|---|---|
| `run_frequency_sweeps.py` | `data/sweeps/impulse_results_{r2b0a1,r4b0a1,r4b60a1}_{final,extra}.pkl` | Figs. 1c, 2e, 3a | ~1.5 min per run; ~165 core-hours in total |
| `run_alt_parameter_sweeps.py` | `data/sweeps/impulse_results_{changing_b_r4,r2b0a1gLexc10_delta_gL,r2b0a1_Qee_*,r2b0a1_Qei_*}.pkl` | Fig. 1d | ~1.5 min per run; 50-80 core-hours per parameter |
| `run_examples.py` | `data/driven/fig1_example_sim.pkl`, `data/driven/slow_input_examples/`, `data/mean_field/mf_burst_sim.pkl` | Figs. 1a, 2b, 3c | ~1 h |
| `run_mean_field_eigenvalues.py` | `data/sweeps/eigen_df_{r2b0,r4b0,r4b60}_dense.pkl` | Fig. 3a (analytical curves) | ~1 min |
| `run_lfp_spontaneous.py` | `data/spontaneous/{adex_r2b0,adex_params}/` | Figs. 1e, 1f | ~1-1.5 min per run (120 runs) |
| `run_lfp_driven.py` | `data/driven/od_ud_55hz/` | Figs. 1g, 1h | ~0.5-1.5 min per run (22 runs) |
| `run_synchrony.py` | `data/synchrony/regimes_cc_r2b0.pkl` | Fig. 1i | ~1 min per run (70 runs) |
| `run_lfp_ringing.py` | `data/supp/lfp_ringing/` | Figs. S8, S9 | ~1.5 min per run (220 runs) |
| `run_delay_sweeps.py` | `data/supp/delays/impulse_results_r4b0a1delay{1,2}_extended.pkl` | Fig. S16 | ~1.5 min per run; 4320 runs, ~110 core-hours |
| `run_fine_grained.py` | `data/supp/fine_grained/impulse_results_fg_slow_{r2b0a1,r4b0a1,r4b60a1}_final.pkl` | Fig. S15 | ~1-1.5 min per run; 3030 runs, 50-75 core-hours |
| `run_supp_frequency_sweeps.py high_drive` | `data/supp/high_drive/impulse_results_r8b0a1{,suppr8}_final.pkl` | Fig. S6 | ~1.5 min per run; 2700 runs, ~70 core-hours |
| `run_supp_frequency_sweeps.py amplitude_2hz` | `data/supp/amplitude_2hz/impulse_results_{r2b0,r4b0,r4b60}a2_final.pkl` | Figs. S5, S19 | ~1.5 min per run; 6300 runs, ~160 core-hours |
| `run_supp_frequency_sweeps.py el_exc` | `data/supp/el_exc/impulse_results_{r2b0,r4b0,r4b60}a1EL{65,75}_final.pkl` | Figs. S17, S18 | ~1.5 min per run; 12180 runs, ~300 core-hours |
| `run_single_cell_traces.py` | `data/supp/single_cell_traces/traces.pkl` | Fig. 7 | ~20 s in total (4 single-cell runs) |
| `run_pulse_inputs.py` | `data/supp/pulse_inputs/{half_cycle,two_cycle}.pkl` | Figs. S1, S2 | ~15-25 s per run (120 runs) |
| `run_pulse_amplitude_sweep.py` | `data/supp/pulse_amplitude/sweep_summary_combined.pkl` | Fig. S3 | ~15-25 s per run; 1600 runs, 7-11 core-hours |
| `run_baseline_metrics.py` | `data/supp/baseline/{baseline_metrics,current_ratios}.csv` | Fig. S7 | ~0.5-2 min per run (4-6 GB each); 420 runs, 4-14 core-hours |
| `hh/run_hh_frequency_sweep.py` | `data/sweeps/impulse_results_hh_grid_final.pkl` | Figs. 5a, 5b | ~2 min per run; 120-150 core-hours |
| `hh/run_hh_spontaneous.py` | `data/spontaneous/hh/` | Fig. 5c | ~3.5 min per run (40 runs) |

All scripts accept `--workers N` to run simulations in parallel; see each script's `--help`.

Library modules:

- `adex_impulse.py`: the AdEx network under a sinusoidally modulated afferent drive, with the
  sinusoid fit of the evoked responses.
- `adex_lfp.py`: the same network recording the LFP proxy, for spontaneous activity, driven
  activity followed by a stimulus-free period, and spike trains for the synchrony analysis.
- `hh/`: the Hodgkin-Huxley network (adapted from Carlu et al., 2020).

## Reproducibility

- Simulations are seeded. Brian2's random-number streams depend on the code-generation target,
  so each script sets the target with which the stored results were computed; with it the
  simulated network activity is reproduced exactly.
- Fitted quantities (least-squares sinusoid fits) can differ in the last bits between
  linear-algebra library builds. The scripts select OpenBLAS's generic kernels
  (`OPENBLAS_CORETYPE=Prescott`); remaining differences are at the 1e-13 relative level.
- The spike-count correlations of `run_synchrony.py` are computed on a random sample of 1000
  neurons drawn without a fixed seed, as in the original analysis, so re-runs agree with the
  stored values within sampling error (of order 1e-4) rather than exactly.
- Likewise the CV and CC fields of the full records of `run_baseline_metrics.py` (not used by
  any figure) are computed on unseeded random samples of neurons.
- The Hodgkin-Huxley scripts use Brian2's C++ standalone mode and need a C++ compiler; the
  Cython target used by some AdEx scripts also needs one.

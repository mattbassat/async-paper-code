# Simulations

These scripts regenerate every data file in `data/` that the main-text figures read.  Running
them is optional: the results are already stored in the repository.  Each script writes to the
same path and file name that the figure scripts read (pass `--out` to write elsewhere), and
most provide `--check`, which re-runs one stored simulation and compares it with the stored file.

| Script | Writes | Used in | Approximate cost (one core) |
|---|---|---|---|
| `run_frequency_sweeps.py` | `data/sweeps/impulse_results_{r2b0a1,r4b0a1,r4b60a1}_{final,extra}.pkl` | Figs. 1c, 2e, 3a | ~1.5 min per run; ~165 core-hours in total |
| `run_alt_parameter_sweeps.py` | `data/sweeps/impulse_results_{changing_b,r2b0a1gLexc10_delta_gL,r2b0a1_Qee_*,r2b0a1_Qei_*}.pkl` | Fig. 1d | ~1.5 min per run; 50-80 core-hours per parameter |
| `run_examples.py` | `data/driven/fig1_example_sim.pkl`, `data/driven/slow_input_examples/`, `data/mean_field/mf_burst_sim.pkl` | Figs. 1a, 2b, 3c | ~1 h |
| `run_mean_field_eigenvalues.py` | `data/sweeps/eigen_df_{r2b0,r4b0,r4b60}_dense.pkl` | Fig. 3a (analytical curves) | ~1 min |
| `run_lfp_spontaneous.py` | `data/spontaneous/{adex_r2b0,adex_params,ach}/` | Figs. 1e, 1f, 6c | ~1-1.5 min per run (140 runs) |
| `run_lfp_driven.py` | `data/driven/od_ud_55hz/` | Figs. 1g, 1h | ~0.5-1.5 min per run (22 runs) |
| `run_lfp_ach_cue.py` | `data/driven/ach_cue/` | Fig. 6c | ~1 min per run (63 runs) |
| `run_synchrony.py` | `data/synchrony/regimes_cc_r2b0.pkl` | Fig. 1i | ~1 min per run (70 runs) |
| `hh/run_hh_frequency_sweep.py` | `data/sweeps/impulse_results_hh_grid_final.pkl` | Figs. 5a, 5b | ~2 min per run; 120-150 core-hours |
| `hh/run_hh_spontaneous.py` | `data/spontaneous/hh/` | Fig. 5c | ~3.5 min per run (40 runs) |

All scripts accept `--workers N` to run simulations in parallel; see each script's `--help`.

Library modules:

- `adex_impulse.py`: the AdEx network under a sinusoidally modulated afferent drive, with the
  sinusoid fit of the evoked responses.
- `adex_lfp.py`: the same network recording the LFP proxies, for spontaneous activity, driven
  activity followed by a stimulus-free period, the acetylcholine-like adaptation ramp, and
  spike trains for the synchrony analysis.
- `hh/`: the Hodgkin-Huxley network (adapted from Carlu et al., 2020).

## Reproducibility

- Simulations are seeded.  Brian2's random-number streams depend on the code-generation target,
  so each script sets the target with which the stored results were computed; with it the
  simulated network activity is reproduced exactly.
- Fitted quantities (least-squares sinusoid fits) can differ in the last bits between
  linear-algebra library builds.  The scripts select OpenBLAS's generic kernels
  (`OPENBLAS_CORETYPE=Prescott`); remaining differences are at the 1e-13 relative level.
- The spike-count correlations of `run_synchrony.py` are computed on a random sample of 1000
  neurons drawn without a fixed seed, as in the original analysis, so re-runs agree with the
  stored values within sampling error (of order 1e-4) rather than exactly.
- The Hodgkin-Huxley scripts use Brian2's C++ standalone mode and need a C++ compiler; the
  Cython target used by some AdEx scripts also needs one.

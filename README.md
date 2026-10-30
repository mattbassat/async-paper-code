# A continuum of asynchronous states in cerebral cortex networks, and their responsiveness to external inputs

Code and data for

> Bassat, M., Tesler, F. & Destexhe, A. A continuum of asynchronous states in cerebral cortex
> networks, and their responsiveness to external inputs.

We simulate spiking networks of adaptive exponential integrate-and-fire (AdEx) regular-spiking
and fast-spiking neurons (a layer 2/3 cortical circuit) in [Brian2](https://brian2.readthedocs.io),
and analyse their responsiveness to oscillatory inputs alongside a two-population mean-field
model (di Volo et al., 2019). A Hodgkin-Huxley version of the network is used to check that the
results generalise.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python figures/make_all_figures.py          # main-text figures -> output/
python figures/make_all_figures.py supp     # supplementary figures -> output/supp/
python figures/fig3/make_fig3.py            # or one figure at a time
```

All figures are built from the stored results in `data/`, so no simulations are needed (Fig. 2
takes about 1.5 min, the rest well under a minute each). The stored results can be regenerated
with the scripts in `simulations/` (see `simulations/README.md`), which takes several hundred
core-hours in total.

## Layout

```
asyncresp/      shared code: paths, loading, mean-field model, analysis helpers, colours
figures/        one folder per main-text figure, plus supp/ for the supplementary figures
simulations/    scripts that generate the stored results (AdEx and Hodgkin-Huxley networks)
data/           stored simulation results (see data/README.md)
output/         figures are written here (not tracked)
```

## Main-text figures

| Figure | Script |
|---|---|
| Fig. 1: OD and UD regimes, LFP signatures, synchrony | `figures/fig1/make_fig1.py` |
| Fig. 2: mean-field reduction and linear stability | `figures/fig2/make_fig2.py` |
| Fig. 3: responsiveness to slow inputs | `figures/fig3/make_fig3.py` |
| Fig. 4: mean-field transfer functions | `figures/fig4/make_fig4.py` |
| Fig. 5: Hodgkin-Huxley network | `figures/fig5/make_fig5.py` |
| Fig. 6 (Methods): state-dependent mean-field timescale T | `figures/fig6/make_fig6.py` |
| Fig. 7 (Methods): single AdEx and HH cell traces | `figures/fig7/make_fig7.py` |

## Supplementary figures

All in `figures/supp/`, written to `output/supp/`.

| Figure | Script (output) |
|---|---|
| S1: half-cycle pulse inputs | `make_figS_pulses.py` (`figS_half_cycle`) |
| S2: two-cycle pulse inputs | `make_figS_pulses.py` (`figS_two_cycle`) |
| S3: pulse amplitude | `make_figS_pulses.py` (`figS_pulse_amplitude`) |
| S4: parameter sets, A = 1 Hz | `make_figS_param_sets.py` (`figS_param_sets_A1`) |
| S5: parameter sets, A = 2 Hz | `make_figS_param_sets.py` (`figS_param_sets_A2`) |
| S6: higher drive (8 Hz) | `make_figS_high_drive.py` |
| S7: baseline activity metrics | `make_figS_baseline.py` |
| S8: LFP ringing, no delay | `lfp_ringing/make_figS_lfp_ringing.py` (`figS_lfp_ringing_nodelay`) |
| S9: LFP ringing, 2 ms delay | `lfp_ringing/make_figS_lfp_ringing.py` (`figS_lfp_ringing_delay`) |
| S10: static vs state-dependent T | `make_figS_static_T.py` |
| S11: alternative E/I balance parameters | `make_figS_alt_params.py` |
| S12: gains and det(J*) | `make_figS_gains.py` (`figS_gains`) |
| S13: discriminant | `make_figS_gains.py` (`figS_discriminant`) |
| S14: logarithmic derivatives | `make_figS_gains.py` (`figS_log_derivatives`) |
| S15: fine-grained slow-input responses | `make_figS_fine_grained.py` |
| S16: synaptic delays | `make_figS_delays.py` |
| S17: E_L,exc = -65 mV | `make_figS_param_sets.py` (`figS_param_sets_EL65`) |
| S18: E_L,exc = -75 mV | `make_figS_param_sets.py` (`figS_param_sets_EL75`) |
| S19: fluctuation-normalised responses | `make_figS_normalised.py` |

Fig. 4 and Figs. S12-S14 use only the mean-field model (no spiking-network data).

## Requirements

Python 3.11 with the versions in `requirements.txt`. The simulations also need a C++ compiler
(Brian2's Cython target and C++ standalone mode).

## Citation

If you use this code, please cite the paper above. The Hodgkin-Huxley network is adapted from
Carlu et al. (2020), *J. Neurophysiol.*; the mean-field transfer-function fits are those of
di Volo et al. (2019), *Neural Comput.*

# A continuum of asynchronous states in cerebral cortex networks, and their responsiveness to external inputs

Code and data to reproduce the main-text figures of

> Bassat, M., Tesler, F. & Destexhe, A. A continuum of asynchronous states in cerebral cortex
> networks, and their responsiveness to external inputs.

Spiking networks of adaptive exponential integrate-and-fire (AdEx) regular-spiking and
fast-spiking neurons, modelling a layer 2/3 cortical circuit, are simulated with
[Brian2](https://brian2.readthedocs.io).  Their responsiveness to oscillatory inputs is analysed
together with a two-population mean-field model (di Volo et al., 2019).  A Hodgkin-Huxley
version of the network is used to test the generality of the results.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python figures/make_all_figures.py          # all main-text figures -> output/
python figures/fig3/make_fig3.py            # or one figure at a time
```

The figures are built from the results stored in `data/`, so no simulation is needed
(Fig. 2 takes about 1.5 min because it computes the mean-field fixed points; the others take
under a minute each).  The stored simulation results can be regenerated with the scripts in
`simulations/` (see `simulations/README.md`); this takes several hundred core-hours in total.

## Repository structure

```
asyncresp/        shared code: paths, loading, mean-field model, analysis helpers, colours
figures/          one folder per figure (figN/make_figN.py plus the panel modules)
simulations/      scripts that generate the stored results (AdEx and Hodgkin-Huxley networks)
data/             stored simulation results read by the figure scripts (see data/README.md)
output/           figures are written here (not tracked)
```

| Figure | Script | Content |
|---|---|---|
| Fig. 1 | `figures/fig1/make_fig1.py` | Tuning E/I balance: low-pass (OD) and resonant (UD) regimes, LFP signatures, synchrony |
| Fig. 2 | `figures/fig2/make_fig2.py` | Mean-field reduction, linear stability, resonant frequencies and the balance metric |
| Fig. 3 | `figures/fig3/make_fig3.py` | Responsiveness to slow inputs and its analytical estimate |
| Fig. 4 | `figures/fig4/make_fig4.py` | Mean-field transfer functions of overdamped, boundary and underdamped networks |
| Fig. 5 | `figures/fig5/make_fig5.py` | Hodgkin-Huxley network |
| Fig. 6 | `figures/fig6/make_fig6.py` | Acetylcholine case study (simulated panels) |

Fig. 6 sets the simulations beside published experimental results (Howe et al., 2017,
*J. Neurosci.* 37:3215-3230: their Figs. 3B, 5C and 6C).  Those panels are third-party material
and are not reproduced here: `make_fig6.py` builds only the simulated panels
(`output/fig6_simulated.png`), at the size they occupy in the published figure.

Fig. 4 is computed directly from the mean-field transfer function and needs no stored data.

## Requirements

Python 3.11 with the package versions in `requirements.txt`.  The simulations additionally need
a C++ compiler (Brian2's Cython target and C++ standalone mode).

## Citation and acknowledgements

If you use this code, please cite the paper above.  The Hodgkin-Huxley network code is adapted
from Carlu et al. (2020), *J. Neurophysiol.*  The mean-field transfer-function fits are those of
di Volo et al. (2019), *Neural Comput.*

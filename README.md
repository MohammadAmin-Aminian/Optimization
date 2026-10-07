# ComPy Inversion Tuner

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**Reproducible hyperparameter optimization for seafloor-compliance inversion.**

[![Regression tests](https://github.com/MohammadAmin-Aminian/Optimization/actions/workflows/tests.yml/badge.svg)](https://github.com/MohammadAmin-Aminian/Optimization/actions/workflows/tests.yml)
[![ComPy](https://img.shields.io/badge/uses-ComPy-2f6f9f)](https://github.com/MohammadAmin-Aminian/ComPy)

This repository is a standalone optimization layer for the Monte Carlo compliance-inversion workflow used with **ComPy**. It uses Optuna to search proposal scales for shear velocity and layer thickness together with the roughness-regularization weight, while keeping the inversion itself in ComPy.

It grew out of the numerical workflow used for seafloor-compliance analysis of RHUM-RUM ocean-bottom seismic stations in the Indian Ocean. The scientific application is described in [Aminian et al. (2025), Geophysical Journal International](https://doi.org/10.1093/gji/ggaf253).

## Why this project exists

A Metropolis sampler can return very different practical performance depending on its proposal scales and regularization. Choosing those values manually is slow and difficult to reproduce. This project separates that tuning problem from the inversion code and makes it explicit, repeatable and testable.

The tool searches three controls:

| Parameter | Meaning |
|---|---|
| `Vs_Step` | proposal scale for shear velocity |
| `H_Step` | proposal scale for layer thickness |
| `Alpha` | weight of the roughness regularization used by the inversion |

The optimization target is the **minimum post-burn-in misfit** returned by ComPy. This is a tuning criterion, not a convergence diagnostic or posterior-quality metric.

## Workflow

```text
measured compliance + uncertainty
            |
            v
      input validation
            |
            v
     Optuna trial sampler
            |
            v
  ComPy compliance inversion
            |
            v
 post-burn-in misfit score
            |
            v
 best proposal/regularization settings
```

## Installation

Python 3.10 or newer:

```bash
git clone https://github.com/MohammadAmin-Aminian/Optimization.git
cd Optimization
python -m pip install -e '.[dev]'
python -m pip install "git+https://github.com/MohammadAmin-Aminian/ComPy.git"
```

## Input data

Create an NPZ file containing the measured compliance, frequency vector and standard deviation:

```python
import numpy as np

np.savez(
    "observations.npz",
    data=compliance,
    frequency=frequency_hz,
    uncertainty=standard_deviation,
)
```

All arrays must be finite, non-empty, one-dimensional and equal in length. Frequencies and uncertainties must be positive. Compliance and uncertainty must use the same units as the ComPy forward model (s²/m).

## Run an optimization

```bash
compy-tune observations.npz \
    --trials 30 \
    --iterations 10000 \
    --burnin 500 \
    --station RR38 \
    --output optimization_results.npz
```

Useful options include `--depth`, `--layers`, `--seed`, `--iterations`, `--burnin` and `--station`. The historical script entry point remains available as `python Optimizing_Hyperparameters.py ...` for backward compatibility. Run `compy-tune --help` for the complete interface.

The output NPZ contains:

- `best_misfit`
- `Vs_Step`
- `H_Step`
- `Alpha`

Existing output files are refused rather than overwritten.

## Scientific context

Seafloor compliance measures vertical seafloor deformation relative to pressure forcing by long-period ocean waves. In the RHUM-RUM study, compliance in the infragravity band was used to constrain shallow shear-velocity structure beneath the Indian Ocean. Because the inversion is nonlinear and the shallow low-velocity structure dominates sensitivity, practical sampler behavior and regularization matter.

This repository addresses **sampler tuning only**. Forward modelling, layered elastic response, compliance calculation and model sampling belong to [ComPy](https://github.com/MohammadAmin-Aminian/ComPy).

## Reproducibility and validation

A fixed seed is supplied to both Optuna and the ComPy sampler. The code validates shapes, finite values, positive frequencies/uncertainties, burn-in bounds, layer count and depth before starting expensive work.

Run:

```bash
python -m pytest -q
```

The test suite checks input validation, uncertainty propagation into the inverter, post-burn-in scoring and the optimization interface. GitHub Actions runs the tests against a fixed ComPy commit so interface changes are visible.

## What this project does not claim

- The lowest short-chain misfit is **not** proof of MCMC convergence.
- Optimized proposal scales are not universal physical parameters.
- Short tuning chains should not replace longer independent production chains.
- Reproducibility of the random sequence does not guarantee identical floating-point results across every platform.
- Scientific interpretation still requires inspecting posterior behaviour, model sensitivity and data quality.

## Relationship to ComPy

This project is intentionally independent in purpose:

- **ComPy**: compliance processing, calibration, forward modelling and inversion.
- **ComPy Inversion Tuner**: systematic search for practical inversion controls.

Keeping the tuner separate makes the optimization strategy easier to test, modify or replace without complicating the core scientific software.

## Reference

Aminian, M. A., Crawford, W., Stutzmann, É., Montagner, J.-P., Cannat, M., & Hadziioannou, C. (2025). *Shallow crustal structures of the Indian ocean derived from compliance function analysis*. Geophysical Journal International, 242(3), ggaf253. https://doi.org/10.1093/gji/ggaf253

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

**Author:** Mohammad Amin Aminian

## Related research software

This repository is part of a broader seismic/geophysical software portfolio:

- [ComPy](https://github.com/MohammadAmin-Aminian/ComPy) — seafloor compliance processing, DPG calibration and layered elastic inversion.
- [OBS Transient Cleaner](https://github.com/MohammadAmin-Aminian/Transients) — periodic OBS instrument-transient removal.
- [ComPy Inversion Tuner](https://github.com/MohammadAmin-Aminian/Optimization) — reproducible tuning of compliance-inversion controls.
- [RHUM-RUM Geospatial Mapper](https://github.com/MohammadAmin-Aminian/Map) — bathymetry, OBS-network and tectonic-context mapping.
- [VRE Seismic Enhancement](https://github.com/MohammadAmin-Aminian/vre-seismic-enhancement) — Virtual Resolution Enhancement for seismic sections.
- [Gabor Seismic Filter](https://github.com/MohammadAmin-Aminian/gabor-seismic-filter) — orientation-selective 2-D seismic filtering in MATLAB.


## License

This software is released under the **GNU General Public License v3.0 only (GPL-3.0-only)**. See [LICENSE](LICENSE) for the full terms.

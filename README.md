# Compliance inversion optimization — version 2

Tune velocity steps, thickness steps, and the roughness regularization weight
for [ComPy v2](https://github.com/MohammadAmin-Aminian/ComPy) using Optuna.

## Install and run

Python 3.10 or newer:

```bash
git clone https://github.com/MohammadAmin-Aminian/Optimization.git
cd Optimization
python -m pip install -r requirements.txt
python -m pip install "git+https://github.com/MohammadAmin-Aminian/ComPy.git"
python Optimizing_Hyperparameters.py observations.npz --trials 20 --station RR38
```

Create `observations.npz` from your measured arrays:

```python
import numpy as np

np.savez(
    "observations.npz",
    data=compliance,
    frequency=frequency_hz,
    uncertainty=standard_deviation,
)
```

Each array must be finite, nonempty, one-dimensional and equal in length.
Frequencies (Hz) and measurement uncertainties must be positive. Compliance and
uncertainty must use the same units as ComPy's forward prediction (s²/m).
Depth is in metres; velocity proposal steps are m/s and thickness steps are metres.
Use `--help` for depth, layer count, iterations, burn-in, seed and output options.
The station controls ComPy's initial model; choose one supported by ComPy.

## Behavior and limitations

Importing the module never starts an inversion. Version 2 removes undefined global
inputs, passes required uncertainties, excludes burn-in from scoring, rejects
zero thickness steps, and avoids allocating unused depth profiles. The output NPZ
contains `best_misfit`, `Vs_Step`, `H_Step`, and `Alpha`; existing outputs are refused.
A seeded Optuna search and sampler make runs reproducible within the same software
and hardware environment. No automatic 200,000-step inversion or plot is launched.

The score is the minimum post-burn-in misfit, retained from the original approach.
It measures fit rather than chain convergence, posterior accuracy or out-of-sample
prediction. Validate selected proposals with longer, independent chains before
scientific interpretation. Real survey results are not bundled or verified here.

## Development

```bash
python -m pytest -q
```

Tests cover input validation, sampler arguments and burn-in exclusion. An integration
test runs the real Optuna/ComPy CLI twice on synthetic compliance and checks identical
seeded results. CI installs ComPy at a fixed commit; install ComPy locally to run
this test (otherwise it is reported as skipped). The short chains test the interface
and reproducibility, not convergence.
Author: Mohammad Amin Aminian. No license was present in the original repository;
no additional reuse rights are asserted here.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and bug reports.

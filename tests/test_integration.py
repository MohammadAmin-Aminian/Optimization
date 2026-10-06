"""Real Optuna/ComPy workflow, independent of the mocked unit-test inverter."""

import numpy as np
import pytest
from Optimizing_Hyperparameters import main


def test_seeded_cli_is_reproducible(tmp_path):
    inv = pytest.importorskip("inv_compy")
    frequency = np.array([0.005, 0.007, 0.01, 0.015])
    model = np.array(
        [[100, 2200, 3000, 1500], [1000, 2800, 6000, 3500], [0, 3300, 8000, 4500]]
    )
    data = inv.calc_norm_compliance(4000, frequency, model)
    source = tmp_path / "observations.npz"
    np.savez(
        source,
        data=data,
        frequency=frequency,
        uncertainty=np.full(len(frequency), 1e-12),
    )
    outputs = [tmp_path / "first.npz", tmp_path / "second.npz"]
    for output in outputs:
        main(
            [
                str(source),
                "--output",
                str(output),
                "--trials",
                "2",
                "--iterations",
                "8",
                "--burnin",
                "2",
                "--layers",
                "3",
                "--depth",
                "4000",
                "--seed",
                "7",
            ]
        )
    with (
        np.load(outputs[0], allow_pickle=False) as first,
        np.load(outputs[1], allow_pickle=False) as second,
    ):
        assert set(first.files) == {"best_misfit", "Vs_Step", "H_Step", "Alpha"}
        for key in first.files:
            assert np.isfinite(first[key]).all()
            np.testing.assert_array_equal(first[key], second[key])

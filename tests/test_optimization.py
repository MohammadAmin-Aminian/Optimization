import numpy as np
import pytest
from Optimizing_Hyperparameters import make_objective, validate_inputs


class Trial:
    def suggest_int(self, name, lo, hi):
        assert lo > 0
        return lo

    def suggest_float(self, name, lo, hi):
        return 0.25


def test_objective_excludes_burnin_and_supplies_uncertainty():
    def inverter(data, frequency, **kwargs):
        assert kwargs["s"].tolist() == [1, 1]
        assert kwargs["return_profiles"] is False
        assert kwargs["seed"] == 42
        return None, None, np.array([[0, 7, 4]]), None, None, None

    objective = make_objective(
        [1, 2], [0.1, 0.2], [1, 1], iterations=3, burnin=1, inverter=inverter
    )
    assert objective(Trial()) == 4


@pytest.mark.parametrize(
    "data,freq,s",
    [
        ([], [], []),
        ([1], [0], [1]),
        ([1], [1], [0]),
        ([1, 2], [1], [1]),
        ([np.nan], [1], [1]),
    ],
)
def test_invalid_inputs(data, freq, s):
    with pytest.raises(ValueError):
        validate_inputs(data, freq, s)

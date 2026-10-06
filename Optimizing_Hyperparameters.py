"""Optuna hyperparameter search for the ComPy v2 compliance sampler."""

import argparse
from pathlib import Path

import numpy as np


def validate_inputs(data, frequency, uncertainty):
    arrays = [np.asarray(x, dtype=float) for x in (data, frequency, uncertainty)]
    if any(x.ndim != 1 for x in arrays) or not len(arrays[0]):
        raise ValueError("Inputs must be nonempty one-dimensional arrays")
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise ValueError("Data, frequency and uncertainty must have equal lengths")
    if any(not np.isfinite(x).all() for x in arrays):
        raise ValueError("Inputs must be finite")
    if np.any(arrays[1] <= 0) or np.any(arrays[2] <= 0):
        raise ValueError("Frequency and uncertainty must be positive")
    return arrays


def make_objective(
    data,
    frequency,
    uncertainty,
    *,
    depth=4560,
    layers=12,
    iterations=10000,
    burnin=500,
    seed=42,
    station="RR38",
    inverter=None,
):
    data, frequency, uncertainty = validate_inputs(data, frequency, uncertainty)
    if (
        not isinstance(iterations, int)
        or not isinstance(burnin, int)
        or not 0 <= burnin < iterations
    ):
        raise ValueError("Require 0 <= burnin < iterations")
    if (
        not np.isfinite(depth)
        or depth <= 0
        or not isinstance(layers, int)
        or layers < 1
    ):
        raise ValueError("Depth and layer count must be positive")
    if inverter is None:
        from inv_compy import invert_compliace

        inverter = invert_compliace

    def objective(trial):
        result = inverter(
            data,
            frequency,
            depth_s=depth,
            s=uncertainty,
            starting_model=None,
            n_layer=layers,
            sigma_v=trial.suggest_int("Vs_Step", 10, 50),
            sigma_h=trial.suggest_int("H_Step", 1, 30),
            alpha=trial.suggest_float("Alpha", 0, 1),
            iteration=iterations,
            sta=station,
            seed=seed,
            return_profiles=False,
        )
        misfit = np.asarray(result[2], dtype=float).reshape(-1)
        if len(misfit) != iterations or not np.isfinite(misfit).all():
            raise ValueError("Sampler returned an invalid misfit chain")
        return float(np.min(misfit[burnin:]))

    return objective


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="NPZ with data, frequency, uncertainty arrays"
    )
    parser.add_argument("--output", type=Path, default=Path("optimization_results.npz"))
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--burnin", type=int, default=500)
    parser.add_argument("--depth", type=float, default=4560)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--station", default="RR38")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.trials < 1:
        parser.error("--trials must be positive")
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    with np.load(args.input, allow_pickle=False) as source:
        objective = make_objective(
            source["data"],
            source["frequency"],
            source["uncertainty"],
            depth=args.depth,
            layers=args.layers,
            iterations=args.iterations,
            burnin=args.burnin,
            seed=args.seed,
            station=args.station,
        )
    import optuna

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed)
    )
    study.optimize(objective, n_trials=args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, best_misfit=study.best_value, **study.best_params)
    print(f"Best misfit: {study.best_value}; parameters: {study.best_params}")


if __name__ == "__main__":
    main()

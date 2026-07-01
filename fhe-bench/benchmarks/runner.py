"""Core benchmarking logic for all CGM model variants."""

import time
import traceback
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.datasets.shanghai import FEATURE_COLS, load_dataset
from src.models.model import CGMModel

_TEST_SIZE = 0.2
_RANDOM_STATE = 42

# FHE inference is slow; cap at this many samples for the FHE execution path.
_FHE_SAMPLE_LIMIT = 100


@dataclass
class BenchmarkResult:
    model_name: str
    family: str
    backend: str
    mae: float = float("nan")
    rmse: float = float("nan")
    r2: float = float("nan")
    mape: float = float("nan")
    compile_time_s: float = 0.0
    inference_time_ms: float = float("nan")
    n_samples: int = 0
    error: str = ""


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _get_test_data(fhe_limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_test, y_test) from the Shanghai dataset."""
    df = load_dataset()
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["cgm"].values.astype(np.float32)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=_TEST_SIZE, random_state=_RANDOM_STATE
    )
    if fhe_limit is not None:
        X_test = X_test[:fhe_limit]
        y_test = y_test[:fhe_limit]
    return X_test, y_test


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": _mape(y_true, y_pred),
    }


def _bench_cleartext(
    name: str,
    model: CGMModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fhe_subset_size: int | None = None,
) -> BenchmarkResult:
    """Run cleartext inference on X_test.
    """

    family, backend = name.rsplit("/", 1)
    preds, times = [], []
    for x in X_test:
        t0 = time.perf_counter()
        pred = model.run_clear(x)
        times.append((time.perf_counter() - t0) * 1000)
        preds.append(pred)

    y_pred = np.array(preds, dtype=np.float32)

    y_test = y_test[:fhe_subset_size] if fhe_subset_size is not None else y_test
    y_pred = y_pred[:fhe_subset_size] if fhe_subset_size is not None else y_pred

    metrics = _compute_metrics(y_test, y_pred)
    n_samples = len(y_test)

    return BenchmarkResult(
        model_name=name,
        family=family,
        backend=backend,
        n_samples=n_samples,
        inference_time_ms=float(np.mean(times[:n_samples])),
        **metrics,
    )


def _bench_fhe(
    name: str, model: CGMModel, X_test: np.ndarray, y_test: np.ndarray
) -> BenchmarkResult:
    """Benchmark FHE execution: encrypt → run_fhe → decrypt."""
    family, backend = name.rsplit("/", 1)
    preds, times = [], []
    for i, x in enumerate(X_test):
        print(f'beginning #{i}')
        enc = model.encrypt_values(x)
        t0 = time.perf_counter()
        enc_result = model.run_fhe(enc)
        times.append((time.perf_counter() - t0) * 1000)
        preds.append(model.decrypt_value(enc_result))
        print(f'finishing #{i}')

    y_pred = np.array(preds, dtype=np.float32)
    metrics = _compute_metrics(y_test, y_pred)
    return BenchmarkResult(
        model_name=name,
        family=family,
        backend=backend,
        n_samples=len(X_test),
        inference_time_ms=float(np.mean(times)),
        **metrics,
    )


@dataclass
class ModelSpec:
    name: str
    factory: callable
    use_fhe: bool = False


def _make_specs() -> list[ModelSpec]:
    from src.models.naive_regressor.cleartext import CleartextNaiveRegressor
    from src.models.nn.cleartext import CleartextMLP
    from src.models.tree.cleartext import CleartextDecisionTree

    specs = [
        ModelSpec("naive_regressor/cleartext", CleartextNaiveRegressor),
        ModelSpec("nn/cleartext", CleartextMLP),
        ModelSpec("tree/cleartext", CleartextDecisionTree),
    ]

    try:
        from src.models.naive_regressor.concrete import ConcreteNaiveRegressor
        from src.models.nn.concrete import ConcreteMLP
        from src.models.tree.concrete import ConcreteDecisionTree
        from src.models.naive_regressor.heir import HeirNaiveRegressor
        # from src.models.nn.heir import HeirMLP

        specs += [
            ModelSpec("naive_regressor/heir", HeirNaiveRegressor, use_fhe=True),
            # ModelSpec("nn/heir", HeirMLP, use_fhe=True),
            ModelSpec("naive_regressor/concrete", ConcreteNaiveRegressor, use_fhe=True),
            ModelSpec("nn/concrete", ConcreteMLP, use_fhe=True),
            ModelSpec("tree/concrete", ConcreteDecisionTree, use_fhe=True)
        ]
    except ImportError as e:
        print(f"  [warn] Import not available: {e}")

    return specs


def run_all(
    fhe_samples: int = _FHE_SAMPLE_LIMIT
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    print("Loading test set...")
    X_test_full, y_test_full = _get_test_data()
    X_test_fhe, y_test_fhe = X_test_full[:fhe_samples], y_test_full[:fhe_samples]
    print(f"  Cleartext: {len(X_test_full)} samples | FHE: {len(X_test_fhe)} samples\n")

    specs = _make_specs()

    for spec in specs:
        print(f"[{spec.name}]")
        result = BenchmarkResult(
            model_name=spec.name,
            family=spec.name.rsplit("/", 1)[0],
            backend=spec.name.rsplit("/", 1)[1],
        )
        try:
            print("  initializing...", end=" ", flush=True)
            t0 = time.perf_counter()
            model = spec.factory()
            compile_time = time.perf_counter() - t0
            print(f"done ({compile_time:.1f}s)")

            if spec.use_fhe:
                result.compile_time_s = compile_time
                print(
                    f"  FHE inference ({len(X_test_fhe)} samples)...",
                    end=" ",
                    flush=True,
                )
                result = _bench_fhe(spec.name, model, X_test_fhe, y_test_fhe)
                result.compile_time_s = compile_time
            else:
                print(
                    f"  cleartext inference ({len(X_test_full)} samples, "
                    f"metrics on first {len(X_test_fhe)} for FHE parity)...",
                    end=" ",
                    flush=True,
                )
                result = _bench_cleartext(
                    spec.name,
                    model,
                    X_test_full,
                    y_test_full,
                    fhe_subset_size=len(X_test_fhe),
                )

            print(
                f"done  MAE={result.mae:.2f}  RMSE={result.rmse:.2f}  "
                f"R²={result.r2:.3f}  {result.inference_time_ms:.1f}ms/sample"
            )

        except Exception as e:
            result.error = str(e)
            print(f"\n  ERROR: {e}")
            traceback.print_exc()

        results.append(result)

    return results

"""Entry point: python -m benchmarks [--fhe-samples N]"""

import argparse

from sklearn.model_selection import train_test_split

from src.datasets.shanghai import FEATURE_COLS, load_dataset
from benchmarks.runner import run_all, _RANDOM_STATE, _TEST_SIZE
from benchmarks.plots import generate_all

import numpy as np


def _get_cleartext_test_data() -> tuple[np.ndarray, np.ndarray]:
    df = load_dataset()
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["cgm"].values.astype(np.float32)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=_TEST_SIZE, random_state=_RANDOM_STATE
    )
    return X_test, y_test


def main():
    parser = argparse.ArgumentParser(description="CGM model benchmark suite")
    parser.add_argument(
        "--fhe-samples",
        type=int,
        default=100,
        help="Number of samples to run through full FHE execution (default: 100)",
    )
    args = parser.parse_args()

    print(f"FHE sample limit: {args.fhe_samples}")

    results = run_all(fhe_samples=args.fhe_samples)

    print(f"\n{'=' * 60}")
    print(f"{'Model':<35} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'Compile':>9} {'Infer':>9}")
    print(f"{'-' * 60}")
    for r in results:
        if r.error:
            print(f"  {r.model_name:<33} ERROR: {r.error[:30]}")
        else:
            compile_str = (
                f"{r.compile_time_s:.1f}s" if r.compile_time_s > 0 else "    —"
            )
            print(
                f"  {r.model_name:<33} {r.mae:>7.2f} {r.rmse:>7.2f} {r.r2:>7.3f} "
                f"{compile_str:>9} {r.inference_time_ms:>7.2f}ms"
            )
    print(f"{'=' * 60}\n")

    X_test, y_test = _get_cleartext_test_data()
    generate_all(results, X_test, y_test)


if __name__ == "__main__":
    main()

"""Plot generation for benchmark results."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.runner import BenchmarkResult

matplotlib.use("Agg")

OUTPUT_DIR = Path(__file__).parent.parent / "plots" / "benchmark"

_BACKEND_COLORS = {
    "cleartext": "#4878d0",
    "concrete": "#ee854a",
    "heir": "#6acc65",
}

_FAMILY_LABELS = {
    "naive_regressor": "Naive Regressor",
    "nn": "Neural Network (MLP)",
    "tree": "Decision Tree",
}


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def _to_df(results: list[BenchmarkResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results if not r.error])


def _grouped_bar(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    log_scale: bool = False,
) -> None:
    """Grouped bar chart with one group per family, bars per backend."""
    families = list(_FAMILY_LABELS.keys())
    backends = sorted(
        df["backend"].unique(),
        key=lambda b: (
            ["cleartext", "concrete", "heir"].index(b)
            if b in ["cleartext", "concrete", "heir"]
            else 99
        ),
    )

    x = np.arange(len(families))
    width = 0.8 / max(len(backends), 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, backend in enumerate(backends):
        sub = df[df["backend"] == backend]
        vals = []
        for fam in families:
            row = sub[sub["family"] == fam]
            vals.append(row[metric].iloc[0] if not row.empty else np.nan)
        offset = (i - len(backends) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width * 0.9,
            color=_BACKEND_COLORS.get(backend, "#888888"),
            label=backend.capitalize(),
        )
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                fmt = f"{val:.3f}" if abs(val) < 10 else f"{val:.1f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    fmt,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([_FAMILY_LABELS.get(f, f) for f in families], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Backend", fontsize=8)
    if log_scale:
        ax.set_yscale("log")
    fig.tight_layout()
    _save(fig, filename)


def plot_accuracy_metrics(results: list[BenchmarkResult]) -> None:
    df = _to_df(results)
    _grouped_bar(
        df,
        "rmse",
        "RMSE (mg/dL)",
        "Root Mean Squared Error by Model",
        "accuracy_rmse.svg",
    )
    _grouped_bar(
        df, "mae", "MAE (mg/dL)", "Mean Absolute Error by Model", "accuracy_mae.svg"
    )
    _grouped_bar(df, "r2", "R²", "R² Score by Model", "accuracy_r2.svg")
    _grouped_bar(
        df,
        "mape",
        "MAPE (%)",
        "Mean Absolute Percentage Error by Model",
        "accuracy_mape.svg",
    )


def plot_inference_time(results: list[BenchmarkResult]) -> None:
    df = _to_df(results)
    names = df["model_name"].tolist()
    times = df["inference_time_ms"].tolist()
    colors = [_BACKEND_COLORS.get(b, "#888888") for b in df["backend"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(range(len(names)), times, color=colors)
    for bar, val in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.05,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=45,
        )

    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Inference time (ms/sample, log scale)")
    ax.set_title("Inference Time per Model")

    from matplotlib.patches import Patch

    legend = [
        Patch(color=c, label=b.capitalize())
        for b, c in _BACKEND_COLORS.items()
        if b in df["backend"].values
    ]
    ax.legend(handles=legend, fontsize=8)
    fig.tight_layout()
    _save(fig, "inference_time.svg")


def plot_compile_time(results: list[BenchmarkResult]) -> None:
    df = _to_df(results)
    fhe_df = df[df["compile_time_s"] > 0].copy()
    if fhe_df.empty:
        return

    names = fhe_df["model_name"].tolist()
    times = fhe_df["compile_time_s"].tolist()
    colors = [_BACKEND_COLORS.get(b, "#888888") for b in fhe_df["backend"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(names)), times, color=colors)
    for bar, val in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:.1f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Compilation + key-generation time (s)")
    ax.set_title("FHE Compilation Time")

    from matplotlib.patches import Patch

    legend = [
        Patch(color=c, label=b.capitalize())
        for b, c in _BACKEND_COLORS.items()
        if b in fhe_df["backend"].values
    ]
    ax.legend(handles=legend, fontsize=8)
    fig.tight_layout()
    _save(fig, "compile_time.svg")


def plot_cleartext_vs_fhe(results: list[BenchmarkResult]) -> None:
    """Paired RMSE comparison: cleartext vs FHE (concrete / heir) per family."""
    df = _to_df(results)
    families = [f for f in _FAMILY_LABELS if f in df["family"].values]
    backends = [
        b for b in ["cleartext", "concrete", "heir"] if b in df["backend"].values
    ]

    x = np.arange(len(families))
    width = 0.8 / max(len(backends), 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, backend in enumerate(backends):
        sub = df[df["backend"] == backend]
        vals = [
            sub.loc[sub["family"] == f, "rmse"].iloc[0]
            if not sub[sub["family"] == f].empty
            else np.nan
            for f in families
        ]
        offset = (i - len(backends) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width * 0.9,
            color=_BACKEND_COLORS.get(backend, "#888888"),
            label=backend.capitalize(),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_FAMILY_LABELS.get(f, f) for f in families], fontsize=9)
    ax.set_ylabel("RMSE (mg/dL)")
    ax.set_title("Cleartext vs FHE Accuracy (RMSE)")
    ax.legend(title="Backend", fontsize=8)
    fig.tight_layout()
    _save(fig, "cleartext_vs_fhe_rmse.svg")


def plot_predicted_vs_actual(
    results: list[BenchmarkResult], X_test: "np.ndarray", y_test: "np.ndarray"
) -> None:
    """Scatter of predicted vs actual CGM for cleartext models."""

    df = _to_df(results)
    cleartext = df[df["backend"] == "cleartext"]
    if cleartext.empty:
        return

    n = len(cleartext)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for ax, (_, row) in zip(axes[0], cleartext.iterrows()):
        name = row["model_name"]
        family = row["family"]

        # Re-run predictions to get per-sample values
        try:
            model = _load_model(name)
            preds = [model.run_clear(x) for x in X_test]
        except Exception:
            continue

        ax.scatter(y_test, preds, alpha=0.3, s=8, color=_BACKEND_COLORS["cleartext"])
        lims = [min(y_test.min(), min(preds)), max(y_test.max(), max(preds))]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("Actual CGM (mg/dL)")
        ax.set_ylabel("Predicted CGM (mg/dL)")
        ax.set_title(_FAMILY_LABELS.get(family, family))
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("Predicted vs Actual CGM (Cleartext Models)")
    fig.tight_layout()
    _save(fig, "predicted_vs_actual.svg")


def _load_model(name: str):
    if name == "naive_regressor/cleartext":
        from src.models.naive_regressor.cleartext import CleartextNaiveRegressor

        return CleartextNaiveRegressor()
    if name == "nn/cleartext":
        from src.models.nn.cleartext import CleartextMLP

        return CleartextMLP()
    if name == "tree/cleartext":
        from src.models.tree.cleartext import CleartextDecisionTree

        return CleartextDecisionTree()
    raise ValueError(f"Unknown model: {name}")


def save_csv(results: list[BenchmarkResult]) -> None:
    import csv

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "results.csv"
    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"  saved {path}")


def generate_all(
    results: list[BenchmarkResult], X_test: "np.ndarray", y_test: "np.ndarray"
) -> None:
    print("Generating plots...")
    save_csv(results)
    plot_accuracy_metrics(results)
    plot_inference_time(results)
    plot_compile_time(results)
    plot_cleartext_vs_fhe(results)
    plot_predicted_vs_actual(results, X_test, y_test)
    print(f"\nAll plots saved to {OUTPUT_DIR}/")

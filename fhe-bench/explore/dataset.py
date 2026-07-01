"""Dataset exploration and visualization for the Shanghai T2DM CGM dataset."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from src.datasets.shanghai import FEATURE_COLS, load_dataset, load_raw_dataset

matplotlib.use("Agg")

OUTPUT_DIR = Path(__file__).parent.parent / "plots" / "dataset"

# Glucose clinical thresholds (mg/dL)
_HYPO_THRESHOLD = 70
_HYPER_THRESHOLD = 180


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_cgm_distribution(df: pd.DataFrame) -> None:
    """Histogram + KDE of all CGM readings."""
    cgm = df["cgm"].values
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(cgm, bins=80, density=True, alpha=0.5, color="#4878d0", label="Histogram")

    kde = gaussian_kde(cgm, bw_method=0.15)
    x = np.linspace(cgm.min(), cgm.max(), 400)
    ax.plot(x, kde(x), lw=2, color="#4878d0", label="KDE")

    ax.axvline(
        _HYPO_THRESHOLD,
        color="#d62728",
        ls="--",
        lw=1.2,
        label=f"Hypo ({_HYPO_THRESHOLD})",
    )
    ax.axvline(
        _HYPER_THRESHOLD,
        color="#ff7f0e",
        ls="--",
        lw=1.2,
        label=f"Hyper ({_HYPER_THRESHOLD})",
    )

    ax.set_xlabel("CGM (mg/dL)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.set_title("CGM Value Distribution")
    _save(fig, "cgm_distribution.svg")


def plot_samples_per_patient(df: pd.DataFrame) -> None:
    """Horizontal bar chart of record count per patient."""
    counts = df.groupby("patient").size().sort_values()
    patients = counts.index.astype(str)

    fig, ax = plt.subplots(figsize=(6, max(4, len(patients) * 0.15)))
    ax.barh(patients, counts.values, color="#4878d0", height=0.7)
    ax.set_xlabel("Number of 5-min CGM readings")
    ax.set_ylabel("Patient ID")
    ax.set_title("Samples per Patient")
    ax.tick_params(axis="y", labelsize=6)
    _save(fig, "samples_per_patient.svg")


def plot_cgm_by_hour(df: pd.DataFrame) -> None:
    """Box plot of CGM grouped by hour of day (circadian pattern)."""
    raw = load_raw_dataset()
    # Reconstruct hourly data from raw (preserves the datetime index)
    frames = []
    for patient_id, patient_df in raw.items():
        patient_df = patient_df.copy()
        patient_df.columns = ["date", "cgm", *patient_df.columns[2:]]
        patient_df = patient_df[["date", "cgm"]].dropna()
        patient_df["hour"] = pd.to_datetime(patient_df["date"]).dt.hour
        frames.append(patient_df[["hour", "cgm"]])

    hourly = pd.concat(frames, ignore_index=True)
    groups = [hourly.loc[hourly["hour"] == h, "cgm"].values for h in range(24)]

    fig, ax = plt.subplots(figsize=(10, 4))
    bp = ax.boxplot(
        groups,
        positions=range(24),
        widths=0.6,
        patch_artist=True,
        medianprops=dict(color="black", lw=1.5),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#4878d0")
        patch.set_alpha(0.6)

    ax.axhline(
        _HYPO_THRESHOLD,
        color="#d62728",
        ls="--",
        lw=1,
        label=f"Hypo ({_HYPO_THRESHOLD})",
    )
    ax.axhline(
        _HYPER_THRESHOLD,
        color="#ff7f0e",
        ls="--",
        lw=1,
        label=f"Hyper ({_HYPER_THRESHOLD})",
    )
    ax.set_xticks(range(24))
    ax.set_xticklabels(
        [f"{h:02d}:00" for h in range(24)], rotation=45, ha="right", fontsize=7
    )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("CGM (mg/dL)")
    ax.legend(fontsize=8)
    ax.set_title("CGM by Hour of Day (Circadian Pattern)")
    _save(fig, "cgm_by_hour.svg")


def plot_cgm_range_per_patient(df: pd.DataFrame) -> None:
    """Violin plot of each patient's CGM range."""
    patients = sorted(df["patient"].unique())
    data = [df.loc[df["patient"] == p, "cgm"].values for p in patients]

    fig, ax = plt.subplots(figsize=(max(8, len(patients) * 0.2), 4))
    parts = ax.violinplot(data, positions=range(len(patients)), showmedians=True)
    for body in parts["bodies"]:
        body.set_facecolor("#4878d0")
        body.set_alpha(0.5)

    ax.axhline(
        _HYPO_THRESHOLD,
        color="#d62728",
        ls="--",
        lw=1,
        label=f"Hypo ({_HYPO_THRESHOLD})",
    )
    ax.axhline(
        _HYPER_THRESHOLD,
        color="#ff7f0e",
        ls="--",
        lw=1,
        label=f"Hyper ({_HYPER_THRESHOLD})",
    )
    ax.set_xticks(range(len(patients)))
    ax.set_xticklabels([str(p) for p in patients], rotation=90, fontsize=5)
    ax.set_xlabel("Patient ID")
    ax.set_ylabel("CGM (mg/dL)")
    ax.legend(fontsize=8)
    ax.set_title("CGM Range per Patient")
    _save(fig, "cgm_range_per_patient.svg")


def plot_feature_correlation(df: pd.DataFrame) -> None:
    """Heatmap of correlations among lagged features and target."""
    cols = ["cgm"] + FEATURE_COLS
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    labels = ["cgm (target)"] + [
        f"cgm t-{5 * i}" for i in range(1, len(FEATURE_COLS) + 1)
    ]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(
                j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7
            )
    ax.set_title("Feature Correlation Matrix")
    _save(fig, "feature_correlation.svg")


def plot_timeseries_sample(raw: dict) -> None:
    """CGM time series for a few representative patients."""
    # Pick patients with short, medium, and long session lengths
    lengths = {pid: len(pdf) for pid, pdf in raw.items() if not np.isnan(pid)}
    sorted_ids = sorted(lengths, key=lengths.get)
    n = len(sorted_ids)
    selected = [sorted_ids[0], sorted_ids[n // 4], sorted_ids[n // 2], sorted_ids[-1]]

    fig, axes = plt.subplots(
        len(selected), 1, figsize=(12, 3 * len(selected)), sharex=False
    )
    for ax, pid in zip(axes, selected):
        patient_df = raw[pid].copy()
        patient_df.columns = ["date", "cgm", *patient_df.columns[2:]]
        patient_df = patient_df[["date", "cgm"]].dropna()
        patient_df["date"] = pd.to_datetime(patient_df["date"])
        patient_df = patient_df.sort_values("date")
        ax.plot(patient_df["date"], patient_df["cgm"], lw=0.8, color="#4878d0")
        ax.axhline(_HYPO_THRESHOLD, color="#d62728", ls="--", lw=0.8)
        ax.axhline(_HYPER_THRESHOLD, color="#ff7f0e", ls="--", lw=0.8)
        ax.set_ylabel("CGM (mg/dL)", fontsize=8)
        ax.set_title(f"Patient {pid} ({len(patient_df)} readings)", fontsize=9)
        ax.tick_params(axis="x", labelsize=7, rotation=30)

    fig.suptitle("CGM Time Series — Representative Patients", fontsize=11)
    fig.tight_layout()
    _save(fig, "cgm_timeseries.svg")


def plot_session_length(raw: dict) -> None:
    """Bar chart: number of 5-min readings per patient file."""
    data = {pid: len(pdf) for pid, pdf in raw.items() if not np.isnan(pid)}
    sorted_items = sorted(data.items(), key=lambda x: x[1])
    pids = [str(pid) for pid, _ in sorted_items]
    counts = [cnt for _, cnt in sorted_items]

    fig, ax = plt.subplots(figsize=(6, max(4, len(pids) * 0.15)))
    ax.barh(pids, counts, color="#4878d0", height=0.7)
    ax.set_xlabel("Number of 5-min CGM readings")
    ax.set_ylabel("Patient file")
    ax.set_title("Session Length per Patient File")
    ax.tick_params(axis="y", labelsize=5)
    _save(fig, "session_length.svg")


def print_summary(df: pd.DataFrame, raw: dict) -> None:
    """Print key dataset statistics."""
    cgm = df["cgm"]
    n_patients = df["patient"].nunique()
    counts = df.groupby("patient").size()

    print("\n=== Dataset Summary ===")
    print(f"  Patients          : {n_patients}")
    print(f"  Raw files         : {len(raw)}")
    print(f"  Total samples     : {len(df):,}")
    print(
        f"  Samples/patient   : {counts.mean():.1f} ± {counts.std():.1f}  "
        f"(min {counts.min()}, max {counts.max()})"
    )
    print(f"  CGM mean ± std    : {cgm.mean():.1f} ± {cgm.std():.1f} mg/dL")
    print(f"  CGM range         : [{cgm.min():.1f}, {cgm.max():.1f}] mg/dL")
    pct_hypo = (cgm < _HYPO_THRESHOLD).mean() * 100
    pct_hyper = (cgm > _HYPER_THRESHOLD).mean() * 100
    print(f"  Hypoglycemia (<{_HYPO_THRESHOLD}) : {pct_hypo:.1f}%")
    print(f"  Hyperglycemia (>{_HYPER_THRESHOLD}): {pct_hyper:.1f}%")
    print(f"  In range (TIR)    : {100 - pct_hypo - pct_hyper:.1f}%")
    print()


def run() -> None:
    print("Loading dataset...")
    df = load_dataset()
    raw = load_raw_dataset()

    print_summary(df, raw)

    print("Generating plots...")
    plot_cgm_distribution(df)
    plot_samples_per_patient(df)
    plot_cgm_by_hour(df)
    plot_cgm_range_per_patient(df)
    plot_feature_correlation(df)
    plot_timeseries_sample(raw)
    plot_session_length(raw)
    print(f"\nAll plots saved to {OUTPUT_DIR}/")

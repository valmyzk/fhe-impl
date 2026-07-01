from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

DATASET_PATH = Path("dataset/shanghai_t2dm")

WINDOW_SIZE = 4
FEATURE_COLS = [f"cgm_{5 * n}" for n in range(1, WINDOW_SIZE + 1)]


def load_raw_dataset() -> dict[float, pd.DataFrame]:
    return {
        get_patient_id(file.name): pd.read_excel(file)
        for file in DATASET_PATH.iterdir()
        if file.suffix in [".xls", ".xlsx"]
    }


def load_dataset() -> pd.DataFrame:
    raw_dataset = load_raw_dataset()

    df = pd.concat([df.assign(patient=patient) for patient, df in raw_dataset.items()])
    df.rename(columns={df.columns[0]: "date", df.columns[1]: "cgm"}, inplace=True)
    df = df[["date", "patient", "cgm"]]
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    for n, feature in enumerate(FEATURE_COLS):
        df[feature] = df.groupby("patient")["cgm"].shift(n + 1)
    df.dropna(inplace=True)
    return df


def plot_cgm_by_patient(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 4))
    plt.scatter(df["patient"], df["cgm"], alpha=0.3, s=5)
    plt.xlabel("Patient")
    plt.ylabel("CGM (mg/dL)")
    plt.tight_layout()
    plt.show()


def get_patient_id(filename: str) -> float:
    id = filename.split("_", maxsplit=1)[0]
    try:
        return float(id) - 2000 + 1
    except Exception:
        return float("nan")


if __name__ == "__main__":
    print(load_dataset())

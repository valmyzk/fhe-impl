from pathlib import Path
from typing import Sequence, Any

import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from src.models.tree.utils import train
from src.models.model import CGMModel

_CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
_CHECKPOINT = _CHECKPOINT_DIR / "decision_tree.joblib"


class CleartextDecisionTree(CGMModel):
    def __init__(self):
        if _CHECKPOINT.exists():
            self.model: DecisionTreeRegressor = joblib.load(_CHECKPOINT)
        else:
            self.model = train(DecisionTreeRegressor(max_depth=10))
            _CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, _CHECKPOINT)

    def encrypt_values(
        self, values: np.ndarray[tuple[float], np.dtype[np.float32]]
    ) -> Sequence[Any]:
        raise NotImplementedError("operation not supported")

    def decrypt_value(self, value: Any) -> float:
        raise NotImplementedError("operation not supported")

    def run_clear(
        self, values: np.ndarray[tuple[float], np.dtype[np.float32]]
    ) -> float:
        x = values.astype(np.float32).reshape(1, -1)
        return float(self.model.predict(x).squeeze())

    def run_fhe(self, values: Sequence[Any]) -> Any:
        raise NotImplementedError("operation not supported")


__all__ = ["CleartextDecisionTree"]


def test_execute():
    from sklearn.model_selection import cross_val_score
    from src.datasets.shanghai import load_dataset, FEATURE_COLS

    model = CleartextDecisionTree()

    df = load_dataset()
    X = df[FEATURE_COLS].values
    y = df["cgm"].values
    scores = cross_val_score(
        model.model, X, y, cv=5, scoring="neg_root_mean_squared_error"
    )
    print(f"CV RMSE: {-scores.mean():.2f} ± {scores.std():.2f}")

    samples = np.array([90.0, 78.0, 67.0, 50.0])
    print(f"{model.run_clear(samples)=}")


if __name__ == "__main__":
    test_execute()

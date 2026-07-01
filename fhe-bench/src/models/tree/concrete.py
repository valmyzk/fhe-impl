from pathlib import Path
from typing import Sequence, Any

import torch
import numpy as np
from concrete.fhe import Configuration
from concrete.ml.sklearn.tree import DecisionTreeRegressor
from concrete.ml.common.serialization.dumpers import dump
from concrete.ml.common.serialization.loaders import load

from src.datasets.shanghai import load_dataset, FEATURE_COLS
from src.models.tree.utils import train
from src.models.model import CGMModel

_CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
_CHECKPOINT = _CHECKPOINT_DIR / "concrete_decision_tree.json"
_Q_BITS = 6

# Brevitas 0.10.2's QONNX export handler relies on eagerly evaluating tensor
# values (e.g. `if bit_width == 1`), which only works with torch.onnx's
# legacy TorchScript-tracing backend. torch>=2.x defaults `dynamo=True`,
# which traces through torch.export instead and turns that comparison into
# an unresolvable data-dependent guard. Force the legacy backend here.
_torch_onnx_export = torch.onnx.export


def _legacy_onnx_export(*args, **kwargs):
    kwargs.setdefault("dynamo", False)
    return _torch_onnx_export(*args, **kwargs)


torch.onnx.export = _legacy_onnx_export


class ConcreteDecisionTree(CGMModel):
    configuration = Configuration(
        show_mlir=False,
        show_graph=False,
        show_statistics=False,
        # global_p_error=0.01, p_error=0.01
    )

    def __init__(self):
        if _CHECKPOINT.exists():
            with open(_CHECKPOINT, "r", encoding="utf-8") as f:
                self.model: DecisionTreeRegressor = load(f)
        else:
            self.model = train(DecisionTreeRegressor(max_depth=10, n_bits=_Q_BITS))
            _CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            with open(_CHECKPOINT, "w+", encoding="utf-8") as f:
                dump(self.model, f)
        # FHE circuits can't be serialized, compilation must be done every time.
        X = load_dataset()[FEATURE_COLS]
        self.model.compile(X, configuration=self.configuration, global_p_error=0.01)

    def encrypt_values(
        self, values: np.ndarray[tuple[float], np.dtype[np.float32]]
    ) -> Sequence[Any]:
        x = values.astype(np.float32).reshape(1, -1)
        q_input = self.model.quantize_input(x)
        return self.model.fhe_circuit.encrypt(q_input)

    def decrypt_value(self, value: Any) -> float:
        q_output = self.model.fhe_circuit.decrypt(value)
        return float(self.model.dequantize_output(q_output).squeeze())

    def run_clear(
        self, values: np.ndarray[tuple[float], np.dtype[np.float32]]
    ) -> float:
        x = values.astype(np.float32).reshape(1, -1)
        return float(self.model.predict(x, fhe="simulate").squeeze())

    def run_fhe(self, values: Sequence[Any]) -> Any:
        return self.model.fhe_circuit.run(values)


__all__ = ["ConcreteDecisionTree"]


def test_execute():
    from sklearn.model_selection import cross_val_score
    from src.datasets.shanghai import load_dataset, FEATURE_COLS

    model = ConcreteDecisionTree()

    df = load_dataset()
    X = df[FEATURE_COLS].values
    y = df["cgm"].values
    scores = cross_val_score(
        model.model, X, y, cv=5, scoring="neg_root_mean_squared_error"
    )
    print(f"CV RMSE: {-scores.mean():.2f} ± {scores.std():.2f}")

    samples = np.array([90.0, 78.0, 67.0, 50.0])
    print(f"{model.run_clear(samples)=}")

    values = model.encrypt_values(samples)
    result = model.run_fhe(values)
    decrypted = model.decrypt_value(result)
    print(f"{decrypted=}")


if __name__ == "__main__":
    test_execute()

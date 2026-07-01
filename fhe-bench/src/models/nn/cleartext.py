from pathlib import Path
from typing import Sequence, Any

import numpy as np
import torch

from src.models.model import CGMModel
from src.models.nn.perceptron import GlucoseMLP
from src.datasets.shanghai import WINDOW_SIZE
from src.models.nn.utils import CHECKPOINT_DIR, load_checkpoint, save_checkpoint, train, normalize, unnormalize

_CHECKPOINT = CHECKPOINT_DIR / "glucose_mlp.pt"
_HIDDEN = 32


class CleartextMLP(CGMModel):
    def __init__(self, checkpoint_path: Path = _CHECKPOINT):
        self.model = GlucoseMLP(in_features=WINDOW_SIZE, hidden=_HIDDEN)
        if not load_checkpoint(self.model, checkpoint_path):
            train(self.model)
            save_checkpoint(self.model, checkpoint_path)

    def encrypt_values(
        self, values: np.ndarray[tuple[float], np.dtype[np.float32]]
    ) -> Sequence[Any]:
        raise NotImplementedError("operation not supported")

    def decrypt_value(self, value: Any) -> float:
        raise NotImplementedError("operation not supported")

    def run_clear(
        self, values: np.ndarray[tuple[float], np.dtype[np.float32]]
    ) -> float:
        x = normalize(torch.tensor(values, dtype=torch.float32)).unsqueeze(0)
        with torch.no_grad():
            return unnormalize(self.model(x)).item()

    def run_fhe(self, values: Sequence[Any]) -> Any:
        raise NotImplementedError("operation not supported")


__all__ = ["CleartextMLP"]


def test_execute():
    model = CleartextMLP()

    samples = np.array([90, 78, 67, 50])
    print(f"{model.run_clear(samples)=}")


if __name__ == "__main__":
    test_execute()

from typing import Sequence, Any

import torch
import numpy as np
from concrete.ml.torch.compile import compile_brevitas_qat_model

from src.models.model import CGMModel
from src.models.nn.perceptron import QGlucoseMLP
from src.datasets.shanghai import WINDOW_SIZE, load_dataset
from src.models.nn.utils import (
    load_checkpoint,
    train,
    save_checkpoint,
    CHECKPOINT_DIR,
    make_tensors, normalize, unnormalize
)

_CHECKPOINT = CHECKPOINT_DIR / "qglucose_mlp.pt"
_HIDDEN = 32
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


class ConcreteMLP(CGMModel):
    def __init__(self):
        qmodel = QGlucoseMLP(in_features=WINDOW_SIZE, hidden=_HIDDEN, q_bits=_Q_BITS)
        if not load_checkpoint(qmodel, _CHECKPOINT):
            train(qmodel, epochs=100, patience=None)
            save_checkpoint(qmodel, _CHECKPOINT)

        df = load_dataset()
        X, _y = make_tensors(df)
        self.circuit = compile_brevitas_qat_model(
            qmodel,
            X,
            rounding_threshold_bits={"n_bits": _Q_BITS, "method": "approximate"},
            show_mlir=False,
            # device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.circuit.fhe_circuit.keygen()

    def encrypt_values(
        self, values: np.ndarray[tuple[float], np.dtype[np.float32]]
    ) -> Sequence[Any]:
        x = normalize(values.astype(np.float32)).reshape(1, -1)
        q_input = self.circuit.quantize_input(x)
        return self.circuit.fhe_circuit.encrypt(q_input)

    def decrypt_value(self, value: Any) -> float:
        q_output = self.circuit.fhe_circuit.decrypt(value)
        return float(unnormalize(self.circuit.dequantize_output(q_output)).squeeze())

    def run_clear(
        self, values: np.ndarray[tuple[float], np.dtype[np.float32]]
    ) -> float:
        x = normalize(values.astype(np.float32)).reshape(1, -1)
        return float(unnormalize(self.circuit.forward(x, fhe="simulate")).squeeze())

    def run_fhe(self, values: Sequence[Any]) -> Any:
        return self.circuit.fhe_circuit.run(values)


def test_execute():
    model = ConcreteMLP()

    samples = np.array([90, 78, 67, 50])
    print(f"{model.run_clear(samples)=}")

    values = model.encrypt_values(samples)
    result = model.run_fhe(values)
    decrypted = model.decrypt_value(result)
    print(f"{decrypted=}")


if __name__ == "__main__":
    test_execute()

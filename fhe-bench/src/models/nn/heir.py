import ctypes
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch_mlir
from torch_mlir.fx import OutputType

from src.models.model import CGMModel
from src.datasets.shanghai import WINDOW_SIZE
from src.models.nn.perceptron import GlucoseMLP
from src.models.nn.utils import train, CHECKPOINT_DIR, load_checkpoint, save_checkpoint, normalize, unnormalize

_CHECKPOINT = CHECKPOINT_DIR / "glucose_mlp.pt"
_HIDDEN = 32
_LATTIGO_DIR = Path(__file__).parent / "lattigo_mlp"
_SO_PATH = _LATTIGO_DIR / "lattigo_mlp.so"


def _generate_go_source(model: GlucoseMLP) -> str:
    from heir.heir_cli import heir_cli

    sample = torch.tensor(normalize(np.array([0, 133.0, 266.0, 400.0], dtype=np.float32)))
    mlir = torch_mlir.fx.export_and_import(
        model, torch.tensor([sample]), output_type=OutputType.LINALG_ON_TENSORS, func_name="mlp"
    )
    heir_opt = heir_cli.HeirOptBackend.from_pip()
    heir_translate = heir_cli.HeirTranslateBackend.from_pip()
    mlir_ckks = heir_opt.run_binary(
        ["--secretize", "--torch-linalg-to-ckks", "--scheme-to-lattigo"], str(mlir)
    )
    return heir_translate.run_binary(
        ["--emit-lattigo", "--package-name=mlpmodel"], mlir_ckks
    )


def _build_library(model: GlucoseMLP) -> None:
    go_src = _generate_go_source(model)
    (_LATTIGO_DIR / "mlpmodel" / "mlp.go").write_text(go_src)
    subprocess.run(
        ["go", "build", "-buildmode=c-shared", "-o", str(_SO_PATH), "."],
        cwd=str(_LATTIGO_DIR),
        check=True,
    )

def _load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_SO_PATH))
    lib.MlpConfigure.restype = ctypes.c_longlong
    lib.MlpEncryptInput.argtypes = [
        ctypes.c_longlong, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ]
    lib.MlpEncryptInput.restype = ctypes.c_longlong
    lib.MlpEvaluate.argtypes = [ctypes.c_longlong, ctypes.c_longlong]
    lib.MlpEvaluate.restype = ctypes.c_longlong
    lib.MlpDecryptResult.argtypes = [
        ctypes.c_longlong, ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ]
    lib.MlpDecryptResult.restype = None
    lib.MlpFree.argtypes = [ctypes.c_longlong]
    lib.MlpFree.restype = None
    return lib


class HeirMLP(CGMModel):
    def __init__(self):
        self._model = GlucoseMLP(in_features=WINDOW_SIZE, hidden=_HIDDEN)
        if not load_checkpoint(self._model, _CHECKPOINT):
            train(self._model)
            save_checkpoint(self._model, _CHECKPOINT)
        self._model.eval()

        if not _SO_PATH.exists():
            _build_library(self._model)

        self._lib = _load_lib()
        self._ctx = self._lib.MlpConfigure()

    def encrypt_values(
        self, values: np.ndarray[tuple[float], np.dtype[np.float32]]
    ) -> int:
        arr = normalize(values.astype(np.float32))
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return self._lib.MlpEncryptInput(self._ctx, ptr, ctypes.c_int(len(arr)))

    def decrypt_value(self, value: int) -> float:
        out = (ctypes.c_float * 1)()
        self._lib.MlpDecryptResult(self._ctx, value, out, ctypes.c_int(1))
        self._lib.MlpFree(value)
        return float(unnormalize(torch.tensor(out[0], dtype=torch.float32)).item())

    def run_clear(
        self, values: np.ndarray[tuple[float], np.dtype[np.float32]]
    ) -> float:
        x = torch.tensor(normalize(values.astype(np.float32)))
        with torch.no_grad():
            y_norm = self._model(x).item()
        return float(unnormalize(torch.tensor(y_norm, dtype=torch.float32)).item())

    def run_fhe(self, values: int) -> int:
        result = self._lib.MlpEvaluate(self._ctx, values)
        self._lib.MlpFree(values)
        return result

def test_execute():
    model = HeirMLP()

    samples = np.array([86. , 88.2, 66.6, 54. , 57.6, 61.2])
    enc_values = model.encrypt_values(samples)
    enc_result = model.run_fhe(enc_values)

    print(f"{model.decrypt_value(enc_result)=}")

if __name__ == "__main__":
    test_execute()

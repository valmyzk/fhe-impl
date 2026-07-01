from typing import Sequence, Any


from src.models.model import CGMModel

import numpy as np
from heir import compile


class HeirNaiveRegressor(CGMModel):
    mlir = """
    module {
      func.func @naive(%arg0: f32 {secret.secret}, %arg1: f32 {secret.secret}) -> (f32 {secret.secret}) {
        %0 = arith.subf %arg0, %arg1 {secret.secret} : f32
        %c = arith.constant 3.333333343e-01 : f32
        %1 = arith.mulf %0, %c {secret.secret} : f32
        %2 = arith.addf %arg0, %1 {secret.secret} : f32
        return {secret.secret} %2 : f32
      }
    }
    """

    def __init__(self):
        # HEIR's Python frontend is very incomplete, and lacks many features that would be desirable here:
        # - MLIR generation for non-integer operands isn't implemented: we must write HEIR's MLIR by-hand to do any integer divisions.
        # - The only supported backend is OpenFHE
        self.circuit = compile(self.mlir, scheme="ckks", debug=True)
        self.circuit.setup()

    def encrypt_values(
        self, values: np.ndarray[tuple[float], np.dtype[np.float32]]
    ) -> Sequence[Any]:
        cgm, _cgm_5, _cgm_10, cgm_15, *tail = values.tolist()
        if tail:
            raise ValueError("model operates in 4-sample windows")
        return self.circuit.encrypt_arg_0(cgm), self.circuit.encrypt_arg_1(cgm_15)

    def decrypt_value(self, value: Any) -> float:
        return self.circuit.decrypt_result(value)

    def run_clear(
        self, values: np.ndarray[tuple[float], np.dtype[np.float32]]
    ) -> float:
        # cgm, _cgm_5, _cgm_10, cgm_15, *tail = values.tolist()
        # if tail:
        #     raise ValueError('model operates in 4-sample windows')
        # return self.circuit.original(cgm, cgm_15)

        # HEIR's Python frontend doesn't allow simulating non-Python MLIR functions.
        raise NotImplementedError("operation not supported")

    def run_fhe(self, values: Sequence[Any]) -> Any:
        return self.circuit.eval(*values)


__all__ = ["HeirNaiveRegressor"]


def test_execute():
    model = HeirNaiveRegressor()

    samples = np.array([90, 78, 67, 50])
    # print(f'{model.run_clear(samples)=}')

    values = model.encrypt_values(samples)
    result = model.run_fhe(values)
    decrypted = model.decrypt_value(result)
    print(f"{decrypted=}")


if __name__ == "__main__":
    test_execute()

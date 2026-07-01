from typing import Sequence, Any

import numpy as np
from concrete import fhe
from concrete.fhe import Configuration, circuit

from src.models.model import CGMModel


class ConcreteNaiveRegressor(CGMModel):
    configuration = Configuration(
        show_mlir=False,
        show_graph=False,
        show_statistics=False,
        dataflow_parallelize=False,
        p_error=0.01,
        global_p_error=0.01,
    )

    def __init__(self):
        self.circuit = circuit(
            {"cgm": "encrypted", "cgm_15": "encrypted"}, self.configuration
        )(self.naive_regressor)
        self.circuit.keygen()

    @staticmethod
    def naive_regressor(cgm: fhe.uint9, cgm_15: fhe.uint9) -> fhe.int10:
        # Calculates: cgm + (cgm - cgm_15) // 3
        # To improve performance and increase parallelization, (a - b) // 3 has been split into a//3 - b // 3
        # which as an error bound <= 1.

        # Operation broadcasting hints the compiler to re-use the same PBS for both inputs.
        x = np.array([cgm, cgm_15]) // 3

        # Unfortunately, there doesn't seem to be a way to upcast a fhe.int9 to fhe.int10 without PBS.
        # Running the compiler with automatic bit_width guessing causes a major slowdown.
        return fhe.univariate(lambda x: x, outputs=fhe.int10)(cgm) + x[0] - x[1]

    def encrypt_values(
        self, values: np.ndarray[tuple[float], np.dtype[np.float32]]
    ) -> Sequence[Any]:
        cgm, _cgm_5, _cgm_10, cgm_15, *tail = values.tolist()
        if tail:
            raise ValueError("model operates in 4-sample windows")
        # Circuit expects integer inputs (fhe.int9)
        return self.circuit.encrypt(int(round(cgm)), int(round(cgm_15)))

    def decrypt_value(self, value: Any) -> float:
        return self.circuit.decrypt(value)

    def run_clear(
        self, values: np.ndarray[tuple[float], np.dtype[np.float32]]
    ) -> float:
        cgm, _cgm_5, _cgm_10, cgm_15, *tail = values.tolist()
        if tail:
            raise ValueError("model operates in 4-sample windows")
        return self.circuit.simulate(int(round(cgm)), int(round(cgm_15)))

    def run_fhe(self, values: Sequence[Any]) -> Any:
        return self.circuit.run(values)


__all__ = ["ConcreteNaiveRegressor"]


def test_execute():
    model = ConcreteNaiveRegressor()

    samples = np.array([90, 78, 67, 50])
    print(f"{model.run_clear(samples)=}")

    values = model.encrypt_values(samples)
    result = model.run_fhe(values)
    decrypted = model.decrypt_value(result)
    print(f"{decrypted=}")


if __name__ == "__main__":
    test_execute()

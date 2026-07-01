from typing import Sequence, Any

import numpy as np

from src.models.model import CGMModel


class CleartextNaiveRegressor(CGMModel):
    """
    Model that performs a basic linear regression by computing the average slope of the past measurements.

    This is inspired by GluPredKit's naive_regressor model:
    https://github.com/replicahealth/GluPredKit/blob/e5bd6357a6d86fb97ab8a13d02dc73823241e860/glupredkit/models/naive_linear_regressor.py
    """

    def encrypt_values(
        self, values: np.ndarray[tuple[float], np.dtype[np.float32]]
    ) -> Sequence[Any]:
        cgm, _cgm_5, _cgm_10, cgm_15, *tail = values.tolist()
        if tail:
            raise ValueError("model operates in 4-sample windows")
        return self.circuit.encrypt(cgm, cgm_15)

    def decrypt_value(self, value: Any) -> float:
        raise NotImplementedError("operation not supported")

    def run_clear(self, values: np.ndarray[tuple[float], np.dtype[np.float32]]) -> float:
        if len(values) < 2:
            raise RuntimeError("not enough data for linear regression")

        # Naive implementation: computes the mean between slopes.
        # return values[0] - np.mean(np.diff(values))
        # Since the mean collapses to a telescoping sum, we can just:
        return values[0] + (values[0] - values[-1]) / (values.size - 1)

    def run_fhe(self, values: Sequence[Any]) -> Any:
        raise NotImplementedError("operation not supported")


__all__ = ["CleartextNaiveRegressor"]

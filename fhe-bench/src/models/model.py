from abc import ABC, abstractmethod
from typing import Any, Sequence
import numpy as np


class CGMModel(ABC):
    """
    A model capable of predicting the next CGM (Continuous Glucose Monitoring) value.
    """

    @abstractmethod
    def encrypt_values(
        self, values: np.ndarray[tuple[float], np.dtype[np.float32]]
    ) -> Sequence[Any]:
        """
        Homomorphically encrypts the provided CGM values.
        :param values: array of glucose levels, sampled at a 5-minute interval.
        :return: the homomorphic representation of those values.
        """
        ...

    @abstractmethod
    def decrypt_value(self, value: Any) -> float:
        """
        Homomorphically decrypts the provided value.
        :param value: encrypted glucose level.
        :return: the glucose level, in plaintext.
        """
        ...

    @abstractmethod
    def run_clear(
        self, values: np.ndarray[tuple[float], np.dtype[np.float32]]
    ) -> float:
        """
        Runs the mode in cleartext, i.e no FHE computing is applied.
        :param values: array of glucose levels, sampled at a 5-minute interval.
        :return: the predicted glucose level in 5 minutes.
        """
        ...

    @abstractmethod
    def run_fhe(self, values: Sequence[Any]) -> Any:
        """
        Runs the model homomorphically.
        :param values: a sequence of encrypted inputs to the model, encoding glucose levels sampled at a 5-minute interval.
        :return: the encrypted predicted glucose level in 5 minutes.
        """
        ...


__all__ = ["CGMModel"]

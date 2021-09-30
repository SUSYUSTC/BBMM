import typing as tp
from collections.abc import Iterable
import numpy as np
from ..kern import Kernel
class Noise(object):
    values: tp.List[float]
    def __init__(self, value: tp.Union[float, tp.Iterable[float]]) -> None:
        if isinstance(value, Iterable):
            self.values = list(value)
        else:
            self.values = [float(value)]

    def get_diag_reg(self, split: tp.List[np.ndarray]) -> np.ndarray:
        assert len(split) == len(self.values)
        N: int = len(np.concatenate(split))
        result = np.zeros((N, ))
        for s, n in zip(split, self.values):
            result[s] = n
        return result

    def get_diag_reg_gradient(self, split: tp.List[np.ndarray]) -> tp.List[np.ndarray]:
        assert len(split) == len(self.values)
        N: int = len(np.concatenate(split))
        results: tp.List[np.ndarray] = []
        for s in split:
            result = np.zeros((N, ))
            result[s] = 1.0
            results.append(result)
        return results

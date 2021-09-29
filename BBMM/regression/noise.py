import typing as tp
import numpy as np
from ..kern import Kernel
class Noise(object):
    values: tp.List[float]
    def __init__(self, value: tp.Union[float, tp.List[float]]) -> None:
        if isinstance(value, float):
            self.values = []
        else:
            self.values = value

    def get_reg(self, split: tp.List[np.ndarray]) -> np.ndarray:
        assert len(split) == len(self.values)
        N: int = np.concatenate(split).sum()
        result = np.zeros((N, N))
        for s, n in zip(split, self.values):
            result[s, s] = n
        return result

    def ge_ref_gradient(self, split: tp.List[np.ndarray]) -> tp.List[np.ndarray]:
        assert len(split) == len(self.values)
        N: int = np.concatenate(split).sum()
        results: tp.List[np.ndarray] = []
        for s in split:
            result = np.zeros((N, N))
            result[s, s] = 1.0
            results.append(result)
        return results

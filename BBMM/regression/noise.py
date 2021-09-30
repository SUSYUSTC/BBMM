import typing as tp
from collections.abc import Iterable
import numpy as np
from ..kern import Kernel
from .. import utils

general_float = tp.Union[float,np.float]
class Noise(object):
    values: tp.List[float]
    def __init__(self, value: tp.Union[general_float, tp.Iterable[general_float]], n: int) -> None:
        self.values = list(map(lambda x: float(x), utils.make_desired_size(value, n)))

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

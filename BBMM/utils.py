import typing as tp
import numpy as np
from collections.abc import Iterable

T = tp.TypeVar('T')
general_float = tp.Union[float, np.float]


def make_desired_size(value: tp.Union[T, tp.List[T]], n: int) -> tp.List[T]:
    result: tp.List[T]
    if isinstance(value, Iterable):
        result = tp.cast(tp.List[T], value)
    else:
        result = [value] * n
    assert len(result) == n
    return result


from . import param
from . import param_transformation
import typing as tp
import numpy as np
try:
    import cupy as cp
    gpu_available = True
except BaseException:
    gpu_available = False


class Kernel(object):
    def __init__(self) -> None:
        self.cache_state: bool = True
        self.cache: tp.Dict[str, tp.Any] = {}
        self.cache_data: tp.Dict[str, tp.Any]
        self.default_cache: tp.Dict[str, tp.Any]
        self.ps: tp.List[param.Param]
        self.set_ps: tp.List[tp.Callable[[float], None]]
        self.dK_dps: tp.List[tp.Callable]
        self.d2K_dpsdX: tp.List[tp.Callable]
        self.d2K_dpsdX2: tp.List[tp.Callable]
        self.d3K_dpsdXdX2: tp.List[tp.Callable]
        self.nout: int
        self.transformations: tp.List[param_transformation.Transformation]

    def check(self):
        assert hasattr(self, 'default_cache')
        assert hasattr(self, 'ps')
        assert hasattr(self, 'set_ps')
        assert hasattr(self, 'dK_dps')
        assert hasattr(self, 'transformations')
        assert hasattr(self, 'nout')

    def set_all_ps(self, params: tp.List[float]) -> None:
        assert len(params) == len(self.ps)
        for i in range(len(self.ps)):
            self.set_ps[i](params[i])

    def K(self, X1, X2=None, cache: tp.Dict[str, tp.Any]={}):
        raise NotImplementedError

    def split_likelihood(self, Nin: int) -> tp.List[np.ndarray]:
        return [np.arange(Nin)]

    def clear_cache(self):
        raise NotImplementedError

    def set_cache_state(self, state: bool):
        raise NotImplementedError

    def to_dict(self) -> tp.Dict[str, tp.Any]:
        raise NotImplementedError

    def from_dict(self, data: tp.Dict[str, tp.Any]) -> 'Kernel':
        raise NotImplementedError

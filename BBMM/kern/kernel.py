from . import param
from . import param_transformation
import typing as tp
import numpy as np
try:
    import cupy as cp
    gpu_available = True
except BaseException:
    gpu_available = False
from .. import utils


class Kernel(object):
    cache_data: tp.Dict[str, tp.Any]
    default_cache: tp.Dict[str, tp.Any]
    ps: tp.List[param.Param]
    set_ps: tp.List[tp.Callable[[utils.general_float], None]]
    dK_dps: tp.List[tp.Callable]
    d2K_dpsdX: tp.List[tp.Callable]
    d2K_dpsdX2: tp.List[tp.Callable]
    d3K_dpsdXdX2: tp.List[tp.Callable]
    nout: int
    transformations: tp.List[param_transformation.Transformation]
    n_likelihood_splits: int
    def __init__(self) -> None:
        self.cache_state: bool = True
        self.cache: tp.Dict[str, tp.Any] = {}
        if not hasattr(self, 'n_likelihood_splits'):
            self.n_likelihood_splits = 1

    def check(self):
        assert hasattr(self, 'default_cache')
        assert hasattr(self, 'ps')
        assert hasattr(self, 'set_ps')
        assert hasattr(self, 'dK_dps')
        assert hasattr(self, 'transformations')
        assert hasattr(self, 'nout')

    def set_all_ps(self, params: tp.List[utils.general_float]) -> None:
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

from . import param
from . import param_transformation
from typing import Any, List, Callable, Union, Dict
import numpy as np
try:
    import cupy as cp
    gpu_available = True
except BaseException:
    gpu_available = False


class Kernel(object):
    def __init__(self) -> None:
        self.cache_state: bool = True
        self.cache: Dict[str, Any] = {}
        self.cache_data: Dict[str, Any]
        self.default_cache: Dict[str, Any]
        self.ps: List[param.Param]
        self.set_ps: List[Callable[[float], None]]
        self.dK_dps: List[Callable]
        self.d2K_dpsdX: List[Callable]
        self.d2K_dpsdX2: List[Callable]
        self.d3K_dpsdXdX2: List[Callable]
        self.nout: int
        self.transformations: List[param_transformation.Transformation]

    def check(self):
        assert hasattr(self, 'default_cache')
        assert hasattr(self, 'ps')
        assert hasattr(self, 'set_ps')
        assert hasattr(self, 'dK_dps')
        assert hasattr(self, 'transformations')
        assert hasattr(self, 'nout')

    def set_all_ps(self, params: List[float]) -> None:
        assert len(params) == len(self.ps)
        for i in range(len(self.ps)):
            self.set_ps[i](params[i])

    def K(self, X1, X2=None, cache: Dict[str, Any]={}):
        raise NotImplementedError

    def likelihood_split(self, Nin: int) -> List[np.ndarray]:
        return [np.arange(Nin)]

    def clear_cache(self):
        raise NotImplementedError

    def set_cache_state(self, state: bool):
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def from_dict(self, data: Dict[str, Any]) -> Kernel:
        raise NotImplementedError


from typing import Any, List, Union, Dict, Tuple
import functools
import numpy as np
try:
    import cupy as cp
    gpu_available = True
except BaseException:
    gpu_available = False
from .kernel import Kernel
from .cache import Cache
from . import get_kern_obj

type_dims = List[Union[slice, np.ndarray]]


def add(x, y):
    return x + y


def concatenate(ls):
    return functools.reduce(add, ls)


class ProductKernel(Kernel):
    def __init__(self, kern_list: List[Kernel], dims: type_dims = None) -> None:
        self.name = 'kern_operation.ProductKernel'
        self.nk = len(kern_list)
        self.kern_list = kern_list
        assert np.all(np.array([k.nout for k in kern_list]) == kern_list[0].nout)
        self.nout = kern_list[0].nout
        self.nps = [len(k.ps) for k in self.kern_list]
        self.cumsum = np.concatenate([np.array([0]), np.cumsum(self.nps)])
        self.cache_K: Dict[str, Any] = {}
        self.cache_dK_dp: Dict[str, Any] = {}
        self.ps = concatenate([k.ps for k in self.kern_list])
        self.set_ps = concatenate([k.set_ps for k in self.kern_list])
        self.dK_dps = []
        self.transformations = concatenate([k.transformations for k in self.kern_list])
        self.default_cache: Dict[str, Any] = {}
        if dims is None:
            self.dims: type_dims = [slice(None, None, None) for i in range(self.nk)]
        else:
            self.dims = dims
            assert len(self.dims) == self.nk
        for i in range(len(self.ps)):
            kern_index, pos_index = self.get_pos(i)

            def func(X, X2=None, i=i, **kwargs):
                return self.dK_dp(i, X, X2, **kwargs)
            self.dK_dps.append(func)
        super().__init__()
        self.check()

    def get_pos(self, i: int) -> Tuple[int, int]:
        kern_index = len(np.where(self.cumsum <= i)[0]) - 1
        pos_index = i - self.cumsum[kern_index]
        return (kern_index, pos_index)

    def cached_K(self, kern_index, X, X2=None):
        X = X[:, self.dims[kern_index]]
        if X2 is not None:
            X2 = X2[:, self.dims[kern_index]]
        if kern_index not in self.cache_K:
            self.cache_K[kern_index] = self.kern_list[kern_index].K(X, X2)
        return self.cache_K[kern_index]

    def cached_dK_dp(self, kern_index, pos_index, X, X2=None):
        X = X[:, self.dims[kern_index]]
        if X2 is not None:
            X2 = X2[:, self.dims[kern_index]]
        if (kern_index, pos_index) not in self.cache_dK_dp:
            self.cache_dK_dp[(kern_index, pos_index)] = self.kern_list[kern_index].dK_dps[pos_index](X, X2)
        return self.cache_dK_dp[(kern_index, pos_index)]

    @Cache('g')
    def K(self, X, X2=None):
        if gpu_available:
            xp = cp.get_array_module(X)
        else:
            xp = np
        Ks = [self.cached_K(i, X, X2) for i in range(self.nk)]
        if not self.cache_state:
            self.clear_cache()
        return functools.reduce(xp.multiply, Ks)

    @Cache('g')
    def dK_dp(self, i, X, X2=None):
        if gpu_available:
            xp = cp.get_array_module(X)
        else:
            xp = np
        kern_index, pos_index = self.get_pos(i)
        Ks = [self.cached_K(j, X, X2) for j in range(self.nk)]
        Ks[kern_index] = self.cached_dK_dp(kern_index, pos_index, X, X2)
        self.Ks = Ks
        if not self.cache_state:
            self.clear_cache()
        return functools.reduce(xp.multiply, Ks)

    def clear_cache(self) -> None:
        self.cache_K = {}
        self.cache_dK_dp = {}
        for i in range(self.nk):
            self.kern_list[i].clear_cache()

    def to_dict(self) -> Dict[str, Any]:
        data = {
            'kern_list': [k.to_dict() for k in self.kern_list],
            'dims': self.dims,
            'name': self.name,
        }
        return data

    @classmethod
    def from_dict(self, data: Dict[str, Any]) -> Kernel:
        kern_list = [get_kern_obj(kerndata) for kerndata in data['kern_list']]
        kernel = self(kern_list, dims=data['dims'])
        return kernel

    def set_cache_state(self, state: bool) -> None:
        self.cache_state = state
        for k in self.kern_list:
            k.set_cache_state(state)

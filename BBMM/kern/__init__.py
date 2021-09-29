import typing as tp
from .kernel import Kernel

def get_kern(name: str) -> Kernel:
    return tp.cast(Kernel, eval(name))


def get_kern_obj(data: tp.Dict[str, tp.Any]) -> Kernel:
    return tp.cast(Kernel, eval(data['name'])).from_dict(data)


from . import stationary
from . import derivative
from . import difference
from .stationary import RBF
from .stationary import Matern32
from .stationary import Matern52
from .derivative import Derivative
from .derivative import FullDerivative
from .difference import Difference
from . import kern_operation
from .kern_operation import ProductKernel
from . import polynomial
from .polynomial import Linear

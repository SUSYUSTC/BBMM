import typing as tp
import numpy as np
class Transformation(object):
    def __init__(self) -> None:
        pass

    def __call__(self, x: float) -> float:
        raise NotImplementedError

    def d(self, x: float) -> float:
        raise NotImplementedError

    def inv(self, x: float) -> float:
        raise NotImplementedError

class Linear(Transformation):
    def __init__(self) -> None:
        pass

    def __call__(self, x: float) -> float:
        return x

    def d(self, x: float) -> float:
        return 1.0

    def inv(self, x: float) -> float:
        return x


class Log(Transformation):
    def __init__(self) -> None:
        pass

    def __call__(self, x: float) -> float:
        return float(np.log(x))

    def d(self, x: float) -> float:
        return 1.0 / x

    def inv(self, x: float) -> float:
        return float(np.exp(x))


linear = Linear()
log = Log()


class Group(object):
    def __init__(self, group: tp.List[Transformation]) -> None:
        self.group = group
        self.n = len(group)

    def __call__(self, x: tp.List[float]) -> tp.List[float]:
        assert len(x) == self.n
        return [self.group[i](x[i]) for i in range(self.n)]

    def d(self, x: tp.List[float]) -> tp.List[float]:
        assert len(x) == self.n
        return [self.group[i].d(x[i]) for i in range(self.n)]

    def inv(self, x: tp.List[float]) -> tp.List[float]:
        assert len(x) == self.n
        return [self.group[i].inv(x[i]) for i in range(self.n)]

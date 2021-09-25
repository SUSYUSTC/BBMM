import numpy as np


class Linear(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return x

    def d(self, x):
        return 1

    def inv(self, x):
        return x


class Log(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return np.log(x)

    def d(self, x):
        return 1 / x

    def inv(self, x):
        return np.exp(x)


linear = Linear()
log = Log()


class Group(object):
    def __init__(self, group):
        self.group = group
        self.n = len(group)

    def __call__(self, x):
        assert len(x) == self.n
        return [self.group[i](x[i]) for i in range(self.n)]

    def d(self, x):
        assert len(x) == self.n
        return [self.group[i].d(x[i]) for i in range(self.n)]

    def inv(self, x):
        assert len(x) == self.n
        return [self.group[i].inv(x[i]) for i in range(self.n)]

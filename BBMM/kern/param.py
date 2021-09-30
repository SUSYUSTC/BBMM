from .. import utils
class Param(object):
    def __init__(self, name: str, value: utils.general_float):
        self.name = name
        self.value = float(value)

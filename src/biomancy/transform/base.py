import random
from abc import ABC
from abc import abstractmethod

from .typing import Probability


class Transform(ABC):
    def __init__(self, *, pb: Probability = 0.5):
        if isinstance(pb, float) and (pb < 0 or pb > 1):
            raise ValueError('Probability must be inside [0, 1] interval.')
        self.pb = pb

    def __call__(self, **sample):
        match self.pb:
            case 'always':
                return self._transform(**sample)
            case 'never':
                return self._no_transform(**sample)
            case pb if random.random() <= pb:
                return self._transform(**sample)
            case _:
                return self._no_transform(**sample)

    def __eq__(self, other):
        return isinstance(other, Transform) and other.pb == self.pb

    __hash__ = None

    @abstractmethod
    def _transform(self, **sample):
        raise NotImplementedError()

    @abstractmethod
    def _no_transform(self, **sample):
        raise NotImplementedError()

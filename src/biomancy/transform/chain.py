from typing import Iterable

from .base import Transform
from .typing import Probability


class Chain(Transform):
    def __init__(self, transforms: Iterable[Transform], *, pb: Probability = 'always', **kwargs):
        super().__init__(pb=pb, **kwargs)
        self.transforms = tuple(transforms)
        if not self.transforms:
            raise ValueError('There must be at least one transform inside the chain.')

    def __eq__(self, other):
        return super().__eq__(other) and \
            isinstance(other, Chain) and \
            other.transforms == self.transforms

    __hash__ = None

    def _transform(self, **sample):
        for tr in self.transforms:
            sample = tr(**sample)
        return sample

    def _no_transform(self, **sample):
        return sample

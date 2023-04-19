from typing import Iterable, Any

from .base import Transform
from .typing import Probability, ToTransform


class Chain(Transform):
    def __init__(self, transforms: Iterable[Transform], *, pb: Probability = 'always'):
        super().__init__(pb=pb)
        self.transforms = tuple(transforms)
        if not self.transforms:
            raise ValueError('There must be at least one transform inside the chain.')

    def __eq__(self, other: object) -> bool:
        return super().__eq__(other) and \
            isinstance(other, Chain) and \
            other.transforms == self.transforms

    def _transform(self, **sample: ToTransform) -> ToTransform:
        for tr in self.transforms:
            sample = tr(**sample)
        return sample

    def _no_transform(self, **sample: ToTransform) -> ToTransform:
        return sample

import random

import math
from pybedtools import Interval

from .base import IntervalTransform
from ..typing import IntervalLimit, Probability


class RandomizeStrand(IntervalTransform):
    def __init__(self, fwdp: float = 0.5, revp: float = 0.5, *, pb: Probability = 'always'):
        super().__init__(pb=pb)
        if fwdp < 0 or revp < 0:
            raise ValueError('fwdp & revp must be >= 0')
        if not math.isclose(fwdp + revp, 1):
            raise ValueError('fwdp + revp must be  == 1')

        self.fwdp = fwdp
        self.revp = revp

    def __eq__(self, other: object) -> bool:
        return super().__eq__(other) and \
            isinstance(other, RandomizeStrand) and \
            other.fwdp == self.fwdp and \
            other.revp == self.revp

    def _transform_interval(self, interval: Interval, _: IntervalLimit) -> Interval:
        if random.random() <= self.fwdp:
            interval.strand = '+'
        else:
            interval.strand = '-'
        return interval

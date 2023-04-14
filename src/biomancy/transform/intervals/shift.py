import random
from typing import Union

from pybedtools import Interval

from .base import IntervalTransform
from ..typing import IntervalLimit


class Shift(IntervalTransform):
    def __init__(self, maxshift: Union[int, float] = 0.5, **kwargs):
        super().__init__(**kwargs)
        if maxshift <= 0:
            raise ValueError('Max interval shift must be > 0')

        self.maxshift = maxshift

    def __eq__(self, other):
        return super().__eq__(other) and \
            isinstance(other, Shift) and \
            other.maxshift == self.maxshift

    __hash__ = None

    def _transform_interval(self, interval: Interval, limits: IntervalLimit) -> Interval:
        left_space = min(limits[0], interval.start) - interval.start
        right_space = max(limits[1], interval.end) - interval.end

        maxshift = self.maxshift if isinstance(self.maxshift, int) else int(interval.length * self.maxshift)
        limits = (max(left_space, -maxshift), min(right_space, maxshift))
        shift = random.randint(limits[0], limits[1])

        interval.start += shift
        interval.end += shift
        return interval

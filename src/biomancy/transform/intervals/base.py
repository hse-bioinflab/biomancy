from abc import abstractmethod

from pybedtools import Interval

from ..base import Transform
from ..typing import IntervalLimit


class IntervalTransform(Transform):
    def __call__(self, **kwargs):
        if 'limits' not in kwargs:
            raise ValueError(
                'Use InjectLimits to specify regions of interest before any IntervalTransform or pass limits manually. '
                'All transformations are guaranteed to retain intervals inside specified limits.',  # noqa: WPS326
            )
        return super().__call__(**kwargs)

    def _transform(self, interval: Interval, limits: IntervalLimit):
        assert limits[0] <= interval.start < interval.end <= limits[1]  # noqa: S101
        interval = self._transform_interval(interval, limits)
        assert limits[0] <= interval.start < interval.end <= limits[1]  # noqa: S101
        return {'interval': interval, 'limits': limits}

    def _no_transform(self, interval: Interval, limits: IntervalLimit):
        return {'interval': interval, 'limits': limits}

    @abstractmethod
    def _transform_interval(self, interval: Interval, limits: IntervalLimit) -> Interval:
        raise NotImplementedError()

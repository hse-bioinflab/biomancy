from abc import abstractmethod

from pybedtools import Interval

from ..base import Transform
from ..typing import IntervalLimit, ToTransform


class IntervalTransform(Transform):
    def __call__(self, **kwargs: ToTransform) -> ToTransform:
        if 'limits' not in kwargs:
            raise ValueError(
                'Use InjectLimits to specify regions of interest before any IntervalTransform or pass limits manually. '
                'All transformations are guaranteed to retain intervals inside specified limits.',  # noqa: WPS326
            )
        return super().__call__(**kwargs)

    def _transform(self, **kwargs: ToTransform) -> ToTransform:
        if 'interval' not in kwargs or 'limits' not in kwargs:
            raise ValueError('TODO')
        interval: Interval = kwargs['interval']
        limits: IntervalLimit = kwargs['limits']

        assert limits[0] <= interval.start < interval.end <= limits[1]  # noqa: S101
        kwargs['interval'] = self._transform_interval(interval, limits)
        assert limits[0] <= interval.start < interval.end <= limits[1]  # noqa: S101
        return kwargs

    def _no_transform(self, **kwargs: ToTransform) -> ToTransform:
        return kwargs

    @abstractmethod
    def _transform_interval(self, interval: Interval, limits: IntervalLimit) -> Interval:
        raise NotImplementedError()

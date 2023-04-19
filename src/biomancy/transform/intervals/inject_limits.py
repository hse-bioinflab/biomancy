import logging
from bisect import bisect_right
from collections import defaultdict

from pybedtools import Interval, BedTool

from ..base import Transform
from ..typing import IntervalLimit, ToTransform


class InjectLimits(Transform):
    limits: dict[str, list[IntervalLimit]]

    def __init__(self, *, rois: list[Interval]):
        super().__init__(pb='always')

        indexed = defaultdict(list)
        for roi in BedTool(rois).sort().merge():
            indexed[roi.chrom].append((roi.start, roi.end))
            if roi.strand != '.':
                logging.warning('InjectLimit is unstranded')

        self.limits = {chrom: sorted(rois) for chrom, rois in indexed.items()}

    def __call__(self, **kwargs: ToTransform) -> ToTransform:
        # Inject limits if needed
        if 'limits' not in kwargs:
            if 'interval' not in kwargs:
                raise ValueError('TODO')
            kwargs['limits'] = self.limits_for(kwargs['interval'])

        return super().__call__(**kwargs)

    def limits_for(self, interval: Interval) -> IntervalLimit:
        index = self.limits.get(interval.chrom, None)
        if index is None:
            raise ValueError(f'There are no limits available for the {interval.chrom} chromosome.')

        ind = bisect_right(index, interval.start, key=lambda it: it[0]) - 1
        # Get the closest correct roi to show a meaningful error
        ind = max(0, min(len(index) - 1, ind))
        roi = index[ind]

        is_inside = roi[0] <= interval.start < interval.end <= roi[1]
        if not is_inside:
            closest = index[ind]
            prv = index[ind - 1] if ind - 1 >= 0 else None
            nxt = index[ind + 1] if ind + 1 < len(index) else None
            raise ValueError(
                'The requested interval is outside the regions of interest:' +
                f'   interval: {interval}' +
                '   rois:' +
                f'      previous: {prv}' +
                f'      closest: {closest}' +
                f'      next: {nxt}',
            )
        return roi

    def _transform(self, **sample: ToTransform) -> ToTransform:
        return sample

    def _no_transform(self, **sample: ToTransform) -> ToTransform:
        return sample

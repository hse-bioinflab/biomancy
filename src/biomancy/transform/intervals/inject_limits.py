import logging
from bisect import bisect_right
from collections import defaultdict
from typing import Optional

from pybedtools import Interval, BedTool

from ..base import Transform
from ..typing import IntervalLimit


class InjectLimits(Transform):
    limits: dict[str, list[IntervalLimit]]

    def __init__(self, *, rois: list[Interval], **kwargs):
        super().__init__(pb='always', **kwargs)

        indexed = defaultdict(list)
        for roi in BedTool(rois).sort().merge():
            indexed[roi.chrom].append((roi.start, roi.end))
            if roi.strand != '.':
                logging.warning('InjectLimit is unstranded')

        self.limits = {chrom: sorted(rois) for chrom, rois in indexed.items()}

    def __call__(self, *, interval: Interval, limits: Optional[tuple[int, int]] = None, **kwargs):
        # Inject limits if needed
        if limits is None:
            limits = self.limits_for(interval)

        return super().__call__(interval=interval, limits=limits, **kwargs)

    def limits_for(self, interval: Interval):
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

    def _transform(self, **sample):
        return sample

    def _no_transform(self, **sample):
        return sample

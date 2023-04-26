from dataclasses import dataclass
from typing import Callable, Any

from pybedtools import Interval

from ..typing import Range


@dataclass(eq=True, slots=True)
class Partitions(object):
    intervals: list[Interval]
    rois: list[Range]

    def __post_init__(self) -> None:
        if len(self.rois) != len(self.intervals):
            raise ValueError('Number of ROIs must be equal to the number of data intervals')

        lengths = {it.length for it in self.intervals}
        if len(lengths) > 1:
            raise ValueError(f'Intervals in batch must have identical lengths, got: {lengths}')

        # TODO: Run checks once at the end
        for roi, it in zip(self.rois, self.intervals):
            rlen = roi[1] - roi[0]
            if rlen > it.length:
                raise ValueError(f'ROI length ({rlen}) must be <= inference interval length ({it.length})')
            if roi[0] < 0 or roi[1] > it.length:
                raise ValueError(
                    f'ROI coordinates ({roi}) must be given relative to the interval where 0 = interval start',
                )

        # TODO: check that ROIs don't overlap
        self.sort(lambda interval, _: (interval.strand, interval.chrom, interval.start))

    def sort(self, key: Callable[[Interval, Range], Any]) -> None:
        indices = range(len(self))
        argsort = sorted(
            indices,
            key=lambda ind: key(self.intervals[ind], self.rois[ind]),  # type: ignore[no-any-return]
        )

        self.intervals = [self.intervals[ind] for ind in argsort]
        self.rois = [self.rois[ind] for ind in argsort]

    def __len__(self) -> int:
        return len(self.intervals)

    def __getitem__(self, item: slice) -> 'Partitions':
        return Partitions(self.intervals[item], self.rois[item])

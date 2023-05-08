from typing import Optional, Any

from numpy import typing as npt
from pybedtools import Interval
from torch.utils.data import Dataset

from .data.sources import DataSource
from biomancy.typing import BedLike
from .transform.intervals import IntervalTransform


class GenomicDataset(Dataset[dict[str, npt.NDArray[Any]]]):
    def __init__(
        self,
        features: dict[str, DataSource],
        intervals: BedLike,
        interval_transform: Optional[IntervalTransform] = None,
    ):
        if not intervals:
            raise ValueError('There must be at least 1 genomic interval.')
        self.features = features
        self.intervals = tuple(intervals)
        self.interval_transform = interval_transform

    def __getitem__(self, idx: int) -> dict[str, npt.NDArray[Any]]:
        it = self.intervals[idx]
        # Copy interval and strip meta information
        it = Interval(it.chrom, it.start, it.end, strand=it.strand)
        if self.interval_transform is not None:
            it = self.interval_transform(interval=it)['interval']

        features = {}
        for key, source in self.features.items():
            features[key] = source.fetch(it.chrom, it.strand, it.start, it.end)
        return features

    def __len__(self) -> int:
        return len(self.intervals)

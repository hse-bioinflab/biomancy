from typing import List, Optional

from pybedtools import Interval

from .data.sources import DataSource
from .transform.intervals import IntervalTransform


class GenomicDataset(object):
    def __init__(
        self,
        features: dict[str, DataSource],
        intervals: List[Interval],
        interval_transform: Optional[IntervalTransform] = None,
    ):
        if not features or not intervals:
            raise ValueError('There must be at least 1 data source for features and at least 1 genomic interval.')
        self.features = features
        self.intervals = intervals
        self.interval_transform = interval_transform

    def __getitem__(self, idx):
        it = self.intervals[idx]
        if self.interval_transform is not None:
            it = self.interval_transform(interval=it)['interval']

        features = {}
        for key, source in self.features.items():
            features[key] = source.fetch(it.chrom, it.strand, it.start, it.end)
        return features

    def __len__(self):
        return len(self.intervals)

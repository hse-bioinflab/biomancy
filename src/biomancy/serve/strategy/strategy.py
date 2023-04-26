from abc import abstractmethod, ABC
from typing import Iterable

from pybedtools import Interval

from biomancy.data.typing import BedLike
from .partitions import Partitions


class PartitionStrategy(ABC):

    def partition(self, intervals: BedLike) -> Partitions:
        intervals = list(intervals)
        if not intervals:
            raise ValueError('There must be at least one inference interval!')

        strands = {it.strand for it in intervals}
        if not strands.issubset({'+', '-'}):
            raise ValueError('All inference intervals must be stranded (i.e. interval.strand == "+" or == "-")')

        partitions = self._partition(intervals)
        return partitions

    @abstractmethod
    def _partition(self, intervals: Iterable[Interval]) -> Partitions:
        raise NotImplementedError()

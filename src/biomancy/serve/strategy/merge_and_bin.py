from typing import Union, Iterable, Generator, Any

from pybedtools import Interval, BedTool

from .partitions import Partitions
from .strategy import PartitionStrategy


def _bins(start: int, end: int, size: int, step: int) -> Generator[tuple[int, int], None, None]:
    ind = 0
    while True:
        bstart = ind * step + start
        bend = bstart + size
        if bend >= end:
            break
        yield bstart, bend
        ind += 1

    # Last item
    bend = min(bend, end)
    yield bstart, bend


class MergeAndBin(PartitionStrategy):
    def __init__(self, binsize: int, roisize: Union[float, int] = 0.9):
        if binsize <= 0:
            raise ValueError(f'Bin size must be >= 0, got {binsize}')

        if roisize < 0:
            raise ValueError(f'ROI size must be >= 0, got {roisize}')

        if isinstance(roisize, float):
            roisize = int(round(binsize * roisize, None))

        if roisize >= binsize:
            raise ValueError(f"ROI size({roisize}) can't be >= binsize({binsize})")

        self.binsize = binsize
        self.roisize = roisize

        left_offset = (self.binsize - self.roisize) // 2
        self.offsets = (left_offset, self.binsize - self.roisize - left_offset)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, MergeAndBin) and \
            self.binsize == other.binsize and \
            self.roisize == other.roisize and \
            self.offsets == other.offsets

    def _make_bins(
        self,
        contig: str,
        strand: str,
        start: int,
        end: int,
        intervals: list[Interval],
        rois: list[tuple[int, int]],
    ) -> None:
        if end - start == self.binsize:
            intervals.append(Interval(contig, start, end, strand=strand))
            rois.append((0, self.binsize))
            return

        # First element is special
        rois.append((0, self.offsets[0] + self.roisize))
        intervals.append(Interval(contig, start, start + self.binsize, strand=strand))

        bingen = _bins(start + rois[-1][1], end, self.roisize, self.roisize)

        for roi in bingen:
            if end - roi[1] <= self.offsets[1]:
                # Last element is special
                it = Interval(contig, end - self.binsize, end, strand=strand)
                assert start <= it.start < it.end <= end and it.length == self.binsize  # noqa: S101

                intervals.append(it)
                rois.append((roi[0] - it.start, self.binsize))
                break

            it = Interval(contig, roi[0] - self.offsets[0], roi[1] + self.offsets[1], strand=strand)
            assert start <= it.start < it.end <= end and it.length == self.binsize  # noqa: S101
            intervals.append(it)
            rois.append((self.offsets[0], self.binsize - self.offsets[1]))

    def _partition(self, intervals: Iterable[Interval]) -> Partitions:
        # Sort-merge intervals
        intervals = BedTool(intervals) \
            .sort() \
            .merge(s=True, c=6, o='distinct') \
            .sort()

        # Sanitize groups & derive bins
        parts: list[Interval] = []
        rois: list[tuple[int, int]] = []
        for it in intervals:
            if it.length < self.binsize:
                raise ValueError(
                    "Can't partition input intervals: " +
                    f'there is an interval ({it}) with length < binsize ({self.binsize})',
                )
            strand = it.fields[3]
            self._make_bins(it.chrom, strand, it.start, it.end, parts, rois)

        return Partitions(parts, rois)

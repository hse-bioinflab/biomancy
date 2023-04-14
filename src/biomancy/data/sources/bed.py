from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from intervaltree import IntervalTree
from pybedtools import BedTool

from .data_source import DataSource, Strand


class BED(DataSource):
    def __init__(
        self,
        bed: Path,
        *,
        strand_specific: bool = True,
        score_col: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not bed.is_file():
            raise ValueError(f"BED file doesn't exist: {bed}")
        self.bed = bed
        self.score_col = score_col
        self.strand_specific = strand_specific

        index = defaultdict(lambda *args: IntervalTree())
        for it in BedTool(bed):
            score = float(it.fields[self.score_col]) if self.score_col else 1
            key = (it.chrom, it.strand) if self.strand_specific else it.chrom
            index[key].addi(it.start, it.end, score)

        self.index = dict(index)

    def __eq__(self, other):
        return isinstance(other, BED) \
            and self.bed == other.bed \
            and self.score_col == other.score_col \
            and self.strand_specific == other.strand_specific \
            and self.index == other.index  # noqa: WPS222

    def _fetch(self, contig: str, strand: Strand, start: int, end: int) -> np.ndarray:
        result = np.zeros(end - start, dtype=self.dtype)

        key = (contig, strand) if self.strand_specific else contig
        if key not in self.index:
            return result

        for hit in self.index[key].overlap(start, end):
            hstart, hend = max(hit.begin, start), min(hit.end, end)
            assert start <= hstart <= hend <= end  # noqa: S101
            result[hstart - start: hend - start] = hit.data
        return result

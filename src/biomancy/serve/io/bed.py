from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TextIO

import numpy as np
from numpy import typing as npt
from pybedtools import Interval, BedTool

from biomancy.typing import PathLike
from .hook import Hook, Adapter


class BED(Hook['BED.Record']):
    @dataclass(frozen=True, slots=True)
    class Record(object):  # noqa: WPS431
        chrom: str
        strand: str
        starts: npt.NDArray[np.int32]
        ends: npt.NDArray[np.int32]
        scores: npt.NDArray[np.uint16]

        def __post_init__(self) -> None:
            lstarts = len(self.starts)
            lends = len(self.ends)
            lscores = len(self.scores)
            if lstarts != lends or lstarts != lscores:
                raise ValueError(
                    f'For a correct BED record start({lstarts}), end({lends}), and value({lscores}) ' +
                    'arrays size must have the same length',
                )

        @staticmethod
        def from_array(  # noqa: WPS210
            it: Interval,
            values: npt.NDArray[np.float32],
            *,
            roi: Optional[tuple[int, int]] = None,
            thr: float = 0.5,
            maxval: float = 1.0,
        ) -> Optional['BED.Record']:
            if values.ndim != 1 or it.length != values.size:
                raise ValueError('Data must be 1D array with length == interval.length')

            # Trim to ROI
            if roi:
                roi_inside_interval = (0 <= roi[0] < roi[1] <= it.length)
                if not roi_inside_interval:
                    raise ValueError('ROI must be in interval coordinates')
                values = values[roi[0]: roi[1]]
                start = it.start + roi[0]
            else:
                start = it.start

            # Threshold values & make dense pos-val arrays
            mask = values >= thr

            # Dense pos-val arrays
            pos = np.flatnonzero(mask) + start
            values = ((values[mask] / maxval) * 1000).round().astype(np.uint16)

            if values.size == 0:
                return None

            # Derive islands: gap-separated intervals with values > thr
            islands = np.flatnonzero((pos[1:] - pos[:-1]) > 1) + 1
            islands = np.concatenate([[0], islands, [pos.size]])

            starts = pos[islands[:-1]]
            ends = starts + (islands[1:] - islands[:-1])

            # Calculate scores for individual islands
            ind = 0
            scores = np.empty(islands.size - 1, dtype=np.uint16)
            for st, en in zip(islands[:-1], islands[1:]):
                scores[ind] = values[st: en].max()
                ind += 1

            return BED.Record(it.chrom, it.strand, starts, ends, scores)

        def __len__(self) -> int:
            return len(self.scores)

    def __init__(self, saveto: PathLike, *, adapter: Adapter['BED.Record']):
        super().__init__(adapter=adapter)
        self.saveto = Path(saveto)
        self.size = 0
        self.stream: Optional[TextIO] = None

    def on_start(self) -> None:
        self.stream = open(self.saveto, 'w')  # noqa: WPS515

    def on_end(self) -> None:
        if not self.stream:
            raise ValueError('Call on_start first.')
        self.stream.close()

        if self.size == 0:
            return

        # Merge intervals
        results = BedTool(self.saveto).sort().merge(s=True, c='5,6', o='max,distinct')
        # fix score / strand columns
        results = [
            Interval(it.chrom, it.start, it.end, score=int(it.fields[3]), strand=it.fields[4])
            for it in results
        ]
        BedTool(results).saveas(self.saveto)

    def __del__(self) -> None:  # noqa: WPS603
        if getattr(self, 'stream', None) is not None:
            self.stream.close()  # type: ignore[union-attr]

    def _on_item_predicted(self, data: 'BED.Record') -> None:
        if self.stream is None:
            raise ValueError('Call on_start first.')

        for start, end, score in zip(data.starts, data.ends, data.scores):
            self.size += 1
            self.stream.write(f'{data.chrom}\t{start}\t{end}\t.\t{score}\t{data.strand}\n')

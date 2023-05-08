from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Any

import numpy as np
import pyBigWig
from numpy import typing as npt
from pybedtools import Interval

from biomancy.typing import ChromSizes, PathLike
from .hook import Hook, Adapter
from .iotools import quantize


class BigWig(Hook['BigWig.Record']):
    @dataclass(frozen=True, slots=True)
    class Record(object):  # noqa: WPS431
        chrom: str
        starts: npt.NDArray[np.int32]
        ends: npt.NDArray[np.int32]
        values: npt.NDArray[np.float32]

        def __post_init__(self) -> None:
            lstarts = len(self.starts)
            lends = len(self.ends)
            lvalues = len(self.values)
            if lstarts != lends or lstarts != lvalues:
                raise ValueError(
                    f'For a correct BigWig record start({lstarts}), end({lends}), and value({lvalues}) ' +
                    'arrays size have the same length',
                )

        @staticmethod
        def from_array(  # noqa: WPS210
            it: Interval,
            data: npt.NDArray[np.float32],
            *,
            roi: Optional[tuple[int, int]] = None,
            quantlvl: Optional[npt.NDArray[Any]] = None,
            skip_below: Union[float, tuple[float, float]] = 1e-3,
            eps: float = 1e-3,
        ) -> Optional['BigWig.Record']:
            if data.ndim != 1 or it.length != data.size:
                raise ValueError('Data must be 1D array with length == interval.length')
            if eps < 0:
                raise ValueError("eps can't be < 0")

            # Trim to ROI
            if roi:
                roi_inside_interval = (0 <= roi[0] < roi[1] <= it.length)
                if not roi_inside_interval:
                    raise ValueError('ROI must be in interval coordinates')
                data = data[roi[0]: roi[1]]
                start = it.start + roi[0]
            else:
                start = it.start

            # Mask low values
            mask = np.ones(len(data), dtype=bool)
            if isinstance(skip_below, tuple):
                left, right = skip_below
                mask[(data >= left) & (data <= right)] = False  # noqa: WPS465
            else:
                mask[data <= skip_below] = False

            # Dense pos-val arrays
            pos = np.flatnonzero(mask) + start
            data = data[mask]

            if data.size == 0:
                return None

            # Quantize data
            if quantlvl is not None:
                quantize(data, quantlvl)

            # Arrays to save results
            starts = np.empty(data.size + 1, dtype=np.int32)
            ends = np.empty(data.size + 1, dtype=np.int32)
            values = np.empty(data.size + 1, dtype=np.float32)
            pointer = 0

            # Derive data 'islands'
            island_starts = np.flatnonzero((pos[1:] - pos[:-1]) > 1) + 1

            island_starts = np.concatenate([[0], island_starts, [pos.size]])
            for st, en in zip(island_starts[:-1], island_starts[1:]):
                pointer = BigWig.Record._continuous_to_encoded(  # noqa: WPS437
                    pos[st], data[st: en], pointer, starts, ends, values, eps,
                )

            starts, ends, values = starts[:pointer], ends[:pointer], values[:pointer]
            if quantlvl is not None:
                quantize(values, quantlvl)

            return BigWig.Record(it.chrom, starts, ends, values)

        def __len__(self) -> int:
            return len(self.values)

        @staticmethod
        def _continuous_to_encoded(
            pos: int,
            data: npt.NDArray[np.float32],
            ind: int,
            starts: npt.NDArray[np.int32],
            ends: npt.NDArray[np.int32],
            values: npt.NDArray[np.float32],
            eps: float,
        ) -> int:
            diff = data[1:] - data[:-1]
            np.abs(diff, out=diff)
            breaks = np.flatnonzero(diff > eps)
            breaks += 1

            N = breaks.size + 1  # noqa: N806,WPS111
            starts, ends, values = starts[ind: ind + N], ends[ind: ind + N], values[ind: ind + N]

            # Intervals coordinates
            starts[0] = pos
            starts[1:] = pos + breaks

            ends[-1] = pos + data.size  # + 1
            ends[:-1] = pos + breaks

            # Values
            cumdata = np.cumsum(data, dtype=np.float32)
            if values.size == 1:
                values[0] = data.mean()
            else:
                values[0] = data[:breaks[0]].mean()
                values[-1] = data[breaks[-1]:].mean()
                if values.size > 2:
                    values[1:-1] = (cumdata[breaks[1:] - 1] - cumdata[breaks[:-1] - 1]) / (ends[1:-1] - starts[1:-1])
            return ind + N

    def __init__(self, chromsizes: ChromSizes, saveto: PathLike, *, adapter: Adapter['BigWig.Record']):
        super().__init__(adapter=adapter)

        if not chromsizes:
            raise ValueError('BigWig file must contain at least one chromosome')

        self.chromsizes = chromsizes
        self.saveto = Path(saveto)
        self.last: Optional[tuple[str, int]] = None
        self.stream: Optional[pyBigWig.pyBigWig] = None

    def on_start(self) -> None:
        self.stream = pyBigWig.open(self.saveto.as_posix(), 'w')
        # Sort by contig name & create a header
        header = sorted(self.chromsizes.items(), key=lambda item: item[0])
        self.stream.addHeader(header)

    def on_end(self) -> None:
        if not self.stream:
            raise ValueError('Call on_start first.')
        self.stream.close()

    def __del__(self) -> None:  # noqa: WPS603
        if getattr(self, 'stream', None) is not None:
            self.stream.close()  # type: ignore[union-attr]

    def _on_item_predicted(self, data: 'BigWig.Record') -> None:
        if not self.stream:
            raise ValueError('Call on_start first.')

        if self.last:
            contig, last_end = self.last
            if data.chrom < contig or (data.chrom == contig and data.starts[0] < last_end):
                raise ValueError('Inputs must be ordered!')

        chrom = [data.chrom for _ in range(len(data))]
        self.stream.addEntries(
            chrom, data.starts, ends=data.ends, values=data.values, validate=False,
        )
        self.last = data.chrom, data.ends[-1]

from typing import Iterable

import numpy as np
from numpy import typing as npt

from .data_source import DataSource
from biomancy.typing import Strand, Data


class Concatenate(DataSource):
    def __init__(
        self,
        sources: Iterable[DataSource],
        dtype: npt.DTypeLike = 'float32',
    ):
        super().__init__(dtype=dtype)

        sources = list(sources)
        if len(sources) <= 1:
            length = len(sources)
            raise ValueError(f'Can concatenate at least 2 data sources, got {length}')

        self.sources = sources

    def _fetch(self, contig: str, strand: Strand, start: int, end: int) -> Data:
        arrays = []
        for sr in self.sources:
            fetched = sr.fetch(contig, strand, start, end)
            if len(fetched.shape) == 1:
                fetched = fetched.reshape(1, -1)
            arrays.append(fetched)

        return np.concatenate(arrays, axis=0, dtype=self.dtype)


def concatenate(*sources: DataSource, dtype: npt.DTypeLike = 'float32') -> DataSource:
    return Concatenate(sources, dtype=dtype)

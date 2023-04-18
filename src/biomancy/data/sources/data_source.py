from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from numpy import typing as npt

from ..typing import Strand


class DataSource(ABC):
    def __init__(self, *, dtype: npt.DTypeLike = 'float32'):
        self.dtype = dtype

    def fetch(self, contig: str, strand: Strand, start: int, end: int) -> np.ndarray:
        if start >= end:
            raise ValueError(f'Start must be <= end, got {start} > {end}')
        data = self._fetch(contig, strand, start, end)

        # Check & force cast the data if needed
        if data.dtype != self.dtype:
            data = data.astype(self.dtype)

        return data

    def apply(self, fn: Callable[[np.ndarray], np.ndarray]) -> 'DataSource':
        return _Lambda(self, fn)

    def __eq__(self, other):
        return isinstance(other, DataSource) and other.dtype == self.dtype

    __hash__ = None

    @abstractmethod
    def _fetch(self, contig: str, strand: Strand, start: int, end: int) -> np.ndarray:
        raise NotImplementedError()


class _Lambda(DataSource):

    def __init__(self, source: DataSource, fn: Callable[[np.ndarray], np.ndarray], **kwargs):
        super().__init__(**kwargs)
        self.source = source
        self.fn = fn

    def _fetch(self, contig: str, strand: Strand, start: int, end: int) -> np.ndarray:
        fetched = self.source.fetch(contig, strand, start, end)
        return self.fn(fetched)

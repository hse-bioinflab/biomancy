from abc import ABC, abstractmethod
from typing import Callable

from numpy import typing as npt

from biomancy.typing import Strand, Data


class DataSource(ABC):
    def __init__(self, *, dtype: npt.DTypeLike = 'float32'):
        self.dtype = dtype

    def fetch(self, contig: str, strand: Strand, start: int, end: int) -> Data:
        if start >= end:
            raise ValueError(f'Start must be <= end, got {start} > {end}')
        data = self._fetch(contig, strand, start, end)

        # Check & force cast the data if needed
        if data.dtype != self.dtype:
            data = data.astype(self.dtype)

        return data

    def apply(self, fn: Callable[[Data], Data]) -> 'DataSource':
        return _Lambda(self, fn, dtype=self.dtype)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DataSource) and other.dtype == self.dtype

    __hash__ = None  # type: ignore[assignment]

    @abstractmethod
    def _fetch(self, contig: str, strand: Strand, start: int, end: int) -> Data:
        raise NotImplementedError()


class _Lambda(DataSource):

    def __init__(self, source: DataSource, fn: Callable[[Data], Data], *, dtype: npt.DTypeLike):
        super().__init__(dtype=dtype)
        self.source = source
        self.fn = fn

    def _fetch(self, contig: str, strand: Strand, start: int, end: int) -> Data:
        fetched = self.source.fetch(contig, strand, start, end)
        return self.fn(fetched)

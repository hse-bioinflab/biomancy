from abc import ABC, abstractmethod

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

    def __eq__(self, other):
        return isinstance(other, DataSource) and other.dtype == self.dtype

    __hash__ = None

    @abstractmethod
    def _fetch(self, contig: str, strand: Strand, start: int, end: int) -> np.ndarray:
        raise NotImplementedError()

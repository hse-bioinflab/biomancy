from pathlib import Path
from typing import Optional, Any

import numpy as np
import pyBigWig
from numpy import typing as npt

from .data_source import DataSource
from ..typing import Data, Strand

_State = tuple[dict[Strand, Path], float, npt.DTypeLike]


class BigWig(DataSource):

    def __init__(
        self, *,
        naval: float = 0,
        fwd: Optional[Path] = None,
        rev: Optional[Path] = None,
        unstranded: Optional[Path] = None,
        dtype: npt.DTypeLike = 'float32',
    ):
        super().__init__(dtype=dtype)

        self.bws: dict[Strand, Any] = {}
        self.paths: dict[Strand, Path] = {}
        self.naval = naval

        match (fwd, rev, unstranded):
            case (None, None, _) if unstranded is not None:
                self.paths = {'+': unstranded, '-': unstranded}
            case (_, _, None) if fwd is not None and rev is not None:
                self.paths = {'+': fwd, '-': rev}
            case _:
                raise ValueError('Data must be either unstranded (fwd == rev == None) or stranded (unstranded == None)')

    def __getstate__(self) -> _State:
        return self.paths, self.naval, self.dtype

    def __setstate__(self, state: _State) -> None:
        self.paths = state[0]
        self.naval = state[1]
        self.dtype = state[2]
        self.bws = {}

    def __del__(self) -> None:  # noqa: WPS603
        for bw in self.bws.values():
            bw.close()
        self.bws.clear()

    def __eq__(self, other: object) -> bool:
        return super().__eq__(other) and \
            isinstance(other, BigWig) \
            and other.naval == self.naval \
            and other.paths == self.paths

    def _fetch(self, contig: str, strand: Strand, start: int, end: int) -> Data:
        if strand not in self.bws:
            self.bws[strand] = pyBigWig.open(self.paths[strand].as_posix())

        values: Data = self.bws[strand].values(contig, start, end, numpy=True)
        values[np.isnan(values)] = self.naval
        return values

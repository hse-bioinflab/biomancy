from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy import typing as npt

from .fasta import Fasta
from ..data_source import DataSource
from biomancy.typing import Strand, Data

DEFAULT_ONE_HOT_ENCODING = (
    ('A', [0]),
    ('C', [1]),
    ('G', [2]),
    ('T', [3]),
    ('U', [3]),
    ('M', [0, 1]),
    ('R', [0, 2]),
    ('W', [0, 3]),
    ('S', [1, 2]),
    ('Y', [1, 3]),
    ('K', [2, 3]),
    ('V', [0, 1, 2]),
    ('H', [0, 1, 3]),
    ('D', [0, 2, 3]),
    ('B', [1, 2, 3]),
    ('X', [0, 1, 2, 3]),
    ('N', [0, 1, 2, 3]),
)

NucleotidesOHE = dict[str, list[int]]


class OneHotEncoder(DataSource):
    def __init__(
        self,
        fasta: Union[Fasta, Path],
        *,
        mapping: Optional[NucleotidesOHE] = None,
        dtype: Optional[npt.DTypeLike] = 'float32',
    ):
        super().__init__(dtype=dtype)

        if isinstance(fasta, Path):
            fasta = Fasta(fasta)
        self.fasta = fasta

        # Prepare the mapping matrix
        self.mapping = self._prepare_mapping(mapping)

    def __eq__(self, other: object) -> bool:
        return super().__eq__(other) and \
            isinstance(other, OneHotEncoder) and \
            np.array_equal(other.mapping, self.mapping) and \
            self.fasta == other.fasta

    def _fetch(self, contig: str, strand: Strand, start: int, end: int) -> Data:
        sequence = self.fasta.load(contig, strand, start, end)
        seqarray = np.fromiter(sequence.encode('ASCII'), dtype=np.uint8, count=(end - start))
        # One-hot-encoding
        encoded = self.mapping[:, seqarray]
        return encoded

    def _prepare_mapping(self, mapping: Optional[NucleotidesOHE]) -> Data:
        if mapping is None:
            mapping = dict(DEFAULT_ONE_HOT_ENCODING)

        ohe = np.zeros((4, 255), dtype=self.dtype)
        for letter, indices in mapping.items():
            if not isinstance(letter, str) or len(letter) != 1:
                raise ValueError('Incorrect mapping format')
            upper, lower = ord(letter.upper()), ord(letter.lower())
            weight = 1 / len(indices)
            for ind in indices:
                ohe[ind, upper] = weight
                ohe[ind, lower] = weight
        return ohe

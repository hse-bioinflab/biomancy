from pathlib import Path
from typing import Union

import numpy as np
from numpy import typing as npt

from .fasta import Fasta
from ..data_source import DataSource
from biomancy.typing import Strand, Data


class Tokenizer(DataSource):
    def __init__(
        self,
        fasta: Union[Fasta, Path],
        kmers: int,
        vocab: dict[str, int],
        *,
        dtype: npt.DTypeLike = 'int32',
    ):
        super().__init__(dtype=dtype)

        if kmers < 1:
            raise ValueError(f'Kmers must be >= 1, found {kmers}')
        if isinstance(fasta, Path):
            fasta = Fasta(fasta)

        self.fasta = fasta
        self.kmers = kmers
        self.vocab = vocab

    @staticmethod
    def parse_vocab(config: Path) -> dict[str, int]:
        if not config.is_file():
            raise ValueError(f"Vocab file doesn't exist: {config}")
        with open(config) as stream:
            lines = stream.read().upper()
        tokens = lines.split('\n')  # [:-1]
        return {token.strip('\\'): ind for ind, token in enumerate(tokens)}

    def pad(self, sequence: Data, tolen: int, padtok: str = '[PAD]') -> Data:
        if len(sequence) > tolen:
            raise ValueError('Padding is only possible when current length < target')
        if len(sequence) < tolen:
            pad = self.vocab[padtok]
            sequence = np.pad(sequence, (0, len(sequence) - tolen), 'constant', constant_values=(pad, pad))
        assert len(sequence) == tolen  # noqa: S101
        return sequence

    def __eq__(self, other: object) -> bool:
        return super().__eq__(other) and \
            isinstance(other, Tokenizer) and \
            self.kmers == other.kmers and \
            self.vocab == other.vocab and \
            self.fasta == other.fasta  # noqa: WPS222

    def _fetch(self, contig: str, strand: Strand, start: int, end: int) -> Data:
        sequence = self.fasta.load(contig, strand, start, end)

        # split sequence into kmers & tokenize
        tokens = []
        kmers = (sequence[ind: ind + self.kmers] for ind in range(len(sequence) - self.kmers + 1))
        for kmer in kmers:
            tk = self.vocab.get(kmer, None)
            if tk is None:
                raise ValueError(f'Unknown kmer: {kmer}')
            tokens.append(tk)

        return np.asarray(tokens, dtype=self.dtype)

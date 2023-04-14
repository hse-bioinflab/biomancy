import gzip
from pathlib import Path

from Bio import SeqIO, Seq

from ..data_source import Strand


def _parse(fasta: Path) -> dict[str, str]:
    sequences = {}
    if fasta.name.endswith('gz'):
        stream = gzip.open(fasta, 'rt')
    else:
        stream = open(fasta, 'rt')  # noqa: WPS515

    for contig in SeqIO.parse(stream, 'fasta'):
        sequences[contig.id] = str(contig.seq).upper()

    stream.close()
    return sequences


class Fasta(object):
    def __init__(self, fasta: Path):
        if not fasta.is_file():
            raise ValueError(f"Fasta doesn't exist: {fasta}")
        self.fasta = fasta
        self.sequences = _parse(fasta)

    def load(self, contig: str, strand: Strand, start: int, end: int) -> str:
        if contig not in self.sequences:
            raise ValueError(f'Contig {contig} was not present in the {self.fasta} fasta file.')

        sequence = self.sequences[contig]

        seqlen = len(sequence)
        is_within_sequence = 0 <= start <= end <= seqlen
        if not is_within_sequence:
            coords = f'{contig}:{start}-{end}'
            raise ValueError(
                f'Requested intervals is outside of the sequence with len {seqlen}: {coords}',
            )
        sequence = sequence[start: end]

        if strand == '-':
            sequence = Seq.complement(sequence)
        return sequence

    def __eq__(self, other):
        return isinstance(other, Fasta) and other.fasta == self.fasta and other.sequences == self.sequences

    __hash__ = None

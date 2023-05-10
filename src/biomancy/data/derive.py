import gzip
from pathlib import Path
from typing import Optional

from Bio import SeqIO
from itertools import chain
from joblib import Parallel, delayed
from pybedtools import Interval, BedTool, chromsizes as fetch_chromsizes

from biomancy.typing import BedLike, ChromSizes


def chromsizes(*, assembly: Optional[str] = None, fasta: Optional[Path] = None) -> ChromSizes:
    options = sum([assembly is None, fasta is None])
    if options != 1:
        raise ValueError("'assembly' id *or* 'fasta' path must be specified")

    if assembly is not None:
        result: dict[str, tuple[int, int]] = fetch_chromsizes(assembly)
        return {contig: end for contig, (_, end) in result.items()}
    else:
        assert fasta is not None  # noqa: S101,WPS503
        index = fasta.parent / (fasta.name + ".fai")

        sizes = {}
        with open(index, 'r') as stream:
            for line in stream:
                contig, size = line.split("\t")[:2]
                sizes[contig] = int(size)
        return sizes


def bins(
    source: BedLike,
    binsize: int,
    step: Optional[int] = None,
    exclude: Optional[BedLike] = None,
    dropshort: bool = True,
) -> BedTool:
    source = BedTool(source).sort()

    # Exclude some regions if needed
    if exclude is not None:
        if isinstance(exclude, list):
            exclude = BedTool(exclude)
        exclude = exclude.sort()
        source = source.subtract(exclude)

    step = binsize if step is None else step
    result = source.window_maker(b=source, w=binsize, s=step)

    if dropshort:
        result = result.filter(lambda wd: wd.length == binsize)

    return result.sort()


def ambiguous_sites(
    fasta: Path,
    allowednuc: tuple[str, ...] = ('A', 'a', 'C', 'c', 'G', 'g', 'T', 't'),
    n_jobs: int = -1,
) -> BedTool:
    def job(contig: str, seq: str, allowed: set[str]) -> list[Interval]:  # noqa: WPS430
        ambcursor = None
        ambiguous = []

        for ind, letter in enumerate(seq):
            if letter in allowed:
                if ambcursor is not None:
                    # Save currently tracked ambiguous regions
                    ambiguous.append(Interval(contig, ambcursor, ind))
                    ambcursor = None
            elif ambcursor is None:
                # Start tracking ambiguous regions
                ambcursor = ind

        if ambcursor is not None:
            ambiguous.append(Interval(contig, ambcursor, len(seq)))
        return ambiguous

    if not fasta.is_file():
        raise ValueError(f"Fasta {fasta} doesn't exist.")

    allowed = set(allowednuc)

    if fasta.name.endswith('.gz'):
        stream = gzip.open(fasta, 'rt')
    else:
        stream = open(fasta, 'r')  # noqa: WPS515

    sequences = SeqIO.parse(stream, 'fasta')
    intervals = Parallel(n_jobs=n_jobs)(delayed(job)(seq.id, seq.seq, allowed) for seq in sequences)
    stream.close()

    return BedTool(list(chain(*intervals))).sort()

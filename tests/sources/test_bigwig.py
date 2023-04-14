import numpy as np
import pytest

from biomancy.data import sources
from tests import test_data
from tests.sources.common import ensure_pickle


def test_bigwig():
    # Entries:
    # bigwig.addHeader([('1', 100), ('MT', 20)])
    # bigwig.addEntries(['1', '1', '1'], [0, 35, 50], ends=[15, 50, 100], values=[0.0, 1.0, 200.0])
    # bigwig.addEntries(['MT', 'MT'], [1, 5], ends=[5, 10], values=[-2.0, 10.0])

    # Single stranded data must be passed only as unstranded
    for kwargs in {'fwd': test_data.BIGWIG}, {'rev': test_data.BIGWIG}:
        with pytest.raises(ValueError):
            sources.BigWig(**kwargs)

    unstranded = sources.BigWig(unstranded=test_data.BIGWIG)
    stranded = sources.BigWig(fwd=test_data.BIGWIG, rev=test_data.BIGWIG)

    # Unavailable positions must be returned as zeros
    # Why? Because life is short.
    for contig in '2', '3', '12', 'MT':
        for start, end in (100, 200), (20, 25), (-5, 10):
            for strand in '+', '-':
                for bw in stranded, unstranded:
                    with pytest.raises(Exception):
                        bw.fetch(contig, strand, start, end)

    for bw in stranded, unstranded:
        for strand in '+', '-':
            assert np.array_equiv(bw.fetch('MT', strand, 15, 20), 0)
            assert np.array_equiv(
                bw.fetch('MT', strand, 0, 20),
                [0.0] + [-2.0] * 4 + [10.0] * 5 + [0.0] * 10,
            )
            assert np.array_equiv(
                bw.fetch('1', strand, 10, 60),
                [0.0] * 5 + [0.0] * 20 + [1.0] * 15 + [200.0] * 10,
            )
            assert np.array_equiv(bw.fetch('1', strand, 0, 5), 0)

    for bw in stranded, unstranded:
        ensure_pickle(bw)

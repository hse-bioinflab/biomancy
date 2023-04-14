import numpy as np
import pytest

from biomancy.data import sources
from tests import test_data
from tests.sources.common import ensure_pickle


@pytest.fixture(params=[test_data.BED, test_data.BED_GZ], ids=['.bed', '.bed.gz'])
def path(request):
    return request.param


@pytest.fixture(params=[True, False], ids=['stranded', 'unstranded'])
def stranded(request):
    return request.param


@pytest.fixture(params=[None, 4], ids=['binary', 'scored'])
def score_col(request):
    return request.param


@pytest.fixture
def bed(path, stranded, score_col):
    return sources.BED(path, strand_specific=stranded, score_col=score_col)


def test_bed(bed):
    ensure_pickle(bed)

    # Unknown contig / outside the range
    assert np.array_equiv(bed.fetch('unknown', '+', -10, 100), 0)
    assert np.array_equiv(bed.fetch('18', '-', -100, -1), 0)
    assert np.array_equiv(bed.fetch('19', '+', 0, 15), 0)

    # Bad path
    bad = [
        # Empty
        ('18', '+', 1, 1),
        # Incorrect range
        ('19', '+', -1, -10),
        ('18', '-', 100, 90),
    ]
    for contig, strand, start, end in bad:
        with pytest.raises(ValueError):
            bed.fetch(contig, strand, start, end)

    # Good path
    empty = [
        ('18', '+', 371, 500),
        ('18', '-', 371, 500),
    ]
    for contig, strand, start, end in empty:
        assert np.array_equiv(bed.fetch(contig, strand, start, end), 0)

    score = 1 if bed.score_col is None else 1_000
    match bed.strand_specific:
        case True:
            assert np.array_equiv(bed.fetch('18', '+', 0, 15), score)
            assert np.array_equiv(bed.fetch('18', '+', 40, 60), score)
            assert np.array_equiv(bed.fetch('18', '-', 0, 10_000), 0)
            assert np.array_equiv(bed.fetch('18', '+', 370, 372), [score, 0])

            expected = [0] * 8 + [score] * 782 + [0] * 63 + [score] * 583 + [0] * 224  # noqa: WPS221
            assert np.array_equiv(bed.fetch('19', '-', 6140, 7800), expected)
            assert np.array_equiv(bed.fetch('19', '+', 6140, 7800), 0)
        case False:
            assert np.array_equiv(bed.fetch('18', '-', 0, 15), score)
            assert np.array_equiv(bed.fetch('18', '+', 40, 60), score)
            assert np.array_equiv(bed.fetch('18', '-', 370, 372), [score, 0])

            expected = [0] * 8 + [score] * 782 + [0] * 63 + [score] * 583 + [0] * 224  # noqa: WPS221
            assert np.array_equiv(bed.fetch('19', '-', 6140, 7800), expected)
            assert np.array_equiv(bed.fetch('19', '+', 6140, 7800), expected)
        case _:
            raise NotImplementedError()

import pickle

import pytest
from pybedtools import Interval

from biomancy.serve import strategy


@pytest.mark.parametrize(
    'constructor', [
        lambda: strategy.MergeAndBin(100),
    ],
)
def test_strategy(constructor):
    strategy = constructor()

    # Unknown strand (.) -> error
    intervals = [Interval('1', 0, 100, strand='.'), Interval('2', 10, 1000, strand='+')]
    with pytest.raises(ValueError):
        strategy.partition(intervals)

    # Empty list -> error
    with pytest.raises(ValueError):
        strategy.partition([])

    # Pickling support
    assert strategy == pickle.loads(pickle.dumps(strategy))


def test_merge_and_bin():
    # Incorrect init parameters
    params = [
        (-1, 0.1), (0, 0.1), (1, 1.2), (10, 100),
        (100, 100),
    ]
    for binsize, minoverlap in params:
        with pytest.raises(ValueError):
            strategy.MergeAndBin(binsize, minoverlap)

    st = strategy.MergeAndBin(binsize=100, roisize=0.8)
    assert st.roisize == 80

    # Detached(!) intervals with length < binsize
    workload = (
        [Interval('X', 0, 10, strand='-')],
        [Interval('Y', 0, 100, strand='+'), Interval('1', 0, 10, strand='+')],
        [Interval('1', 0, 50, strand='+'), Interval('1', 50, 100, strand='-')],
    )
    for intervals in workload:
        with pytest.raises(ValueError):
            st.partition(intervals)

    # One interval = one bin
    isingle_interval = [Interval('1', 0, 100, strand='+')]
    single_interval = strategy.Partitions(intervals=isingle_interval, rois=[(0, 100)])
    assert single_interval == st.partition([Interval('1', 0, 100, strand='+')])

    # # Single strand, single chromosome
    single_strand = strategy.Partitions(
        rois=[
            (0, 90), (10, 90), (10, 90), (50, 100),
            (0, 90), (10, 100),
            (0, 90), (10, 90), (10, 100),
            (0, 90), (51, 100),
        ],
        intervals=[
            # 3 intervals chopped
            Interval('2', 0, 100, strand='+'),
            Interval('2', 80, 180, strand='+'),
            Interval('2', 160, 260, strand='+'),
            Interval('2', 200, 300, strand='+'),
            # 2 full intervals
            Interval('2', 310, 410, strand='+'),
            Interval('2', 390, 490, strand='+'),
            # 3 full intervals
            Interval('2', 350, 450, strand='-'),
            Interval('2', 430, 530, strand='-'),
            Interval('2', 510, 610, strand='-'),
            # 2 chopped intervals
            Interval('2', 611, 711, strand='-'),
            Interval('2', 650, 750, strand='-'),
        ],
    )
    isingle_strand = [
        Interval('2', 611, 750, strand='-'), Interval('2', 310, 490, strand='+'),
        Interval('2', 200, 300, strand='+'), Interval('2', 0, 250, strand='+'), Interval('2', 350, 610, strand='-'),
    ]
    assert st.partition(isingle_strand) == single_strand

    # Overlapping strands
    overlapping_strands = strategy.Partitions(
        rois=[
            (0, 90), (45, 100),
            (0, 90), (10, 100),
            (0, 90), (10, 100),
        ],
        intervals=[
            Interval('3', 5, 105, strand='+'),
            Interval('3', 50, 150, strand='+'),

            Interval('3', 200, 300, strand='+'),
            Interval('3', 280, 380, strand='+'),

            Interval('3', 30, 130, strand='-'),
            Interval('3', 110, 210, strand='-'),
        ],
    )

    ioverlapping_strands = [
        Interval('3', 5, 150, strand='+'), Interval('3', 200, 380, strand='+'), Interval('3', 30, 210, strand='-'),
    ]
    assert st.partition(ioverlapping_strands) == overlapping_strands

    # Multi-strand & multi chromosome
    intervals = isingle_interval + isingle_strand + ioverlapping_strands
    expected = strategy.Partitions(
        intervals=single_interval.intervals + single_strand.intervals + overlapping_strands.intervals,
        rois=single_interval.rois + single_strand.rois + overlapping_strands.rois,
    )
    expected.sort(lambda it, _: (it.strand, it.chrom, it.start))

    assert st.partition(intervals) == expected

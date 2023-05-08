from collections import defaultdict

import pytest
from pybedtools import Interval

from biomancy import GenomicDataset
from biomancy.data import StratifiedGenomicSampler


def _make_genomic_dataset(*intervals: tuple[str, int, int, str]) -> GenomicDataset:
    intervals = [Interval(chrom, start, end, strand=strand) for chrom, start, end, strand in intervals]
    return GenomicDataset({}, intervals)


def test_stratified_genomic_sampler_creation():
    dummy = _make_genomic_dataset(('1', 100, 200, '+'))

    # No strata -> error
    with pytest.raises(ValueError):
        StratifiedGenomicSampler(dummy)

    # Incorrect number of requested samples
    for num_samples in -1, 0, 0.32:
        with pytest.raises(ValueError):
            StratifiedGenomicSampler(dummy, num_samples=num_samples)

    def dotest(sampler: StratifiedGenomicSampler):
        cnts = defaultdict(int)
        strata = len(sampler._strata)
        id2strata = {}
        for stratum, ids in enumerate(sampler._strata):
            for id in ids:
                assert id not in id2strata
                id2strata[id] = stratum

        # Sampling size: 1..len(strata) <- each stratum at most 1 time
        workload = [(ss, lambda x: x <= 1) for ss in range(1, strata)]
        workload.extend([
            # Sampling size: len(strata) <- each stratum 1 time
            (strata, lambda x: x == 1),
            # Sampling size: len(strata) + len(strata) - 1 <- each stratum <= 2 times
            (strata + strata - 1, lambda x: 1 <= x <= 2),
            # Sampling size: len(strata) * 94 <- each stratum 94 times
            (strata * 94, lambda x: x == 94),
            # Sampling size: len(strata) * 100 + len(strata) // 2 <- each stratum <= 101 times
            (strata * 100 + strata // 2, lambda x: 100 <= x <= 101)
        ])

        for num_samples, cond in workload:
            cnts.clear()
            sampler.num_samples = num_samples
            for id in sampler:
                cnts[id2strata[id]] += 1
            assert len(cnts) == min(num_samples, strata)
            assert all(cond(stratum) for stratum in cnts.values())

    dataset = _make_genomic_dataset(
        ('1', 100, 200, '+'), ('1', 50, 150, '-'),
        ('2', 300, 400, '+'), ('2', 250, 300, '-'),
        ('3', 0, 100, '+'),
        ('4', 1, 101, '-'),
    )

    # One stratum
    sampler = StratifiedGenomicSampler(dataset, [Interval('1', 0, 1000), Interval('3', 0, 100)])
    dotest(sampler)

    # Multiple strata
    sampler = StratifiedGenomicSampler(
        dataset,
        [Interval('1', 0, 101)],
        [Interval('2', 275, 301)],
        [Interval('3', 50, 100)]
    )
    dotest(sampler)

    # All intervals are inside one stratum
    sampler = StratifiedGenomicSampler(
        dataset,
        [Interval('1', 100, 200), Interval('2', 250, 400), Interval('3', 0, 100), Interval('4', 1, 101)]
    )
    dotest(sampler)

    # All intervals are outside all strata
    sampler = StratifiedGenomicSampler(
        dataset,
        [Interval('1', 1000, 2000)],
        [Interval('2', 2500, 4000)],
        [Interval('3', 1000, 2000), Interval('4', 101, 102)]
    )
    assert len(sampler._strata) == 1
    dotest(sampler)

    # Some strata are not covered
    sampler = StratifiedGenomicSampler(
        dataset,
        [Interval('1', 100, 200), Interval('2', 250, 400)],
        [Interval('3', 0, 100), Interval('4', 1, 101)],
        [Interval('Y', 10, 110)]
    )
    dotest(sampler)

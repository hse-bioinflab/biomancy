from collections import defaultdict
from pathlib import Path

import pytest
from pybedtools import Interval

from biomancy.data import derive
from tests import test_data


def test_ambiguous_sites():
    with pytest.raises(ValueError):
        derive.ambiguous_sites(Path('I-am-missing.fa'))

    for fasta in test_data.FASTA, test_data.FASTA_GZ:
        assert list(derive.ambiguous_sites(fasta)) == [
            Interval('N', 0, 1),
            Interval('N', 5, 10),
            Interval('N', 14, 19),
            Interval('N', 23, 27),
            Interval('__all_upac__', 4, 16),
        ]

        not_a = defaultdict(int)
        for interval in derive.ambiguous_sites(fasta, allowednuc=('A', 'a')):
            not_a[interval.chrom] += interval.length
        assert not_a == {
            '__all_upac__': 15,
            'N': 25,
            '18': 10608,
            '19': 7228,
        }

        everything = tuple('ACGTUWSMKRYBDHVN')
        assert not derive.ambiguous_sites(fasta, allowednuc=everything)


@pytest.mark.parametrize(
    'chromsizes,window,step,exclude,expected',
    [
        ({'1': 100, '2': 20}, 50, None, None, [('1', 0, 50), ('1', 50, 100), ('2', 0, 20)]),
        ({'1': 100, '2': 20}, 50, 75, None, [('1', 0, 50), ('1', 75, 100), ('2', 0, 20)]),
        ({'1': 100, '2': 20}, 50, 40, None, [('1', 0, 50), ('1', 40, 90), ('1', 80, 100), ('2', 0, 20)]),
        ({'1': 100, '2': 20}, 50, None, [('1', 0, 90)], [('1', 90, 100), ('2', 0, 20)]),
        # Fetched chrominfo
        ('dm3', 20_000_000, None, [('chrUextra', 0, 29004656), ('chrYHet', 0, 200_000)], [
            ('chr2L', 0, 20_000_000), ('chr2L', 20_000_000, 23_011_544),
            ('chr3L', 0, 20_000_000), ('chr3L', 20_000_000, 24_543_557),
            ('chr3R', 0, 20_000_000), ('chr3R', 20_000_000, 27_905_053),
            ('chr2R', 0, 20_000_000), ('chr2R', 20_000_000, 21_146_708),
            ('chrX', 0, 20_000_000), ('chrX', 20_000_000, 22_422_827),
            ('chr2LHet', 0, 368_872), ('chr2RHet', 0, 3_288_761), ('chr3LHet', 0, 2_555_491),
            ('chr3RHet', 0, 2_517_507), ('chr4', 0, 1_351_857), ('chrM', 0, 19_517),
            ('chrU', 0, 10_049_037), ('chrXHet', 0, 204_112), ('chrYHet', 200_000, 347_038),
        ]),
    ],
)
def test_bins(chromsizes, window, step, exclude, expected):
    if exclude:
        exclude = [Interval(*excl) for excl in exclude]

    if isinstance(chromsizes, str):
        chromsizes = derive.chromsizes(assembly=chromsizes)

    chromsizes = [Interval(contig, 0, length) for contig, length in chromsizes.items()]

    expected = sorted([Interval(*exp) for exp in expected], key=lambda it: (it.chrom, it.start))
    result = list(derive.bins(chromsizes, window, step, exclude, dropshort=False))
    assert expected == result

    expected = [exp for exp in expected if exp.length == window]
    result = list(derive.bins(chromsizes, window, step, exclude, dropshort=True))
    assert expected == result

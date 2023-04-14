import pickle

import pytest
from pybedtools import Interval

from biomancy import T  # noqa: WPS347


@pytest.fixture
def rois():
    return (
        # 1: 5->60
        Interval('1', 5, 25, strand='+'),
        Interval('1', 10, 30, strand='+'),
        Interval('1', 30, 35, strand='+'),
        Interval('1', 25, 60, strand='-'),
        # 1: 100->120
        Interval('1', 100, 120, strand='.'),
        # 1: 500, 550
        Interval('1', 500, 550, strand='+'),

        # 2: 100->220
        Interval('2', 100, 200, strand='-'),
        Interval('2', 125, 220, strand='+'),

        # MT: 10->11
        Interval('MT', 10, 11, strand='.'),
    )


@pytest.mark.parametrize(
    'constructor', [
        lambda **kwargs: T.intervals.Shift(1, **kwargs),
        lambda **kwargs: T.intervals.RandomizeStrand(**kwargs),
    ],
)
def test_transforms(constructor, rois):
    transform = constructor(pb='always')

    # Empty limits -> error
    with pytest.raises(ValueError):
        transform(interval=Interval('1', 52, 58, strand='+'))

    # Interval outside the limits -> error
    with pytest.raises(AssertionError):
        transform(interval=Interval('3', 1, 10, strand='-'), limits=(15, 20))

    # Pickling support
    assert transform == pickle.loads(pickle.dumps(transform))


def test_inject_limits(rois):
    injector = T.intervals.InjectLimits(rois=rois)

    # Bad path
    bad = [
        # No overlap with ROIs
        ('3', 10, 100, '.'),
        ('2', 0, 1, '+'),
        ('1', 0, 5, '+'),
        # Partial overlap with ROIs
        ('2', 0, 200, '-'),
        ('2', 0, 250, '+'),
        ('1', 60, 61, '+'),
        ('1', 0, 6, '-'),
    ]
    for contig, start, end, strand in bad:
        with pytest.raises(ValueError):
            injector(interval=Interval(contig, start, end, strand=strand))

    # Good path
    workload = [
        [('1', 5, 60), (5, 60)],
        [('1', 10, 32), (5, 60)],
        [('1', 100, 120), (100, 120)],
        [('1', 549, 550), (500, 550)],

        [('2', 100, 220), (100, 220)],
        [('2', 125, 150), (100, 220)],
        [('2', 125, 220), (100, 220)],

        [('MT', 10, 11), (10, 11)],
    ]
    for (contig, start, end), limits in workload:
        assert injector(interval=Interval(contig, start, end))['limits'] == limits


def test_randomize_strand(rois, monkeypatch):
    # fwdp + revp != 1
    bad = [(-1, 2), (0.25, 0.5), (0.0, 0.99)]
    for fwdp, revp in bad:
        with pytest.raises(ValueError):
            T.intervals.RandomizeStrand(fwdp=fwdp, revp=revp)

    randomizer = T.intervals.RandomizeStrand(fwdp=0.6, revp=0.4, pb='always')
    dummy = {'interval': Interval('1', 1, 2, strand='+'), 'limits': (1, 2)}

    monkeypatch.setattr('random.random', lambda *args: 0.5)
    assert randomizer(**dummy)['interval'].strand == '+'
    dummy['interval'].strand = '-'
    assert randomizer(**dummy)['interval'].strand == '+'

    monkeypatch.setattr('random.random', lambda *args: 0.65)
    assert randomizer(**dummy)['interval'].strand == '-'
    dummy['interval'].strand = '-'
    assert randomizer(**dummy)['interval'].strand == '-'


def test_shift(rois, monkeypatch):
    def do_test(start, end, limits, shift, transform):  # noqa: WPS430
        def randint(low, high):  # noqa: WPS430
            assert limits == (low, high)
            return shift

        monkeypatch.setattr('random.randint', randint)
        result = transform(interval=Interval('X', start, end), limits=(50, 60))

        assert result['interval'] == Interval('X', start + shift, end + shift)
        assert result['limits'] == (50, 60)

    # Integer shift
    transform = T.intervals.Shift(5, pb='always')
    workload = [
        [(52, 54), (-2, 5), -2],
        [(50, 60), (0, 0), 0],
        [(50, 51), (0, 5), 5],
        [(59, 60), (-5, 0), -2],
    ]
    for (start, end), limits, shift in workload:
        do_test(start, end, limits, shift, transform)

    # Float shift
    transform = T.intervals.Shift(0.1, pb='always')
    do_test(55, 56, (0, 0), 0, transform)
    do_test(50, 60, (0, 0), 0, transform)

    transform = T.intervals.Shift(0.5, pb='always')
    do_test(50, 60, (0, 0), 0, transform)
    do_test(50, 55, (0, 2), 1, transform)
    do_test(51, 57, (-1, 3), 3, transform)

    transform = T.intervals.Shift(4.0, pb='always')
    do_test(55, 56, (-4, 4), -2, transform)

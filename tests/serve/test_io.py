import numpy as np
import pyBigWig
import pytest
from pybedtools import Interval, BedTool

from biomancy import serve


def test_bigwig(tmp_path):
    # Chromsizes can't be empty
    with pytest.raises(ValueError):
        serve.io.BigWig(chromsizes={}, saveto='', adapter=lambda *args: None)

    # We must be able to create the target file
    # with pytest.raises(FileNotFoundError):
    #     serve.io.BigWig({'1': 50}, "/home/user/i-dont-exist", adapter=lambda *args: None)

    # Normal path
    chromsizes = {'1': 1000, '2': 130, '3': 10, '4': 395}
    quantlvl = np.arange(0.1, 1, step=0.05, dtype=np.float32).round(2)

    def dotest(workload, expect):
        file = tmp_path.joinpath('result.bw')
        bw = serve.io.BigWig(
            chromsizes, file,
            adapter=lambda it, roi, _, pr: serve.io.BigWig.Record.from_array(
                it, pr['proba'], roi=roi, quantlvl=quantlvl,
            ),
        )
        intervals, rois, proba = zip(*workload)
        partitions = serve.Partitions(intervals=intervals, rois=rois)
        assert tuple(rois) == tuple(partitions.rois)
        predicts = {'proba': np.asarray(proba, dtype=np.float32)}

        bw.on_start()
        bw.on_batch_predicted(partitions, {}, predicts)
        bw.on_end()

        with pyBigWig.open(file.as_posix(), 'r') as result:
            for contig, start, end, values in expect:
                bwvalues = result.values(contig, start, end, numpy=True)
                assert np.allclose(bwvalues, values, equal_nan=True)

    # 0 predicts doesn't work -> bug in pybigwig
    # dotest(
    #     [],
    #     [('1', 10, 900, np.nan)]
    # )

    # One interval, one unique value
    workload = [
        [Interval('1', 150, 200, strand='+'), (0, 10), [0.51] * 50],
        [Interval('2', 0, 50, strand='+'), (1, 9), [2] * 50],
        [Interval('4', 300, 350, strand='+'), (0, 1), [1e-12] * 50],
        [Interval('4', 345, 395, strand='+'), (45, 50), [1e-12] * 49 + [0.5]],
    ]
    expected = [
        ('1', 140, 170, [np.nan] * 10 + [0.5] * 10 + [np.nan] * 10),
        ('2', 0, 20, [np.nan] + [0.95] * 8 + [np.nan] * 11),
        ('4', 0, 395, [np.nan] * 394 + [0.5]),
    ]
    dotest(workload, expected)

    # One interval, two unique values
    workload = [
        [Interval('1', 100, 200, strand='+'), (0, 2), [0.1, 0.2] + [123] * 98],
        [Interval('1', 200, 300, strand='+'), (0, 4), [0.1, 0.2, 0.2, 0.2] + [-1] * 96],
        [Interval('1', 300, 400, strand='+'), (0, 3), [0.25, 0.25, 0.1] + [-2] * 97],
        [Interval('1', 350, 450, strand='+'), (0, 100), [0.3] * 50 + [0.5] * 50],
    ]
    expected = [
        ('1', 100, 450,
         [0.1, 0.2] + [np.nan] * 98 +
         [0.1, 0.2, 0.2, 0.2] + [np.nan] * 96 +
         [0.25, 0.25, 0.1] + [np.nan] * 47 +
         [0.3] * 50 + [0.5] * 50),
    ]
    dotest(workload, expected)

    # One interval, multiple unique values
    workload = [
        [Interval('3', 0, 10), (0, 10), [0.1] * 2 + [0.25] * 2 + [0.5] + [0.3] + [0.2] * 4],
        [Interval('4', 10, 20), (1, 9), [0.1] + [0.33] + [0.59] * 3 + [0.6] * 3 + [0.112] + [0.5]],
    ]
    expected = [
        ('3', 0, 10, [0.1] * 2 + [0.25] * 2 + [0.5] + [0.3] + [0.2] * 4),
        ('4', 10, 20, [np.nan] + [0.35] + [0.6] * 6 + [0.1] + [np.nan]),
    ]
    dotest(workload, expected)

    # Constant gapped intervals
    workload = [
        [Interval('1', 10, 20, strand='+'), (0, 10), [0.23] * 1 + [0] * 8 + [0.26] * 1],
        [Interval('2', 25, 35, strand='+'), (1, 9), [0.24] * 2 + [0.23] * 2 + [0] + [0.1] + [0] + [0.9] + [0.89] * 2],
        [Interval('3', 0, 10, strand='+'), (0, 10), [0.2, 0, 0.2, 0.3, 0.4, 0, 0, 0.9, 0.85, 0.85]],
    ]
    expected = [
        ('1', 10, 20, [0.25] * 1 + [np.nan] * 8 + [0.25] * 1),
        ('2', 25, 35, [np.nan] + [0.25] * 3 + [np.nan] + [0.1] + [np.nan] + [0.9, 0.9] + [np.nan]),
        ('3', 0, 10, [0.2, np.nan, 0.2, 0.3, 0.4, np.nan, np.nan, 0.9, 0.85, 0.85]),
    ]
    dotest(workload, expected)


def test_bed(tmp_path):
    def dotest(workload, expect):
        file = tmp_path.joinpath('result.bed')
        bed = serve.io.BED(
            file,
            adapter=lambda it, roi, _, pr: serve.io.BED.Record.from_array(
                it, pr['proba'], roi=roi
            ),
        )

        bed.on_start()

        if workload:
            intervals, rois, proba = zip(*workload)
            partitions = serve.Partitions(intervals=intervals, rois=rois)
            assert tuple(rois) == tuple(partitions.rois)
            predicts = {'proba': np.asarray(proba, dtype=np.float32)}
            bed.on_batch_predicted(partitions, {}, predicts)

        bed.on_end()

        intervals = {(it.chrom, it.strand, it.start, it.end, int(it.score)) for it in BedTool(file)}
        expect = set(expect)
        assert intervals == expect

    # No predictions
    dotest([], ())

    # Single interval in prediction
    workload = [
        [Interval('1', 100, 200, strand='+'), (10, 90), [1.0] * 100],
        [Interval('X', 305, 405, strand='+'), (50, 60), [0.3] * 52 + [0.981] * 6 + [0.49] * 42],
        [Interval('1', 110, 210, strand='-'), (0, 100), [0.0] * 10 + [0.511] * 90],
        [Interval('X', 350, 450, strand='-'), (0, 50), [0.598] * 25 + [0.13] * 75]
    ]
    expect = [
        ('1', '+', 110, 190, 1000),
        ('X', '+', 357, 363, 981),
        ('1', '-', 120, 210, 511),
        ('X', '-', 350, 375, 598),
    ]
    dotest(workload, expect)

    # Multiple intervals in prediction
    workload = [
        # Center
        [Interval('1', 10, 30, strand='+'), (0, 20), [0.0] * 2 + [1.0, 0.9, 0.9, 1.0] + [0.0] * 12 + [0.887] + [0.0]],
        # Left + center
        [Interval('1', 50, 70, strand='+'), (0, 10), [0.559] * 1 + [0.0] * 1 + [0.984, 0.51] * 6 + [0.0] * 6],
        # Right + center
        [Interval('2', 100, 120, strand='+'), (8, 20), [0.13] * 10 + [0.765, 0.51] + [0.0] * 2 + [0.675, 0.669] * 3],
        # Both ends
        [Interval('1', 10, 30, strand='-'), (0, 20), [0.5] * 5 + [0.1] * 10 + [0.5] * 5],
        # Both ends + center
        [Interval('2', 110, 130, strand='-'), (1, 19), [0.5] * 5 + [0.1] + [0.511, .501] * 4 + [0.3] + [0.5] * 5]
    ]
    expect = [
        # Center
        ('1', '+', 12, 16, 1000),
        ('1', '+', 28, 29, 887),
        # Left + center
        ('1', '+', 50, 51, 559),
        ('1', '+', 52, 60, 984),
        # Right + center
        ('2', '+', 110, 112, 765),
        ('2', '+', 114, 120, 675),
        # # Both ends
        ('1', '-', 10, 15, 500),
        ('1', '-', 25, 30, 500),
        # # Both ends + center
        ('2', '-', 111, 115, 500),
        ('2', '-', 116, 124, 511),
        ('2', '-', 125, 129, 500)
    ]
    dotest(workload, expect)

    # Touching intervals
    workload = [
        [Interval('1', 100, 110, strand='+'), (0, 10), [1.0] * 10],
        [Interval('1', 110, 120, strand='+'), (0, 5), [0.55] * 10],
    ]
    expect = [
        ('1', '+', 100, 115, 1000),
    ]
    dotest(workload, expect)

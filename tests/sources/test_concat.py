import numpy as np

from biomancy.data import sources
from tests import test_data
from tests.sources.common import ensure_pickle


def test_concat():
    source = sources.concatenate(
        sources.BED(test_data.BED, strand_specific=False), sources.BED(test_data.BED, strand_specific=True),
        dtype=np.float64,
    )

    result = source.fetch('MT', '+', 25, 100)
    assert result.dtype == 'float64'
    assert np.array_equiv(result, [[1] * 75, [0] * 25 + [1] * 50])

    ensure_pickle(source)

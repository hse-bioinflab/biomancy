import numpy as np
import pytest

from biomancy.data import sources
from tests import test_data
from tests.sources.common import ensure_pickle

BAD_FASTA_INTERVALS = (
    # Unknown contig
    ('MT', '+', 10, 20),
    # Outside the fasta
    ('18', '+', 100000, 200000),
    ('18', '-', -10, 0),
    # Empty
    ('19', '+', 9, 9),
    # Incorrect range
    ('19', '-', 10, 9),
)


@pytest.fixture(
    params=[
        (test_data.FASTA, True), (test_data.FASTA, False), (test_data.FASTA_GZ, True), (test_data.FASTA_GZ, False),
    ],
    ids=['Fasta(.fa)', 'Path(.fa)', 'Fasta(.fa.gz)', 'Path(.fa.gz)'],
)
def fasta(request):
    path, toseq = request.param
    seq = sources.fasta.Fasta(path) if toseq else path
    return seq


@pytest.fixture
def ohe(fasta):
    return sources.fasta.OneHotEncoder(fasta)


def test_ohe(ohe):
    ensure_pickle(ohe)

    # Bad path
    bad = BAD_FASTA_INTERVALS
    for contig, strand, start, end in bad:
        with pytest.raises(ValueError):
            ohe.fetch(contig, strand, start, end)

    # Good path with default mapping
    # TTTTTTGAGA
    encoded = ohe.fetch('18', '+', 5, 15)
    assert np.array_equal(encoded.T, [
        [0, 0, 0, 1], [0, 0, 0, 1],
        [0, 0, 0, 1], [0, 0, 0, 1],
        [0, 0, 0, 1], [0, 0, 0, 1],
        [0, 0, 1, 0], [1, 0, 0, 0],
        [0, 0, 1, 0], [1, 0, 0, 0],
    ])

    # AAAAAACTCT
    encoded = ohe.fetch('18', '-', 5, 15)
    assert np.array_equal(encoded.T, [
        [1, 0, 0, 0], [1, 0, 0, 0],
        [1, 0, 0, 0], [1, 0, 0, 0],
        [1, 0, 0, 0], [1, 0, 0, 0],
        [0, 1, 0, 0], [0, 0, 0, 1],
        [0, 1, 0, 0], [0, 0, 0, 1],
    ])

    # ACGTUWSMKRYBDHVN
    encoded = ohe.fetch('__all_upac__', '+', 0, 16)
    assert np.allclose(encoded.T, [
        [1, 0, 0, 0],  # A
        [0, 1, 0, 0],  # C
        [0, 0, 1, 0],  # G
        [0, 0, 0, 1],  # T
        [0, 0, 0, 1],  # U
        [0.5, 0, 0, 0.5],  # W = A|T
        [0, 0.5, 0.5, 0],  # S = C|G
        [0.5, 0.5, 0, 0],  # M = A|C
        [0, 0, 0.5, 0.5],  # K = G|T
        [0.5, 0, 0.5, 0],  # R = A|G
        [0, 0.5, 0, 0.5],  # Y = C|T
        [0, 1 / 3, 1 / 3, 1 / 3],  # B = not A
        [1 / 3, 0, 1 / 3, 1 / 3],  # D = not C
        [1 / 3, 1 / 3, 0, 1 / 3],  # H = not G
        [1 / 3, 1 / 3, 1 / 3, 0],  # V = not T
        [1 / 4, 1 / 4, 1 / 4, 1 / 4],  # N = any
    ])


@pytest.fixture
def vocab():
    return sources.fasta.Tokenizer.parse_vocab(test_data.VOCAB)


@pytest.fixture
def tokenizer(fasta, vocab):
    return sources.fasta.Tokenizer(fasta, 3, vocab)


def test_tokenizer(tokenizer):
    ensure_pickle(tokenizer)

    assert len(tokenizer.vocab) == 65
    assert tokenizer.vocab['GCT'] == 58

    # Bad path
    bad = list(BAD_FASTA_INTERVALS) + [
        # Outside the vocabulary
        ('__all_upac__', '+', 0, 16),
    ]
    for contig, strand, start, end in bad:
        with pytest.raises(ValueError):
            tokenizer.fetch(contig, strand, start, end)

    # Good path
    # TTTTTTTTTTT
    assert np.array_equiv(tokenizer.fetch('18', '+', 0, 11), [tokenizer.vocab['TTT']] * 9)
    assert np.array_equiv(tokenizer.fetch('18', '-', 0, 11), [tokenizer.vocab['AAA']] * 9)
    # GCTGGGA
    assert np.array_equiv(tokenizer.fetch('19', '+', 33, 40), [58, 40, 32, 64, 61])
    assert np.array_equiv(tokenizer.fetch('19', '-', 33, 40), [45, 51, 11, 43, 42])

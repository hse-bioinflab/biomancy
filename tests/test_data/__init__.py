# flake8: noqa
from pathlib import Path

ROOT = Path(__file__).parent

FASTA = ROOT.joinpath('example.fa')
FASTA_GZ = ROOT.joinpath('example.fa.gz')

BED = ROOT.joinpath('example.bed')
BED_GZ = ROOT.joinpath('example.bed.gz')

BIGWIG = ROOT.joinpath('example.bw')

VOCAB = ROOT.joinpath('vocab.txt')

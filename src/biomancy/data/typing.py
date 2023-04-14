from typing import Union, Literal

from pybedtools import Interval, BedTool

BedLike = Union[BedTool, list[Interval]]
Strand = Literal['+', '-']
ChromSizes = dict[str, int]

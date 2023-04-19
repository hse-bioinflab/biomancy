from typing import Union, Literal, Any

from numpy import typing as npt
from pybedtools import Interval, BedTool

BedLike = Union[BedTool, list[Interval]]
Strand = Literal['+', '-']
ChromSizes = dict[str, int]
Data = npt.NDArray[Any]

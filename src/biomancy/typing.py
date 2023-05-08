import os
from typing import Union, Literal, Any, Iterable

from numpy import typing as npt
from pybedtools import Interval, BedTool

PathLike = Union[str, os.PathLike]
BedLike = Union[BedTool, Iterable[Interval]]
Strand = Literal['+', '-']
ChromSizes = dict[str, int]
Data = npt.NDArray[Any]

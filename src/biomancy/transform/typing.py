from typing import Union, Literal

Probability = Union[Literal['always', 'never'], float]
IntervalLimit = tuple[int, int]

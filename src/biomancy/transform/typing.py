from typing import Union, Literal, Any

Probability = Union[Literal['always', 'never'], float]
IntervalLimit = tuple[int, int]
ToTransform = Any

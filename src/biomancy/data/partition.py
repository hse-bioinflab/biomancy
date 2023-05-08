from typing import Union, Optional

from numpy.random import RandomState
from numpy.typing import ArrayLike
from sklearn.model_selection import train_test_split

from biomancy.typing import BedLike


def randomly(
    *bedlike: BedLike,
    test_size: Union[float, int] = 0.25,
    random_state: Optional[Union[RandomState, int]] = None,
    shuffle: bool = True,
    stratify: Optional[ArrayLike] = None,
) -> list[BedLike]:
    # No reasons to re-implement this function (yet?)

    intervals = [bed if isinstance(bed, list) else list(bed) for bed in bedlike]
    result: list[BedLike] = train_test_split(
        *intervals, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=stratify,
    )
    return result

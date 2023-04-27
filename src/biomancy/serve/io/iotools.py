import numpy as np
from numpy import typing as npt


def quantize(data: npt.NDArray[np.float32], quantlvl: npt.NDArray[np.float32]) -> None:
    diff = (data[:, None] - quantlvl[None, :])
    np.abs(diff, out=diff)
    closest = diff.argmin(axis=-1)
    data[:] = quantlvl[closest]

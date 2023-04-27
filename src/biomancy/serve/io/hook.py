from abc import ABC, abstractmethod
from typing import Any, Optional
from typing import Generic, TypeVar, Callable

from pybedtools import Interval

from ..strategy import Partitions

T = TypeVar('T')  # noqa: WPS111
AdapterResult = Optional[T]
Adapter = Callable[[Interval, tuple[int, int], Any, Any], AdapterResult[T]]


class Hook(Generic[T], ABC):
    def __init__(self, *, adapter: Adapter[T]):
        self.adapter = adapter

    @abstractmethod
    def on_start(self) -> None:
        raise NotImplementedError()

    def on_batch_predicted(self, partitions: Partitions, inputs: Any, predicts: Any) -> None:
        # Split batch into individual items & call the adapter
        N = len(partitions)  # noqa: WPS111,N806
        for ind, (it, roi) in enumerate(zip(partitions.intervals, partitions.rois)):
            # Slice inputs if possible. If not - use as is.
            inpt = {key: ndarray[ind] if len(ndarray) == N else ndarray for key, ndarray in inputs.items()}
            pred = {key: ndarray[ind] if len(ndarray) == N else ndarray for key, ndarray in predicts.items()}

            aresult = self.adapter(it, roi, inpt, pred)
            if aresult:
                self._on_item_predicted(aresult)

    @abstractmethod
    def on_end(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _on_item_predicted(self, data: T) -> None:
        raise NotImplementedError()

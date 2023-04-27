from typing import Iterable, Any

import torch
from numpy import typing as npt
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler

from .io import Hook
from .strategy import PartitionStrategy
from .typing import Predictor
from ..data.sources import DataSource
from ..data.typing import BedLike
from ..genomic_dataset import GenomicDataset

ModelOutput = dict[str, npt.NDArray[Any]]


def run(
    model: Predictor,
    features: dict[str, DataSource],
    intervals: BedLike,
    *,
    hooks: Iterable[Hook[ModelOutput]],
    strategy: PartitionStrategy,
    **kwargs: Any,
) -> None:
    batch_size = kwargs.get('batch_size', None)
    if batch_size is None:
        raise ValueError('batch_size is a must')

    partitions = strategy.partition(intervals)

    # Prepare dataloader
    dataset = GenomicDataset(features, partitions.intervals)
    loader = DataLoader(dataset, sampler=SequentialSampler(dataset), **kwargs)

    # Inference loop
    for hk in hooks:
        hk.on_start()

    ind = 0
    with torch.inference_mode():
        for batch in loader:
            predicts = model(batch)
            pt = partitions[ind: ind + batch_size]

            for hk in hooks:  # noqa: WPS440
                hk.on_batch_predicted(pt, batch, predicts)

            ind += batch_size

    for hk in hooks:  # noqa: WPS440
        hk.on_end()

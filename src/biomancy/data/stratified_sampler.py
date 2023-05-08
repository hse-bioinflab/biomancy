from collections import defaultdict
from typing import Optional, Iterator, Iterable, DefaultDict

import intervaltree as it
import torch
from pybedtools import Interval
from torch.utils.data import Sampler

from biomancy.typing import BedLike
from ..genomic_dataset import GenomicDataset


class StratifiedGenomicSampler(Sampler[int]):
    def __init__(
        self,
        dataset: GenomicDataset,
        *strata: BedLike,
        num_samples: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__(dataset)
        self._dataset = dataset
        self._num_samples = 0

        self.generator = generator
        self.num_samples = num_samples if num_samples else len(dataset)

        if not strata:
            raise ValueError('There must be at least 1 set of intervals defining strata')

        index = self._make_interval_tree(strata)
        self._strata = self._stratify(index, self._dataset.intervals)

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @num_samples.setter
    def num_samples(self, val: int) -> None:
        if not isinstance(val, int) or val <= 0:
            raise ValueError(
                f'num_samples should be a positive integer, but got num_samples={self._num_samples}',
            )
        self._num_samples = val

    def __iter__(self) -> Iterator[int]:
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        N = len(self._strata)  # noqa: WPS111,N806
        to_draw = [self.num_samples // N for _ in range(N)]

        # Add what is left to fill the batch
        tail = self.num_samples % N
        if tail > 0:
            indices = torch.randperm(N, dtype=torch.int64, generator=generator)[:tail].tolist()
            for stidx in indices:
                to_draw[stidx] += 1

        for stratum, cnt in zip(self._strata, to_draw):
            if cnt > 0:
                indices = torch.randint(high=len(stratum), size=(cnt,), generator=generator, dtype=torch.int64).tolist()
                # Map strata indices to dataset indices
                indices = [stratum[ind] for ind in indices]
                yield from indices

    def __len__(self) -> int:
        return self.num_samples

    def _make_interval_tree(self, strata: Iterable[BedLike]) -> dict[str, it.IntervalTree]:
        index: DefaultDict[str, it.IntervalTree] = defaultdict(lambda *args: it.IntervalTree())
        for ind, stratum in enumerate(strata):
            for interval in stratum:
                index[interval.chrom].addi(interval.start, interval.end, data=ind)
        return dict(index)

    def _stratify(self, index: dict[str, it.IntervalTree], intervals: Iterable[Interval]) -> list[list[int]]:
        stratified = defaultdict(list)
        for ind, interval in enumerate(intervals):
            tree: Optional[it.IntervalTree] = index.get(interval.chrom, None)
            if tree is None:
                stratified['None'].append(ind)
                continue

            strata = {rec.data for rec in tree.overlap(interval.start, interval.end)}
            if len(strata) == 0:  # noqa: WPS507
                stratified['None'].append(ind)
            elif len(strata) == 1:
                stratified[strata.pop()].append(ind)
            else:
                raise ValueError(
                    f"Can't stratify interval {interval}, got overlaps with multiple strata files: {strata}",
                )
        return list(stratified.values())

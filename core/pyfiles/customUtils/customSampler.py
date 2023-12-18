import torch.utils.data.distributed as dist
from typing import Iterator, Any, Callable, TypeVar, Generic, Sequence, List, Optional
import torch
import torch.utils.data
T_co = TypeVar('T_co', covariant=True)


class PrepSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: torch.utils.data.Dataset, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        
        self.dataset = dataset
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.total_size = len(self.dataset)
        
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        self.indices = indices

        # Prepare dataset indices with start: Oracles
        try:
            self.dataset.inMem(self.indices)
        except:
            try:
                self.dataset.dataset.inMem(self.indices)
            except:
                raise Exception(f"There is no method name 'inMem' in {self.dataset.__name__}")
    def __iter__(self) -> Iterator[T_co]:
        return iter(self.indices)

    def __len__(self) -> int:
        return self.total_size
    
    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

class PrepDistributedSampler(dist.DistributedSampler):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super(PrepDistributedSampler, self).__init__(*args,**kwargs)

        # Pre-prepare dataset indices with start: Oracles
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        self.indices = indices


        try:
            self.dataset.inMem(self.indices)
        except:
            try:
                self.dataset.dataset.inMem(self.indices)
            except:
                raise Exception("There is no method name 'inMem'")
    def __iter__(self) -> Iterator[T_co]:
        return iter(self.indices)


class OverwriteBatchSampler(torch.utils.data.BatchSampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """
    def __init__(self, sampler, batch_size, drop_last, num_workers):
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        super(OverwriteBatchSampler, self).__init__(sampler, batch_size, drop_last)
        self.num_workers = num_workers

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                try: 
                    rep = 1
                    while rep < self.num_workers:
                        yield batch
                        rep += 1
                except: 
                    pass
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
            try: 
                rep = 1
                while rep < self.num_workers:
                    yield batch
                    rep += 1
            except: 
                pass

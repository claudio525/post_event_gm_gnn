import abc
import dataclasses

import numpy as np


@dataclasses.dataclass
class BaseBatchData(abc.ABC):

    @abc.abstractmethod
    def to_tensor(self) -> "BaseBatchData":
        pass


class BaseDataset(abc.ABC):

    @abc.abstractmethod
    def get_batch(self, indices: np.ndarray, shuffle_rels: bool) -> BaseBatchData:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass


class CustomTabularDataLoader:
    """
    Loosely based on
    https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, dataset: BaseDataset, batch_size: int, shuffle: bool, shuffle_rels: bool):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_rels = shuffle_rels

        # Calculate number of batches
        self.n_samples = len(self.dataset)
        self.n_batches = int(np.ceil(self.n_samples // self.batch_size))

    def __iter__(self):
        if self.shuffle:
            self.indices = np.random.permutation(self.n_samples)
        else:
            self.indices = np.arange(self.n_samples)
        self.i = 0
        return self

    def __len__(self) -> int:
        return self.n_batches

    def __next__(self) -> BaseBatchData:
        if self.i >= len(self.dataset):
            raise StopIteration

        batch_ind = self.indices[self.i: min(self.i + self.batch_size, self.n_samples)]
        self.i += self.batch_size

        # Get, convert and return batch
        return self.dataset.get_batch(batch_ind, self.shuffle_rels).to_tensor()

from abc import abstractmethod
from typing import Iterable, List

from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t


class ClosableDataset(Dataset):
    @abstractmethod
    def close(self): ...


class ClosableDataLoader(DataLoader):
    def close(self):
        self.dataset.close()

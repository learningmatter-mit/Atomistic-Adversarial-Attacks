import random
import torch as ch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def gen_data(x, y, n=5, std=0.1, xnoise=False):
    x = ch.cat([x] * n)
    y = ch.cat([y] * n)
    y += std * ch.randn_like(y)
    if xnoise:
        x += std * ch.randn_like(x)

    return x, y


def gen_2d_data(x, y, z, n=5, std=0.1, xnoise=False, ynoise=False):
    x = ch.cat([x] * n)
    y = ch.cat([y] * n)
    z = z.repeat(n, n)
    z += std * ch.randn_like(z)
    if xnoise:
        x += std * ch.randn_like(x)
    if ynoise:
        y += std * ch.randn_like(y)

    return x, y, z


class VectorDataset(Dataset):
    def __init__(self, *vectors):
        super().__init__()
        self.vectors = ch.stack(vectors, dim=-1)

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, i):
        return self.vectors[i]


class PotentialDataset(Dataset):
    def __init__(self, x, energy, forces):
        super().__init__()
        self.x = self.format(x)
        self.e = self.format(energy)
        self.f = self.format(forces, reshape=False)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.e[i], self.f[i]

    def __add__(self, other):
        if self.is_empty and other.is_empty:
            return self

        if self.is_empty:
            return other

        if other.is_empty:
            return self

        x = ch.cat([self.x, other.x])
        e = ch.cat([self.e, other.e])
        f = ch.cat([self.f, other.f])

        return self.__class__(x, e, f)

    def format(self, tensor, reshape=True):
        if len(tensor) == 0:
            return ch.tensor([])

        if reshape:
            return ch.tensor(tensor).reshape(len(tensor), -1)
        return ch.tensor(tensor)

    def split_train_test(self, test_size=0.2):
        idx = list(range(len(self)))
        train, test = train_test_split(idx, test_size=test_size)
        return self.__class__(*self[train]), self.__class__(*self[test])

    def split_train_validation_test(self, val_size=0.2, test_size=0.2):
        train, validation = self.split_train_test(test_size=val_size)
        train, test = train.split_train_test(test_size=test_size / (1 - val_size))

        return train, validation, test

    @property
    def is_empty(self):
        """Returns True if the dataset is empty"""
        return ch.prod(ch.tensor(self.x.shape)).item() == 0

    def add_batch(self, batch):
        """Adds a new batch to the dataset"""
        x, e, f = batch

        if self.is_empty:
            self.x = x
            self.e = e
            self.f = f

        else:
            self.x = ch.cat([self.x, x], dim=0)
            self.e = ch.cat([self.e, e], dim=0)
            self.f = ch.cat([self.f, f], dim=0)

    def copy(self):
        return self.__class__(self.x, self.e, self.f)

    @classmethod
    def from_empty_dataset(cls):
        return cls(
            ch.tensor([[]]),
            ch.tensor([[]]),
            ch.tensor([[]]),
        )

    def get_loader(self, batch_size, **kwargs):
        return DataLoader(self, batch_size, **kwargs)

    def sample(self, sample_size):
        idx = list(range(len(self)))
        sample = random.sample(idx, sample_size)
        return self.__class__(*self[sample])

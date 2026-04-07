"""Lightweight torch stub backed by numpy.

Only implements the subset used by camel-oasis recsys:
  torch.device, torch.cuda.is_available, torch.tensor, torch.matmul,
  torch.norm, torch.rand, torch.topk, torch.stack, torch.cat, torch.no_grad
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

import numpy as np

from torch import cuda  # noqa: F401 – make ``torch.cuda`` importable


# -- Tensor wrapper ----------------------------------------------------------

class Tensor:
    """Thin wrapper around numpy.ndarray that oasis recsys expects."""

    __slots__ = ("_data",)

    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data)

    # numpy interop
    def numpy(self) -> np.ndarray:
        return self._data

    def __array__(self, dtype=None, copy=None):
        if dtype is not None:
            return self._data.astype(dtype)
        return self._data

    # shape / dtype forwarding
    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def T(self):
        return Tensor(self._data.T)

    def view(self, *shape):
        return Tensor(self._data.reshape(shape))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        result = self._data[key]
        if isinstance(result, np.ndarray):
            return Tensor(result)
        return result

    def __repr__(self):
        return f"stub_tensor({self._data!r})"


# -- Factory functions -------------------------------------------------------

def tensor(data: Any, *, dtype=None, device=None) -> Tensor:
    arr = np.asarray(data)
    if dtype is not None:
        arr = arr.astype(dtype)
    return Tensor(arr)


def rand(*size, device=None) -> Tensor:
    return Tensor(np.random.rand(*size))


def stack(tensors: Sequence, dim: int = 0) -> Tensor:
    arrays = [np.asarray(t) for t in tensors]
    return Tensor(np.stack(arrays, axis=dim))


def cat(tensors: Sequence, dim: int = 0) -> Tensor:
    arrays = [np.asarray(t) for t in tensors]
    return Tensor(np.concatenate(arrays, axis=dim))


# -- Math operations ---------------------------------------------------------

def matmul(a, b) -> Tensor:
    return Tensor(np.asarray(a) @ np.asarray(b))


def norm(input, dim=None, **kwargs) -> Tensor:
    return Tensor(np.linalg.norm(np.asarray(input), axis=dim))


def topk(input, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
    arr = np.asarray(input)
    if largest:
        indices = np.argsort(arr, axis=dim)[..., -k:]
        if sorted:
            indices = np.flip(indices, axis=dim)
    else:
        indices = np.argsort(arr, axis=dim)[..., :k]
    values = np.take_along_axis(arr, indices, axis=dim)
    return Tensor(values), Tensor(indices.astype(np.int64))


# -- Device ------------------------------------------------------------------

class _Device:
    def __init__(self, name: str = "cpu"):
        self._name = name

    def __repr__(self):
        return f"device('{self._name}')"

    def __str__(self):
        return self._name

    def __eq__(self, other):
        if isinstance(other, _Device):
            return self._name == other._name
        if isinstance(other, str):
            return self._name == other
        return NotImplemented


def device(name: str = "cpu") -> _Device:
    return _Device(name)


# -- Autograd no-op ----------------------------------------------------------

class no_grad:
    """Context manager / decorator that is a no-op in this stub."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper


# -- dtype aliases -----------------------------------------------------------

float32 = np.float32
float64 = np.float64
int32 = np.int32
int64 = np.int64

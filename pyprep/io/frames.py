"""Readers for movie / dose-fraction files, dispatched by file extension.

Every reader returns frames as a numpy array of shape (n_frames, ny, nx) in the
file's native integer or float type; conversion to float happens on the GPU.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import mrc

MOVIE_EXTENSIONS = (".mrc", ".mrcs", ".tif", ".tiff", ".eer")


class MovieReader:
    path: Path
    n_frames: int
    shape: tuple  # (ny, nx)
    pixel_size: float | None = None

    def read(self, start: int = 0, count: int | None = None) -> np.ndarray:
        raise NotImplementedError


class MrcMovie(MovieReader):
    def __init__(self, path, signed_bytes: bool | None = None):
        self.path = Path(path)
        self.header = mrc.read_header(self.path)
        self.n_frames = self.header.nz
        self.shape = (self.header.ny, self.header.nx)
        self.pixel_size = self.header.pixel_size
        self.signed_bytes = signed_bytes

    def read(self, start=0, count=None):
        return mrc.read_sections(self.path, start, count, header=self.header,
                                 signed_bytes=self.signed_bytes)


class TiffMovie(MovieReader):
    def __init__(self, path):
        import tifffile
        self.path = Path(path)
        with tifffile.TiffFile(self.path) as tf:
            self.n_frames = len(tf.pages)
            self.shape = tuple(tf.pages[0].shape[-2:])

    def read(self, start=0, count=None):
        import tifffile
        if count is None:
            count = self.n_frames - start
        data = tifffile.imread(self.path, key=range(start, start + count))
        return data.reshape(count, *self.shape)


def open_movie(path: str | os.PathLike, signed_bytes: bool | None = None) -> MovieReader:
    ext = Path(path).suffix.lower()
    if ext in (".mrc", ".mrcs"):
        return MrcMovie(path, signed_bytes)
    if ext in (".tif", ".tiff"):
        return TiffMovie(path)
    if ext == ".eer":
        raise NotImplementedError("EER support is planned but not implemented yet")
    raise ValueError(f"Unrecognised movie format: {path}")

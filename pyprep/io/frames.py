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
    shape: tuple  # (ny, nx) as rendered
    pixel_size: float | None = None
    upsampling: int = 1             # rendered pixels per physical pixel (EER super-resolution)
    is_eer: bool = False

    def read(self, start: int = 0, count: int | None = None) -> np.ndarray:
        raise NotImplementedError

    def frame_counts(self) -> list[int] | None:
        """Raw detector frames summed into each returned frame (EER fractions), or None."""
        return None

    def describe(self) -> str:
        return f"{self.path.suffix[1:].upper()} {self.n_frames} frames {self.shape[1]}x{self.shape[0]}"


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


def open_movie(path: str | os.PathLike, options=None, signed_bytes: bool | None = None) -> MovieReader:
    """Open a movie; ``options`` is an InputSettings (EER fractionation/rendering)."""
    ext = Path(path).suffix.lower()
    if ext in (".mrc", ".mrcs"):
        return MrcMovie(path, signed_bytes)
    if ext in (".tif", ".tiff"):
        return TiffMovie(path)
    if ext == ".eer":
        from .eer import EerMovie
        group = int(getattr(options, "eer_group", 0) or 0)
        movie = EerMovie(path, fractions=None if group else int(getattr(options, "eer_fractions", 10)),
                         group=group or None, upsampling=int(getattr(options, "eer_upsampling", 1)))
        movie.is_eer = True
        return movie
    raise ValueError(f"Unrecognised movie format: {path}")

"""Blosc2 dataset adapter for Windows paths containing non-ASCII characters."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import re
from threading import RLock

import numpy as np

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2


_CHDIR_LOCK = RLock()
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@contextmanager
def _directory_for_ascii_basename(directory: Path):
    # Blosc2 3.11.1 on this Windows runtime fails on Unicode absolute urlpaths.
    # A process-local cwd and an ASCII basename preserve the original file format.
    with _CHDIR_LOCK:
        previous = Path.cwd()
        try:
            os.chdir(directory)
            yield
        finally:
            os.chdir(previous)


class FDKDatasetBlosc2(nnUNetDatasetBlosc2):
    @staticmethod
    def save_case(data, seg, properties, output_filename_truncated: str, **kwargs):
        target = Path(output_filename_truncated)
        if not _IDENTIFIER.fullmatch(target.name):
            raise ValueError("Unsafe FDK case identifier")
        with _directory_for_ascii_basename(target.parent):
            nnUNetDatasetBlosc2.save_case(data, seg, properties, target.name, **kwargs)

    def load_case(self, identifier):
        if not _IDENTIFIER.fullmatch(identifier):
            raise ValueError("Unsafe FDK case identifier")
        with _directory_for_ascii_basename(Path(self.source_folder)):
            original = self.source_folder
            try:
                self.source_folder = "."
                data, seg, previous_seg, properties = super().load_case(identifier)
                directory = Path(original).resolve()
                return (UnicodePathBloscArray(data, directory),
                        UnicodePathBloscArray(seg, directory),
                        UnicodePathBloscArray(previous_seg, directory) if previous_seg is not None else None,
                        properties)
            finally:
                self.source_folder = original


class UnicodePathBloscArray:
    """Keep Blosc2's patch-level lazy access while resolving its relative urlpath."""

    def __init__(self, array, directory: Path):
        self._array = array
        self._directory = directory
        self.shape = array.shape
        self.dtype = array.dtype
        self.ndim = len(array.shape)

    def __getitem__(self, index):
        with _directory_for_ascii_basename(self._directory):
            return self._array[index]

    def __array__(self, dtype=None):
        result = self[()]
        return np.asarray(result, dtype=dtype)

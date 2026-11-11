"""FDK NIfTI I/O preserving the original ijk grid and physical units."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import nibabel as nib
import numpy as np

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.phase3.nifti_io import SpatialMeta, read_nifti


class FDKNiftiIO(BaseReaderWriter):
    supported_file_endings = [".nii", ".nii.gz"]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        if not image_fnames:
            raise ValueError("At least one FDK image is required")
        images = []
        reference = None
        for name in image_fnames:
            array, meta = read_nifti(name)
            if reference is None:
                reference = meta
            elif (meta.original_shape != reference.original_shape
                  or meta.unit != reference.unit
                  or not np.array_equal(meta.affine, reference.affine)):
                raise ValueError(f"FDK channels have different physical grids: {image_fnames[0]} and {name}")
            images.append(array)
        assert reference is not None
        spacing_ijk = np.linalg.norm(reference.affine_mm[:3, :3], axis=0)
        properties = {
            "spacing": tuple(float(v) for v in spacing_ijk[::-1]),
            "fdk_meta": reference,
        }
        return np.stack(images, axis=0).astype(np.float32, copy=False), properties

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        seg, properties = self.read_images((seg_fname,))
        if not np.isin(seg, (0, 1)).all():
            raise ValueError("Raw FDK labels must contain only 0 and 1")
        return seg, properties

    @staticmethod
    def _write(array: np.ndarray, output_fname: str, properties: dict, *, probability: bool) -> None:
        meta = properties.get("fdk_meta")
        if not isinstance(meta, SpatialMeta):
            raise ValueError("Missing FDK original-grid metadata")
        array = np.asarray(array)
        expected = tuple(reversed(meta.original_shape))
        if array.shape != expected:
            raise ValueError(f"Output shape {array.shape} differs from original grid {expected}")
        if probability:
            if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
                raise ValueError("Probability must be finite and in [0, 1]")
            dtype = np.float32
        else:
            if not np.isin(array, (0, 1)).all():
                raise ValueError("Segmentation must be binary")
            dtype = np.uint8
        header = meta.header.copy()
        header.set_data_dtype(dtype)
        header.set_slope_inter(1.0, 0.0)
        nii = nib.Nifti1Image(array.transpose(2, 1, 0).astype(dtype, copy=False), meta.affine, header)
        qform, qcode = meta.header.get_qform(coded=True)
        sform, scode = meta.header.get_sform(coded=True)
        if qcode:
            nii.set_qform(qform, int(qcode))
        if scode:
            nii.set_sform(sform, int(scode))
        nii.header.set_slope_inter(1.0, 0.0)
        nib.save(nii, str(output_fname))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        self._write(seg, output_fname, properties, probability=False)

    def write_probability(self, array: np.ndarray, path: str | Path, properties: dict) -> None:
        self._write(array, str(path), properties, probability=True)

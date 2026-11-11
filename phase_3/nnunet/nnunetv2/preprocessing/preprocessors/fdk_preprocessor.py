"""FDK preprocessor with full-volume bbox, fixed views and no interpolation."""

from __future__ import annotations

import json
from typing import List, Union

import numpy as np

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.training.dataloading.nnunet_dataset import comp_blosc2_params
from nnunetv2.phase3.planning import validate_phase3_plans
from nnunetv2.phase3.storage import FDKDatasetBlosc2


class FDKPreprocessor(DefaultPreprocessor):
    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str,
                      plans_manager, configuration_manager, dataset_json: Union[dict, str]):
        data, seg, properties = self.run_case(image_files, seg_file, plans_manager,
                                              configuration_manager, dataset_json)
        data = data.astype(np.float32, copy=False)
        seg = seg.astype(np.int16, copy=False)
        blocks, chunks = comp_blosc2_params(data.shape, tuple(configuration_manager.patch_size), data.itemsize)
        blocks_seg, chunks_seg = comp_blosc2_params(seg.shape, tuple(configuration_manager.patch_size), seg.itemsize)
        FDKDatasetBlosc2.save_case(data, seg, properties, output_filename_truncated,
                                   chunks=chunks, blocks=blocks,
                                   chunks_seg=chunks_seg, blocks_seg=blocks_seg)

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager, configuration_manager, dataset_json: Union[dict, str]):
        validate_phase3_plans(plans_manager.plans)
        if isinstance(dataset_json, str):
            dataset_json = json.loads(open(dataset_json, encoding="utf-8").read())
        if data.ndim != 4 or data.shape[0] != 2 or not np.isfinite(data).all():
            raise ValueError("FDK preprocessing requires two finite aligned views")
        if seg is not None and (seg.shape != (1, *data.shape[1:]) or not np.isin(seg, (0, 1)).all()):
            raise ValueError("FDK raw segmentation must be aligned and binary")
        if plans_manager.transpose_forward != [0, 1, 2]:
            raise ValueError("FDK axis transpose must be identity")
        if configuration_manager.normalization_schemes != ["NoNormalization", "NoNormalization"]:
            raise ValueError("FDK views must not be normalized twice")
        original_spacing = tuple(float(v) for v in properties["spacing"])
        target_spacing = tuple(float(v) for v in configuration_manager.spacing)
        shape = tuple(int(v) for v in data.shape[1:])
        data = configuration_manager.resampling_fn_data(data.astype(np.float32, copy=True), shape,
                                                         original_spacing, target_spacing)
        if seg is not None:
            seg = configuration_manager.resampling_fn_seg(seg.copy(), shape, original_spacing, target_spacing)
        result_properties = dict(properties)
        result_properties["shape_before_cropping"] = shape
        result_properties["bbox_used_for_cropping"] = [[0, length] for length in shape]
        result_properties["shape_after_cropping_and_before_resampling"] = shape
        if seg is None:
            seg = np.zeros((1, *shape), dtype=np.int8)
        else:
            result_properties["class_locations"] = self._sample_foreground_locations(
                seg, [1, (-1, 0, 1)], verbose=self.verbose)
        return data, seg.astype(np.int8, copy=False), result_properties

    def run_case(self, image_files: List[str], seg_file: Union[str, None], plans_manager,
                 configuration_manager, dataset_json: Union[dict, str]):
        reader = plans_manager.image_reader_writer_class()
        data, properties = reader.read_images(image_files)
        if seg_file is None:
            seg = None
        else:
            seg, seg_properties = reader.read_seg(seg_file)
            left = properties["fdk_meta"]
            right = seg_properties["fdk_meta"]
            if (left.original_shape != right.original_shape or left.unit != right.unit
                    or not np.array_equal(left.affine, right.affine)):
                raise ValueError("FDK image and label physical grids differ")
        return self.run_case_npy(data, seg, properties, plans_manager, configuration_manager, dataset_json)

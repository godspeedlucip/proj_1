"""Reproducible group-balanced native nnU-Net patch loader."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import random
from threading import RLock
from typing import List, Tuple, Union

import numpy as np
import torch

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from .prior_stats import seed_for


_RNG_LOCK = RLock()


class Phase3DataLoader(nnUNetDataLoader):
    def __init__(self, data, batch_size: int, patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray], label_manager,
                 oversample_foreground_percent: float = 0.33, sampling_probabilities=None,
                 pad_sides=None, probabilistic_oversampling: bool = True, transforms=None,
                 *, case_groups: dict[str, str], seed: int):
        if oversample_foreground_percent != 0.33 or not probabilistic_oversampling or sampling_probabilities is not None:
            raise ValueError("Phase 3 uses group sampling and foreground Bernoulli p=0.33")
        if type(seed) is not int or seed < 0:
            raise ValueError("Sampling seed must be a nonnegative integer")
        identifiers = tuple(data.identifiers)
        if not identifiers or any(case_id not in case_groups for case_id in identifiers):
            raise ValueError("Every preprocessed case requires a source group")
        groups = tuple(sorted({case_groups[case_id] for case_id in identifiers}))
        cases_by_group = {group: tuple(sorted(case_id for case_id in identifiers if case_groups[case_id] == group))
                          for group in groups}
        if any(not cases for cases in cases_by_group.values()):
            raise ValueError("Empty source group")
        for case_id in identifiers:
            _, _, _, properties = data.load_case(case_id)
            locations = properties.get("class_locations", {}).get((-1, 0, 1))
            if locations is None or len(locations) == 0:
                raise ValueError(f"Case {case_id} has no valid annotated voxels")
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager,
                         oversample_foreground_percent=oversample_foreground_percent,
                         sampling_probabilities=None, pad_sides=pad_sides,
                         probabilistic_oversampling=True, transforms=transforms)
        self._groups = groups
        self._cases_by_group = cases_by_group
        self._sampling_rng = np.random.default_rng(seed_for(seed, "phase3_group_sampling"))
        self._numpy_state = np.random.RandomState(seed_for(seed, "phase3_patch_bbox")).get_state()
        self._torch_state = torch.Generator().manual_seed(seed_for(seed, "phase3_mirror")).get_state()
        self._python_state = random.Random(seed_for(seed, "phase3_python")).getstate()
        identity = {"cases": cases_by_group, "patch": tuple(int(v) for v in patch_size),
                    "final_patch": tuple(int(v) for v in final_patch_size), "batch": batch_size,
                    "seed": seed, "transforms": repr(transforms)}
        self._identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def get_indices(self):
        result = []
        for _ in range(self.batch_size):
            group = self._groups[int(self._sampling_rng.integers(len(self._groups)))]
            cases = self._cases_by_group[group]
            result.append(cases[int(self._sampling_rng.integers(len(cases)))])
        return result

    def state_dict(self) -> dict:
        return deepcopy({"identity": self._identity,
                         "sampling": self._sampling_rng.bit_generator.state,
                         "numpy": self._numpy_state,
                         "torch": self._torch_state,
                         "python": self._python_state})

    def load_state_dict(self, state: dict) -> None:
        if state.get("identity") != self._identity:
            raise ValueError("Loader case, patch, transform or seed identity mismatch")
        self._sampling_rng.bit_generator.state = deepcopy(state["sampling"])
        self._numpy_state = deepcopy(state["numpy"])
        self._torch_state = state["torch"].clone()
        self._python_state = deepcopy(state["python"])

    def generate_train_batch(self):
        with _RNG_LOCK:
            global_numpy = np.random.get_state()
            global_torch = torch.get_rng_state()
            global_python = random.getstate()
            try:
                np.random.set_state(self._numpy_state)
                torch.set_rng_state(self._torch_state)
                random.setstate(self._python_state)
                for _ in range(16):
                    batch = super().generate_train_batch()
                    target = batch["target"][0] if isinstance(batch["target"], list) else batch["target"]
                    valid_per_sample = (target != 2).flatten(1).any(dim=1)
                    if bool(valid_per_sample.all()):
                        return batch
                raise ValueError("No valid target voxels after 16 patch samples")
            finally:
                self._numpy_state = np.random.get_state()
                self._torch_state = torch.get_rng_state()
                self._python_state = random.getstate()
                np.random.set_state(global_numpy)
                torch.set_rng_state(global_torch)
                random.setstate(global_python)

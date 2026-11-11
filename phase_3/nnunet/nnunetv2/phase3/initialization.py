"""Independent, replayable random streams for model initialization and resume."""

from contextlib import contextmanager
import hashlib
import random

import numpy as np
import torch


def seed_for(root_seed: int, scope: str) -> int:
    payload = f"fdk-v3:{root_seed}:{scope}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2**32


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG state requires CUDA for exact resume")
        torch.cuda.set_rng_state_all([part.cpu() for part in state["cuda"]])


@contextmanager
def named_rng(root_seed: int, scope: str):
    state = capture_rng_state()
    seed = seed_for(root_seed, scope)
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        restore_rng_state(state)

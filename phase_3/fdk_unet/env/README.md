# Environment verification

The default host Python 3.14.3 / PyTorch 2.11.0+cpu runs the ordinary model.
MONAI 1.3.0 fails to import there because `FileFinder.find_module` was removed.
Do not patch the MONAI dependency or silently upgrade its implementation contract.

An isolated `tmp/fdk_v3/fusion310` virtual environment reuses installed Python
3.10.19 / PyTorch 2.3.1+cu118 / NumPy 1.26.4 without modifying the source
environment. MONAI 1.3.0 imports successfully there. The system has an RTX 4070
Ti SUPER (16 GiB); the default CPU-only Python was not evidence of absent GPU.

Prefix, actual SSL loading, offline reconstruction and GPU capacity are separate
acceptance checks. The final verification table and any tested lock file are
updated after those checks; package import alone is not full acceptance.

Verified with the recorded lock: official CPU 32³ and CUDA 128³ prefix equality,
real model_swinvit.pt 100% backbone loading, offline reconstruction, and all six
core models at CUDA 128³/batch 2/float16. CUDA MaxPool3d backward cannot satisfy
strict deterministic algorithms in PyTorch 2.3.1. CUDA therefore warns explicitly
and saves the limitation; CPU continuation remains strictly reproducible.

Official source: <https://monai.readthedocs.io/en/1.3.0/_modules/monai/networks/nets/swin_unetr.html>.

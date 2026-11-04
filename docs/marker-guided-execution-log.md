# Phase 2 E5-U execution log

Date: 2026-10-28

## Implemented

- Added strict Phase 1 package/manifest validation, finite-segment point/band/validity generation, four-channel TIFF export, fold-wise nnU-Net splits, source metadata, and a CLI.
- Added the plans-driven 2D PlainConvUNet prior branch, identity-initialized residual adapters, raw-only intensity transforms, the E5-U Trainer, and the short-budget smoke Trainer.
- Added unit tests, an opt-in actual-command integration test, and the [runbook](marker-guided-runbook.md).
- Export refuses to reuse a non-empty output directory, preventing stale TIFFs from contaminating a new dataset.

## Verification

Command:

```text
python -m pytest phase_2/nnUNet/nnunetv2/tests/test_marker_prior_channels.py phase_2/nnUNet/nnunetv2/tests/test_residual_prior_adapter.py phase_2/nnUNet/nnunetv2/tests/test_marker_guided_transforms.py phase_2/nnUNet/nnunetv2/tests/integration_tests/test_marker_guided_smoke.py -q
```

Result: **17 passed, 1 skipped**. The skipped integration test requires
`MARKER_GUIDED_RUN_INTEGRATION=1`, a Phase 1 manifest/package, installed
nnU-Net console commands, and configured nnU-Net data directories. `compileall`
also completed successfully for the added Python modules/tests.

## Outstanding end-to-end gate

The workspace does not contain the Phase 1 implementation or an exported
prediction package. The active environment has `batchgenerators 0.25.1`; this
nnU-Net checkout requires `batchgenerators >=0.25.5`. Importing the Trainer
fails because `batchgenerators.augmentations.utils.SEG_TIEBREAKS` is missing,
so the real `plan_and_preprocess → train → checkpoint reload → predict` smoke
has not run. The synthetic converter tests verify data wiring only and make
no segmentation-performance claim.

## Ruling

Ruling: define the Phase 2 package as `manifest.json` plus `package.json` with
an inline `cases` list and explicit source registry fields — the Phase 1 code
and exporter are not present, while the handoff contract fixes the fields and
joins but not these container filenames. If the Phase 1 exporter chooses a
different container, adapt the Phase 2 loader to that format; this may require
updating the CLI and its fixtures.

The workspace root has no Git metadata, so there is no branch/base commit,
commit range, or repository diff available for this execution.

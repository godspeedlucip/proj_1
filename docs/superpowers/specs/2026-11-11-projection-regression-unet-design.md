# Phase 2 continuous projection U-Net

## Intent

Predict the continuous implant projection, including its radiopaque markers, from the synthetic stacked image and the same point, band, and validity priors used by the Phase 2 nnU-Net experiment. Keep the source TIFFs in place and pair them through the Phase 1 physical manifest. Divide every target by the training-set maximum and clip to [0, 1], saving the fixed scale in the checkpoint.

## Constraints and approach

- Use the saved Dataset501 `nnUNetPlans.json` 2D `PlainConvUNet` topology through the existing Phase 2 E5-U `MarkerGuidedPlainConvUNet`: one image channel in the main encoder, point and band in a separate prior encoder, and validity-gated modulation of the first two skips. Replace the segmentation head with one linear regression output channel. Retain the saved 1280×1024 patch and batch size 2 as defaults. Use the nnU-Net SGD/poly schedule as the initial training recipe.
- Use `phase1_200/phase1/physical_manifest.json` and its group fold. Resolve each target to `stent_projections/<series>/<variant>/projections/implant/<angle>.tif`; reject missing or misaligned pairs. Train and validate on separate stent series.
- Normalize only the stacked image with whole-image z-score. Build the three prior channels from geometry truth. Use one training-derived target scale for all samples; never scale the target independently per frame.
- Use weighted Smooth L1 for sparse regression and report foreground MAE plus background prediction energy. Save a checkpoint with the architecture and data configuration; permit a short CPU smoke run.

## Approaches considered

1. Adapt the nnU-Net trainer to float labels. This retains the framework but requires changing its segmentation preprocessing, loss, metrics and prediction export.
2. Initially chosen: instantiate the exact planned `PlainConvUNet` from `dynamic-network-architectures` in a small regression training loop. This provides the one-epoch baseline, but only concatenates prior maps at the input.
3. **Chosen after user correction:** reuse the Phase 2 E5-U prior encoder and gated skip adapters, replacing only the output head and loss for regression. This preserves the marker-fusion design while keeping float targets intact.
4. Implement a new U-Net from scratch. This adds architectural differences that would make comparison with the saved plan less direct.

## Limitations

The source projection is sharper than the Gaussian-smoothed implant added to the image. The first experiment measures recovery of that sharp source from the synthetic input. Geometry-truth priors do not measure the effect of Phase 1 predicted marker errors. No real-image accuracy claim is possible without labels.

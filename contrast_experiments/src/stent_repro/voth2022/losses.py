def soft_dice_B(pred, target, smoothing=1):
    """Batch-and-spatial global soft Dice loss; reduction is a porting choice."""
    if smoothing <= 0:
        raise ValueError('smoothing must be positive')
    return 1 - (2 * (pred * target).sum() + smoothing) / (pred.sum() + target.sum() + smoothing)

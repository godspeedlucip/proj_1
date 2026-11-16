def soft_dice_A(pred, target):
    return 1 - (2 * (pred * target).sum() + 1) / (pred.sum() + target.sum() + 1)

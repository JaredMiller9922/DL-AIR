import torch

def pit_mse_loss(pred, y, y_alt=None):
    """
    pred:  (B, 4, T)
    y:     (B, 4, T)
    y_alt: (B, 4, T) or None
    """
    loss_main = ((pred - y) ** 2).mean(dim=(1, 2))

    if y_alt is None:
        return loss_main.mean()

    loss_alt = ((pred - y_alt) ** 2).mean(dim=(1, 2))
    return torch.minimum(loss_main, loss_alt).mean()

def mse_loss(pred, y):
    return torch.mean((pred - y) ** 2)
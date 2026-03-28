import torch

def mse_loss(pred, y):
    return torch.mean((pred - y) ** 2)

def calculate_sdr(pred, y):
    """Returns SDR in dB. High positive values = better separation."""
    noise = y - pred
    # 10 * log10(Signal Power / Noise Power)
    # Adding 1e-8 to avoid log of zero
    res = 10 * torch.log10(torch.sum(y**2, dim=(1,2)) / (torch.sum(noise**2, dim=(1,2)) + 1e-8))
    return res.mean()

def pit_mse_loss(pred, y, y_alt=None):
    """Solves Permutation Ambiguity so loss doesn't start at 50."""
    loss_main = ((pred - y) ** 2).mean(dim=(1, 2))
    if y_alt is None:
        return loss_main.mean()
    loss_alt = ((pred - y_alt) ** 2).mean(dim=(1, 2))
    # Pick the best assignment for each item in the batch
    return torch.minimum(loss_main, loss_alt).mean()
import torch

def _feature_dims(x):
    return tuple(range(1, x.ndim))


def mse_loss(pred, y):
    return torch.mean((pred - y) ** 2)


def calculate_sdr(pred, y):
    """Returns SDR in dB. High positive values = better separation."""
    noise = y - pred
    dims = _feature_dims(y)
    res = 10 * torch.log10(torch.sum(y**2, dim=dims) / (torch.sum(noise**2, dim=dims) + 1e-8))
    return res.mean()


def align_to_pit_target(pred, y, y_alt=None):
    if y_alt is None:
        return y

    dims = _feature_dims(pred)
    loss_main = ((pred - y) ** 2).mean(dim=dims)
    loss_alt = ((pred - y_alt) ** 2).mean(dim=dims)
    use_alt = loss_alt < loss_main

    view_shape = [pred.shape[0]] + [1] * (pred.ndim - 1)
    return torch.where(use_alt.view(*view_shape), y_alt, y)


def pit_mse_loss(pred, y, y_alt=None):
    """Solves Permutation Ambiguity so loss doesn't start at 50."""
    dims = _feature_dims(pred)
    loss_main = ((pred - y) ** 2).mean(dim=dims)
    if y_alt is None:
        return loss_main.mean()
    loss_alt = ((pred - y_alt) ** 2).mean(dim=dims)
    return torch.minimum(loss_main, loss_alt).mean()


def pit_sdr(pred, y, y_alt=None):
    aligned_target = align_to_pit_target(pred, y, y_alt)
    return calculate_sdr(pred, aligned_target)

import copy
import random
import time

import numpy as np
import torch

from utils.model_utils.losses import pit_mse_loss, pit_sdr


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _joint_normalize(x, y=None, y_alt=None, eps=1e-8):
    scale = torch.sqrt(torch.mean(x ** 2, dim=(1, 2), keepdim=True) + eps)
    x = x / scale
    if y is not None:
        y = y / scale
    if y_alt is not None:
        y_alt = y_alt / scale
    return x, y, y_alt


def _apply_receiver_dropout(x, drop_prob):
    if drop_prob <= 0 or x.shape[1] % 2 != 0:
        return x

    batch, channels, time = x.shape
    num_receivers = channels // 2
    x_pairs = x.view(batch, num_receivers, 2, time)

    keep = torch.rand(batch, num_receivers, device=x.device) > drop_prob
    dropped_all = keep.sum(dim=1) == 0
    if dropped_all.any():
        keep[dropped_all, 0] = True

    keep = keep.unsqueeze(-1).unsqueeze(-1)
    kept_receivers = keep.squeeze(-1).squeeze(-1).sum(dim=1, keepdim=True).clamp_min(1)
    scale = (num_receivers / kept_receivers).view(batch, 1, 1, 1)

    x_pairs = x_pairs * keep * scale
    return x_pairs.view(batch, channels, time)


def _prepare_batch(batch, device, normalize_batches=False, training=False, input_noise_std=0.0, receiver_drop_prob=0.0):
    x = batch["x"].to(device, non_blocking=True)
    y = batch["y"].to(device, non_blocking=True)
    y_alt = batch.get("y_alt")
    if y_alt is not None:
        y_alt = y_alt.to(device, non_blocking=True)

    if normalize_batches:
        x, y, y_alt = _joint_normalize(x, y, y_alt)

    if training:
        if receiver_drop_prob > 0:
            x = _apply_receiver_dropout(x, receiver_drop_prob)
        if input_noise_std > 0:
            x = x + input_noise_std * torch.randn_like(x)

    return x, y, y_alt


def _make_scheduler(optimizer, scheduler_name, epochs, scheduler_patience):
    if scheduler_name is None or scheduler_name == "none":
        return None
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=scheduler_patience,
            min_lr=1e-6,
        )
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def evaluate_model(model, data_loader, device, normalize_batches=False):
    model.eval()
    total_val = 0.0
    total_sdr = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            x, y, y_alt = _prepare_batch(batch, device, normalize_batches=normalize_batches)
            pred = model(x)
            total_val += pit_mse_loss(pred, y, y_alt).item()
            total_sdr += pit_sdr(pred, y, y_alt).item()
            total_batches += 1

    return {
        "val_pit_mse": total_val / max(1, total_batches),
        "val_pit_sdr_db": total_sdr / max(1, total_batches),
    }


def train_model(
    model,
    train_loader,
    val_loader,
    plotter=None,
    epochs=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    lr=1e-3,
    weight_decay=0.0,
    optimizer_name="adam",
    scheduler_name=None,
    scheduler_patience=5,
    early_stopping_patience=None,
    grad_clip=None,
    normalize_batches=False,
    input_noise_std=0.0,
    receiver_drop_prob=0.0,
    checkpoint_path=None,
    checkpoint_extra=None,
    use_amp=False,
    seed=None,
):
    if seed is not None:
        seed_everything(seed)

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    scheduler = _make_scheduler(optimizer, scheduler_name, epochs, scheduler_patience)
    amp_enabled = use_amp and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_hist, val_hist = [], []
    best_val = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epoch_times = []
    epochs_without_improvement = 0
    stop_reason = "max_epochs"

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_train = 0.0

        for batch in train_loader:
            x, y, y_alt = _prepare_batch(
                batch,
                device,
                normalize_batches=normalize_batches,
                training=True,
                input_noise_std=input_noise_std,
                receiver_drop_prob=receiver_drop_prob,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                pred = model(x)
                loss = pit_mse_loss(pred, y, y_alt)

            if amp_enabled:
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_train += loss.item()

        avg_train = total_train / len(train_loader)
        val_metrics = evaluate_model(model, val_loader, device, normalize_batches=normalize_batches)
        avg_val = val_metrics["val_pit_mse"]

        train_hist.append(avg_train)
        val_hist.append(avg_val)
        epoch_times.append(time.perf_counter() - epoch_start)

        if scheduler is not None:
            if scheduler_name == "plateau":
                scheduler.step(avg_val)
            else:
                scheduler.step()

        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            if checkpoint_path is not None:
                payload = {
                    "epoch": epoch,
                    "model": best_state,
                    "optimizer": optimizer.state_dict(),
                    "best_val_pit_mse": best_val,
                    "config": checkpoint_extra or {},
                }
                torch.save(payload, checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch}: Train PIT-MSE {avg_train:.4f} | "
                f"Val PIT-MSE {avg_val:.4f} | Val PIT-SDR {val_metrics['val_pit_sdr_db']:.2f} dB"
            )

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            stop_reason = "early_stopping"
            break

    model.load_state_dict(best_state)
    metadata = {
        "best_val_pit_mse": best_val,
        "best_epoch": best_epoch,
        "stopped_after_epoch": len(train_hist),
        "stop_reason": stop_reason,
        "epoch_times_sec": epoch_times,
        "avg_epoch_time_sec": sum(epoch_times) / len(epoch_times) if epoch_times else 0.0,
        "optimizer_name": optimizer_name,
        "scheduler_name": scheduler_name or "none",
        "normalize_batches": normalize_batches,
        "input_noise_std": input_noise_std,
        "receiver_drop_prob": receiver_drop_prob,
        "grad_clip": grad_clip,
        "weight_decay": weight_decay,
        "learning_rate": lr,
        "used_amp": amp_enabled,
    }
    return model, train_hist, val_hist, metadata

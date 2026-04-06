import copy
import time

import torch

from utils.model_utils.losses import pit_mse_loss


def evaluate_model(model, data_loader, device):
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for batch in data_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_alt = batch["y_alt"].to(device)
            pred = model(x)
            total += pit_mse_loss(pred, y, y_alt).item()
            batches += 1
    return total / max(1, batches)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    device,
    lr,
    weight_decay,
    grad_clip,
    checkpoint_path,
    checkpoint_extra,
    optimizer_name="adamw",
    scheduler_name="plateau",
    scheduler_patience=4,
    early_stopping_patience=10,
    use_amp=True,
):
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=scheduler_patience, min_lr=1e-6
        )

    amp_enabled = use_amp and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    train_hist, val_hist = [], []

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_train = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_alt = batch["y_alt"].to(device)

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
        avg_val = evaluate_model(model, val_loader, device)
        train_hist.append(avg_train)
        val_hist.append(avg_val)

        if scheduler is not None:
            scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": best_state,
                    "optimizer": optimizer.state_dict(),
                    "best_val_pit_mse": best_val,
                    "config": checkpoint_extra,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            elapsed = time.perf_counter() - epoch_start
            print(f"Epoch {epoch}: Train PIT-MSE {avg_train:.4f} | Val PIT-MSE {avg_val:.4f} | {elapsed:.1f}s")

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            break

    model.load_state_dict(best_state)
    return model, train_hist, val_hist, {"best_val_pit_mse": best_val, "best_epoch": best_epoch}

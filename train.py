import copy
from pathlib import Path

import torch

from utils.model_utils.losses import pit_mse_loss


def normalize_batch(x, y, y_alt=None, eps=1e-8):
    scale = torch.sqrt(torch.mean(x ** 2, dim=(1, 2), keepdim=True) + eps)
    x = x / scale
    y = y / scale
    if y_alt is not None:
        y_alt = y_alt / scale
    return x, y, y_alt


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    lr=1e-3,
    weight_decay=1e-4,
    grad_clip=1.0,
    checkpoint_path=None,
    normalize_inputs=True,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
        min_lr=1e-5,
    )
    train_hist, val_hist = [], []
    best_val = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    final_lr = lr

    for epoch in range(1, epochs + 1):
        model.train()
        total_train = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_alt = batch.get("y_alt")
            if y_alt is not None:
                y_alt = y_alt.to(device)
            if normalize_inputs:
                x, y, y_alt = normalize_batch(x, y, y_alt)
            pred = model(x)
            loss = pit_mse_loss(pred, y, y_alt)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_train += loss.item()

        avg_train = total_train / len(train_loader)
        
        # Validation Phase
        model.eval()
        total_val = 0
        with torch.no_grad():
            for v_batch in val_loader:
                vx = v_batch["x"].to(device)
                vy = v_batch["y"].to(device)
                vy_alt = v_batch.get("y_alt")
                if vy_alt is not None:
                    vy_alt = vy_alt.to(device)
                if normalize_inputs:
                    vx, vy, vy_alt = normalize_batch(vx, vy, vy_alt)
                v_pred = model(vx)
                total_val += pit_mse_loss(v_pred, vy, vy_alt).item()
        
        avg_val = total_val / len(val_loader)
        train_hist.append(avg_train)
        val_hist.append(avg_val)
        scheduler.step(avg_val)
        final_lr = optimizer.param_groups[0]["lr"]

        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                checkpoint = Path(checkpoint_path)
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model": best_state,
                        "best_val_pit_mse": best_val,
                        "lr": lr,
                        "weight_decay": weight_decay,
                    },
                    checkpoint,
                )
        
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch}: Train MSE {avg_train:.4f} | Val MSE {avg_val:.4f} | LR {final_lr:.6f}")

    model.load_state_dict(best_state)
    metadata = {
        "best_val_pit_mse": best_val,
        "best_epoch": best_epoch,
        "learning_rate": lr,
        "final_learning_rate": final_lr,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "epochs": epochs,
        "normalize_inputs": normalize_inputs,
    }
    return model, train_hist, val_hist, metadata

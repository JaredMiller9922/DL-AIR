import copy
import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from networks.hybrid_separator import HybridSeparator
from networks.iq_cnn_separator import IQCNNSeparator
from networks.lstm_separator import LSTMSeparator
from utils.data_utils.dataset import SavedRFDataset
from utils.model_utils.losses import align_to_pit_target, calculate_sdr, mse_loss, pit_mse_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PITSavedRFDataset(Dataset):
    def __init__(self, split_dir: str):
        self.ds = SavedRFDataset(split_dir)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        if "y_alt" not in sample:
            y = sample["y"]
            sample["y_alt"] = torch.cat([y[2:4], y[0:2]], dim=0)
        return sample


def make_loader(split_dir: str, batch_size: int, shuffle: bool):
    ds = PITSavedRFDataset(split_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available()), ds


def evaluate_model(model, loader, device):
    model.eval()
    total_pit_mse = 0.0
    total_aligned_mse = 0.0
    total_sdr = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_alt = batch["y_alt"].to(device)
            pred = model(x)
            aligned = align_to_pit_target(pred, y, y_alt)
            total_pit_mse += pit_mse_loss(pred, y, y_alt).item()
            total_aligned_mse += mse_loss(pred, aligned).item()
            total_sdr += calculate_sdr(pred, aligned).item()
            n_batches += 1

    return {
        "pit_mse": total_pit_mse / max(1, n_batches),
        "aligned_mse": total_aligned_mse / max(1, n_batches),
        "sdr_db": total_sdr / max(1, n_batches),
    }


def train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    train_hist = []
    val_hist = []
    best_val = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        total_train = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_alt = batch["y_alt"].to(device)

            pred = model(x)
            loss = pit_mse_loss(pred, y, y_alt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train += loss.item()

        avg_train = total_train / len(train_loader)
        val_metrics = evaluate_model(model, val_loader, device)
        avg_val = val_metrics["pit_mse"]
        train_hist.append(avg_train)
        val_hist.append(avg_val)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch}: train PIT-MSE {avg_train:.6f} | val PIT-MSE {avg_val:.6f}")

    model.load_state_dict(best_state)
    return model, train_hist, val_hist, {"best_epoch": best_epoch, "best_val_pit_mse": best_val}


def plot_curves(results, output_dir: Path, title_prefix: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        plt.plot(result["train_history"], label=f"{name} train")
    plt.xlabel("Epoch")
    plt.ylabel("PIT-MSE")
    plt.title(f"{title_prefix}: Training PIT-MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "train_curves.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        plt.plot(result["val_history"], label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("PIT-MSE")
    plt.title(f"{title_prefix}: Validation PIT-MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "val_curves.png", dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="experiments/single_rx_debug/outputs_current_generator")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--title-prefix", type=str, default="Single-RX Current Data")
    args = parser.parse_args()

    seed_everything(13)
    data_root = PROJECT_ROOT / args.data_root
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, train_ds = make_loader(str(data_root / "train"), batch_size=32, shuffle=True)
    val_loader, val_ds = make_loader(str(data_root / "val"), batch_size=32, shuffle=False)
    in_ch = next(iter(train_loader))["x"].shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = {
        "IQ_CNN": IQCNNSeparator(in_ch=in_ch, out_ch=4).to(device),
        "Hybrid": HybridSeparator(in_ch=in_ch, out_ch=4).to(device),
        "LSTM": LSTMSeparator(in_ch=in_ch, out_ch=4).to(device),
    }

    results = {}
    for name, model in models.items():
        print(f"--- {name} ---")
        model, train_hist, val_hist, meta = train_model(model, train_loader, val_loader, device, epochs=args.epochs)
        train_metrics = evaluate_model(model, train_loader, device)
        val_metrics = evaluate_model(model, val_loader, device)
        results[name] = {
            "train_history": train_hist,
            "val_history": val_hist,
            "training_metadata": meta,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

    plot_curves(results, output_dir, args.title_prefix)
    summary = {
        "data_root": str(data_root),
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "in_ch": int(in_ch),
        "device": device,
        "results": results,
    }
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps({name: result["val_metrics"] for name, result in results.items()}, indent=2))
    print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()

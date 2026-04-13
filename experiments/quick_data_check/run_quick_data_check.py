import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from networks.hybrid_separator import HybridSeparator
from networks.iq_cnn_separator import IQCNNSeparator
from networks.lstm_separator import LSTMSeparator
from train import normalize_batch, train_model
from utils.data_utils.dataset import make_loader
from utils.model_utils.losses import align_to_pit_target, calculate_sdr, mse_loss, pit_mse_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model, loader, device, normalize_inputs=False):
    model.eval()
    total_pit_mse = 0.0
    total_aligned_mse = 0.0
    total_sdr = 0.0
    batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_alt = batch.get("y_alt")
            if y_alt is not None:
                y_alt = y_alt.to(device)
            if normalize_inputs:
                x, y, y_alt = normalize_batch(x, y, y_alt)

            pred = model(x)
            aligned = align_to_pit_target(pred, y, y_alt)
            total_pit_mse += pit_mse_loss(pred, y, y_alt).item()
            total_aligned_mse += mse_loss(pred, aligned).item()
            total_sdr += calculate_sdr(pred, aligned).item()
            batches += 1

    return {
        "pit_mse": total_pit_mse / max(1, batches),
        "aligned_mse": total_aligned_mse / max(1, batches),
        "sdr_db": total_sdr / max(1, batches),
    }


def infer_in_ch(train_loader) -> int:
    batch = next(iter(train_loader))
    return int(batch["x"].shape[1])


def plot_histories(output_root: Path, results: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for model_name, result in results.items():
        plt.plot(result["train_history"], label=f"{model_name} train")
    plt.xlabel("Epoch")
    plt.ylabel("PIT-MSE")
    plt.title("Quick Learnability Check: Training PIT-MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "train_curves.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 6))
    for model_name, result in results.items():
        plt.plot(result["val_history"], label=f"{model_name} val")
    plt.xlabel("Epoch")
    plt.ylabel("PIT-MSE")
    plt.title("Quick Learnability Check: Validation PIT-MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "val_curves.png", dpi=220)
    plt.close()


def main():
    seed = 11
    seed_everything(seed)

    data_root = PROJECT_ROOT / "data"
    output_root = PROJECT_ROOT / "experiments" / "quick_data_check" / "outputs"
    checkpoints_root = output_root / "checkpoints"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    train_loader, train_ds = make_loader(str(data_root / "train"), batch_size=32, shuffle=True)
    val_loader, val_ds = make_loader(str(data_root / "val"), batch_size=32, shuffle=False)

    in_ch = infer_in_ch(train_loader)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 20

    models = {
        "IQ_CNN": IQCNNSeparator(in_ch=in_ch, out_ch=4, base_channels=32, dropout=0.0).to(device),
        "Hybrid": HybridSeparator(in_ch=in_ch, out_ch=4, hidden=64, num_blocks=4, dropout=0.0).to(device),
        "LSTM": LSTMSeparator(in_ch=in_ch, out_ch=4, hidden_size=128, dropout=0.1).to(device),
    }

    results = {}
    for model_name, model in models.items():
        print(f"--- Training {model_name} ---")
        checkpoint_path = checkpoints_root / f"{model_name.lower()}_best.pt"
        trained_model, train_hist, val_hist, train_meta = train_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            device=device,
            lr=1e-3,
            weight_decay=1e-4,
            grad_clip=1.0,
            checkpoint_path=str(checkpoint_path),
            normalize_inputs=True,
        )

        train_metrics = evaluate_model(trained_model, train_loader, device, normalize_inputs=True)
        val_metrics = evaluate_model(trained_model, val_loader, device, normalize_inputs=True)
        train_metrics_raw = evaluate_model(trained_model, train_loader, device, normalize_inputs=False)
        val_metrics_raw = evaluate_model(trained_model, val_loader, device, normalize_inputs=False)
        results[model_name] = {
            "train_history": train_hist,
            "val_history": val_hist,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_metrics_raw": train_metrics_raw,
            "val_metrics_raw": val_metrics_raw,
            "training_metadata": train_meta,
            "checkpoint_path": str(checkpoint_path),
        }

        with open(output_root / f"{model_name.lower()}_summary.json", "w") as handle:
            json.dump(results[model_name], handle, indent=2)

    plot_histories(output_root, results)

    summary = {
        "seed": seed,
        "device": device,
        "data_root": str(data_root),
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "epochs": epochs,
        "in_ch": in_ch,
        "notes": "Current dataset is QPSK SOI plus bandlimited-noise interferer, so this quick check reports wave metrics only.",
        "results": results,
    }
    with open(output_root / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    table = {
        model_name: {
            "best_epoch": result["training_metadata"]["best_epoch"],
            "best_val_pit_mse": result["training_metadata"]["best_val_pit_mse"],
            "final_val_pit_mse": result["val_metrics"]["pit_mse"],
            "final_val_pit_mse_raw": result["val_metrics_raw"]["pit_mse"],
            "final_val_sdr_db": result["val_metrics"]["sdr_db"],
        }
        for model_name, result in results.items()
    }
    print("Quick data check complete")
    print(json.dumps(table, indent=2))
    print(f"Artifacts saved to {output_root}")


if __name__ == "__main__":
    main()

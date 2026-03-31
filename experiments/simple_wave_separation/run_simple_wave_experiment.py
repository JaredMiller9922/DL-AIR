import json
import math
import random
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from networks.iq_cnn_separator import IQCNNSeparator
from train import train_model
from utils.model_utils.conversion_helpers import complex_matrix_to_iq_channels, stacked_sources_to_iq
from utils.model_utils.losses import align_to_pit_target, calculate_sdr, mse_loss, pit_mse_loss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleWaveDataset(Dataset):
    def __init__(self, split_dir):
        self.files = sorted(Path(split_dir).glob("*.npz"))
        if not self.files:
            raise ValueError(f"No .npz files found in {split_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        return {
            "x": torch.from_numpy(data["x"]).float(),
            "y": torch.from_numpy(data["y"]).float(),
            "y_alt": torch.from_numpy(data["y_alt"]).float(),
        }


def generate_sources(length, rng):
    t = np.linspace(0.0, 1.0, length, endpoint=False, dtype=np.float32)
    amp_a = rng.uniform(0.9, 1.1)
    amp_b = rng.uniform(0.7, 1.0)
    phase_a = rng.uniform(0.0, 2.0 * math.pi)
    phase_b = rng.uniform(0.0, 2.0 * math.pi)
    source_a = amp_a * np.sin(2.0 * math.pi * 3.0 * t + phase_a)
    source_b = amp_b * np.sin(2.0 * math.pi * 7.0 * t + phase_b)
    return t, source_a.astype(np.float32), source_b.astype(np.float32)


def make_sample(length, rng):
    _, source_a, source_b = generate_sources(length, rng)
    source_a_c = source_a.astype(np.complex64)
    source_b_c = source_b.astype(np.complex64)

    mixing = np.array(
        [
            [1.00, 0.35],
            [0.70, 1.10],
            [1.15, -0.40],
            [-0.45, 0.85],
        ],
        dtype=np.float32,
    )
    mixture = np.zeros((4, length), dtype=np.complex64)
    for rx in range(4):
        mixture[rx] = mixing[rx, 0] * source_a_c + mixing[rx, 1] * source_b_c

    noise = 0.002 * (rng.standard_normal((4, length)) + 1j * rng.standard_normal((4, length)))
    mixture = mixture + noise.astype(np.complex64)

    x = complex_matrix_to_iq_channels(mixture)
    y = stacked_sources_to_iq(source_a_c, source_b_c)
    y_alt = stacked_sources_to_iq(source_b_c, source_a_c)
    return x, y, y_alt, source_a, source_b, mixture.real


def generate_dataset(root_dir, train_size=512, val_size=128, test_size=128, length=256, seed=7):
    root = Path(root_dir)
    if root.exists():
        shutil.rmtree(root)
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "val").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)

    split_sizes = {"train": train_size, "val": val_size, "test": test_size}
    for split_idx, (split_name, split_size) in enumerate(split_sizes.items()):
        rng = np.random.default_rng(seed + split_idx)
        for idx in range(split_size):
            x, y, y_alt, _, _, _ = make_sample(length, rng)
            np.savez_compressed(root / split_name / f"sample_{idx:05d}.npz", x=x, y=y, y_alt=y_alt)

    with open(root / "manifest.json", "w") as handle:
        json.dump({"train": train_size, "val": val_size, "test": test_size, "length": length}, handle, indent=4)


def make_loader(split_dir, batch_size, shuffle):
    dataset = SimpleWaveDataset(split_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def evaluate_on_loader(model, loader, device):
    model.eval()
    total_pit_mse = 0.0
    total_sdr = 0.0
    total_mse = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_alt = batch["y_alt"].to(device)
            pred = model(x)
            aligned = align_to_pit_target(pred, y, y_alt)
            total_pit_mse += pit_mse_loss(pred, y, y_alt).item()
            total_mse += mse_loss(pred, aligned).item()
            total_sdr += calculate_sdr(pred, aligned).item()
            batches += 1
    return {
        "pit_mse": total_pit_mse / batches,
        "aligned_mse": total_mse / batches,
        "sdr_db": total_sdr / batches,
    }


def plot_results(output_dir, source_a, source_b, mixture_rx0, pred_a, pred_b):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.plot(source_a, label="Source A")
    plt.plot(source_b, label="Source B")
    plt.title("Generated Source Waves")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "01_sources.png")
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(mixture_rx0, label="Mixture RX0", color="black")
    plt.title("Mixture Wave (Receiver 0)")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "02_mixture.png")
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(source_a, label="True A")
    axes[0].plot(pred_a, label="Pred A", linestyle="--")
    axes[0].set_title("Separated Wave A")
    axes[0].legend()
    axes[1].plot(source_b, label="True B")
    axes[1].plot(pred_b, label="Pred B", linestyle="--")
    axes[1].set_title("Separated Wave B")
    axes[1].legend()
    axes[1].set_xlabel("Sample")
    plt.tight_layout()
    plt.savefig(output_dir / "03_outputs.png")
    plt.close(fig)


def main():
    seed_everything(7)
    experiment_root = Path(__file__).resolve().parent
    data_root = experiment_root / "data"
    outputs_root = experiment_root / "outputs"
    ckpt_path = outputs_root / "simple_iqcnn_best.pt"

    generate_dataset(data_root, train_size=512, val_size=128, test_size=128, length=256, seed=7)

    train_loader = make_loader(data_root / "train", batch_size=32, shuffle=True)
    val_loader = make_loader(data_root / "val", batch_size=32, shuffle=False)
    test_loader = make_loader(data_root / "test", batch_size=32, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = IQCNNSeparator(in_ch=8, out_ch=4, base_channels=32, dropout=0.0).to(device)
    model, train_hist, val_hist, metadata = train_model(
        model,
        train_loader,
        val_loader,
        epochs=40,
        device=device,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        checkpoint_path=str(ckpt_path),
        normalize_inputs=False,
    )

    val_metrics = evaluate_on_loader(model, val_loader, device)
    test_metrics = evaluate_on_loader(model, test_loader, device)

    sample_rng = np.random.default_rng(999)
    x_np, y_np, y_alt_np, source_a, source_b, mixture_real = make_sample(256, sample_rng)
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)
    y = torch.from_numpy(y_np).unsqueeze(0).to(device)
    y_alt = torch.from_numpy(y_alt_np).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)
        aligned = align_to_pit_target(pred, y, y_alt).cpu().numpy()[0]

    pred_a = aligned[0]
    pred_b = aligned[2]
    plot_results(outputs_root, source_a, source_b, mixture_real[0], pred_a, pred_b)

    summary = {
        "device": device,
        "train_history": train_hist,
        "val_history": val_hist,
        "metadata": metadata,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "checkpoint_path": str(ckpt_path),
        "artifacts": {
            "sources_plot": str(outputs_root / "01_sources.png"),
            "mixture_plot": str(outputs_root / "02_mixture.png"),
            "outputs_plot": str(outputs_root / "03_outputs.png"),
        },
    }
    outputs_root.mkdir(parents=True, exist_ok=True)
    with open(outputs_root / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=4)

    print("Simple wave experiment complete")
    print(json.dumps({"val_metrics": val_metrics, "test_metrics": test_metrics, "best_epoch": metadata["best_epoch"]}, indent=4))
    print(f"Artifacts saved to {outputs_root}")


if __name__ == "__main__":
    main()

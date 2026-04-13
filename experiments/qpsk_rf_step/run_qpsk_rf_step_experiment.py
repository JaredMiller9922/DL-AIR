import json
import math
import random
import shutil
import sys
import argparse
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


class SavedWaveDataset(Dataset):
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


QPSK_CONSTELLATION = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex64) / np.sqrt(2.0)
FIXED_H_A = np.array([1.0 + 0.0j, 0.75 + 0.25j, 1.10 - 0.20j, 0.70 + 0.45j], dtype=np.complex64)
FIXED_H_B = np.array([0.35 + 0.80j, -0.15 + 0.95j, 0.55 + 0.50j, -0.60 + 0.30j], dtype=np.complex64)
PULSE = np.array([0.15, 0.35, 0.35, 0.15], dtype=np.float32)


def qpsk_symbols(num_symbols, rng):
    idx = rng.integers(0, 4, size=num_symbols)
    return QPSK_CONSTELLATION[idx]


def pulse_shape(symbols, sps):
    up = np.zeros(len(symbols) * sps, dtype=np.complex64)
    up[::sps] = symbols
    real = np.convolve(up.real, PULSE, mode="same")
    imag = np.convolve(up.imag, PULSE, mode="same")
    return (real + 1j * imag).astype(np.complex64)


def window_signal(sig, length, start):
    out = np.zeros(length, dtype=np.complex64)
    stop = min(length, start + len(sig))
    out[start:stop] = sig[: stop - start]
    return out


def apply_carrier(sig, base_freq, cfo, phase):
    n = np.arange(len(sig), dtype=np.float32)
    carrier = np.exp(1j * (2.0 * math.pi * (base_freq + cfo) * n / len(sig) + phase))
    return (sig * carrier).astype(np.complex64)


def build_source(length, rng, symbol_range, base_freq):
    sps = 8
    num_symbols = int(rng.integers(symbol_range[0], symbol_range[1] + 1))
    symbols = qpsk_symbols(num_symbols, rng)
    shaped = pulse_shape(symbols, sps)
    start = int(rng.integers(0, max(1, length - len(shaped))))
    framed = window_signal(shaped, length, start)
    amp = rng.uniform(0.8, 1.2)
    cfo = rng.uniform(-0.25, 0.25)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    source = amp * apply_carrier(framed, base_freq=base_freq, cfo=cfo, phase=phase)
    return source.astype(np.complex64), {
        "num_symbols": int(num_symbols),
        "start": int(start),
        "amp": float(amp),
        "cfo": float(cfo),
        "phase": float(phase),
        "base_freq": float(base_freq),
    }


def receiver_coeffs(n_rx):
    if n_rx < 1 or n_rx > len(FIXED_H_A):
        raise ValueError(f"n_rx must be between 1 and {len(FIXED_H_A)}, got {n_rx}")
    return FIXED_H_A[:n_rx], FIXED_H_B[:n_rx]


def generate_sample(length, rng, n_rx=4):
    source_a, meta_a = build_source(length, rng, symbol_range=(20, 28), base_freq=14.0)
    source_b, meta_b = build_source(length, rng, symbol_range=(18, 24), base_freq=36.0)
    h_a, h_b = receiver_coeffs(n_rx)

    mixture = np.zeros((n_rx, length), dtype=np.complex64)
    for rx in range(n_rx):
        mixture[rx] = h_a[rx] * source_a + h_b[rx] * source_b

    noise_std = rng.uniform(0.001, 0.004)
    noise = noise_std * (rng.standard_normal((n_rx, length)) + 1j * rng.standard_normal((n_rx, length)))
    mixture = mixture + noise.astype(np.complex64)

    x = complex_matrix_to_iq_channels(mixture)
    y = stacked_sources_to_iq(source_a, source_b)
    y_alt = stacked_sources_to_iq(source_b, source_a)
    metadata = {
        "source_a": meta_a,
        "source_b": meta_b,
        "noise_std": float(noise_std),
        "n_rx": int(n_rx),
        "h_a_real": np.real(h_a).astype(float).tolist(),
        "h_a_imag": np.imag(h_a).astype(float).tolist(),
        "h_b_real": np.real(h_b).astype(float).tolist(),
        "h_b_imag": np.imag(h_b).astype(float).tolist(),
    }
    return x, y, y_alt, source_a, source_b, mixture, metadata


def save_dataset(root_dir, train_size=1024, val_size=256, test_size=256, length=256, seed=29, n_rx=4):
    root = Path(root_dir)
    if root.exists():
        shutil.rmtree(root)
    for split in ["train", "val", "test"]:
        (root / split).mkdir(parents=True, exist_ok=True)

    split_sizes = {"train": train_size, "val": val_size, "test": test_size}
    metadata = []
    for split_idx, (split_name, split_size) in enumerate(split_sizes.items()):
        rng = np.random.default_rng(seed + split_idx)
        split_meta = []
        for idx in range(split_size):
            x, y, y_alt, _, _, _, meta = generate_sample(length, rng, n_rx=n_rx)
            np.savez_compressed(root / split_name / f"sample_{idx:05d}.npz", x=x, y=y, y_alt=y_alt)
            split_meta.append(meta)
            metadata.append(meta)
        with open(root / f"{split_name}_metadata.json", "w") as handle:
            json.dump(split_meta, handle, indent=2)

    with open(root / "manifest.json", "w") as handle:
        json.dump(
            {
                "train": train_size,
                "val": val_size,
                "test": test_size,
                "length": length,
                "n_rx": n_rx,
                "description": "Two pulse-shaped QPSK packets with distinct center frequencies, random timing/phase/CFO/amplitude, fixed receiver channel mixing, and light noise.",
            },
            handle,
            indent=2,
        )
    return metadata


def make_loader(split_dir, batch_size, shuffle):
    dataset = SavedWaveDataset(split_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def evaluate_model(model, loader, device):
    model.eval()
    total_pit_mse = 0.0
    total_mse = 0.0
    total_sdr = 0.0
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


def plot_dataset_randomness(metadata, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    source_a_cfo = [m["source_a"]["cfo"] for m in metadata]
    source_b_cfo = [m["source_b"]["cfo"] for m in metadata]
    source_a_amp = [m["source_a"]["amp"] for m in metadata]
    source_b_amp = [m["source_b"]["amp"] for m in metadata]
    noise = [m["noise_std"] for m in metadata]
    starts_a = [m["source_a"]["start"] for m in metadata]
    starts_b = [m["source_b"]["start"] for m in metadata]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(source_a_cfo, bins=24, alpha=0.8, label="A CFO")
    axes[0, 0].hist(source_b_cfo, bins=24, alpha=0.8, label="B CFO")
    axes[0, 0].set_title("Carrier Offset Distribution")
    axes[0, 0].legend()
    axes[0, 1].hist(source_a_amp, bins=24, alpha=0.8, label="A amp")
    axes[0, 1].hist(source_b_amp, bins=24, alpha=0.8, label="B amp")
    axes[0, 1].set_title("Amplitude Distribution")
    axes[0, 1].legend()
    axes[1, 0].hist(noise, bins=24, color="tab:red")
    axes[1, 0].set_title("Noise Std Distribution")
    axes[1, 1].scatter(starts_a[:400], starts_b[:400], s=10, alpha=0.6)
    axes[1, 1].set_title("Packet Start Positions")
    axes[1, 1].set_xlabel("source_a start")
    axes[1, 1].set_ylabel("source_b start")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "00_dataset_randomness.png")
    plt.close(fig)


def plot_sample_results(output_dir, source_a, source_b, mixture, pred_aligned, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_a = pred_aligned[0] + 1j * pred_aligned[1]
    pred_b = pred_aligned[2] + 1j * pred_aligned[3]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(source_a.real, label="Source A real")
    axes[0].plot(source_b.real, label="Source B real")
    axes[0].set_title("Clean QPSK-Derived Source Waves")
    axes[0].legend()
    axes[1].plot(source_a.imag, label="Source A imag")
    axes[1].plot(source_b.imag, label="Source B imag")
    axes[1].set_title("Ground-Truth Imaginary Components")
    axes[1].set_xlabel("Sample")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "01_clean_sources.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(mixture[0].real, label="RX0 real", color="black")
    if mixture.shape[0] > 1:
        axes[0].plot(mixture[1].real, label="RX1 real", alpha=0.7)
    axes[0].set_title("Received Mixture Real Parts")
    axes[0].legend()
    axes[1].plot(mixture[0].imag, label="RX0 imag", color="black")
    if mixture.shape[0] > 1:
        axes[1].plot(mixture[1].imag, label="RX1 imag", alpha=0.7)
    axes[1].set_title("Received Mixture Imag Parts")
    axes[1].legend()
    axes[1].set_xlabel("Sample")
    fig.tight_layout()
    fig.savefig(output_dir / "02_received_mixture.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(source_a.real, label="True A real")
    axes[0].plot(pred_a.real, label="Pred A real", linestyle="--")
    axes[0].set_title("Separated Output A")
    axes[0].legend()
    axes[1].plot(source_b.real, label="True B real")
    axes[1].plot(pred_b.real, label="Pred B real", linestyle="--")
    axes[1].set_title("Separated Output B")
    axes[1].legend()
    axes[1].set_xlabel("Sample")
    fig.tight_layout()
    fig.savefig(output_dir / "03_model_outputs.png")
    plt.close(fig)

    freqs = np.fft.fftfreq(len(source_a), d=1.0 / len(source_a))
    keep = freqs >= 0
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(freqs[keep], np.abs(np.fft.fft(source_a))[keep], label="Source A")
    axes[0].plot(freqs[keep], np.abs(np.fft.fft(source_b))[keep], label="Source B")
    axes[0].plot(freqs[keep], np.abs(np.fft.fft(mixture[0]))[keep], label="RX0 mixture", alpha=0.8)
    axes[0].set_title("Frequency-Domain View")
    axes[0].legend()
    axes[1].plot(freqs[keep], np.abs(np.fft.fft(pred_a))[keep], label="Pred A")
    axes[1].plot(freqs[keep], np.abs(np.fft.fft(pred_b))[keep], label="Pred B")
    axes[1].set_title("Separated Output Spectra")
    axes[1].set_xlabel("Frequency Bin")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "04_frequency_view.png")
    plt.close(fig)

    meta_lines = [
        f"n_rx={metadata['n_rx']}",
        f"noise_std={metadata['noise_std']:.4f}",
        f"A: symbols={metadata['source_a']['num_symbols']} start={metadata['source_a']['start']} amp={metadata['source_a']['amp']:.3f} cfo={metadata['source_a']['cfo']:.3f}",
        f"B: symbols={metadata['source_b']['num_symbols']} start={metadata['source_b']['start']} amp={metadata['source_b']['amp']:.3f} cfo={metadata['source_b']['cfo']:.3f}",
        f"h_a_real={np.round(metadata['h_a_real'], 3)}",
        f"h_a_imag={np.round(metadata['h_a_imag'], 3)}",
        f"h_b_real={np.round(metadata['h_b_real'], 3)}",
        f"h_b_imag={np.round(metadata['h_b_imag'], 3)}",
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.text(0.01, 0.98, "Per-sample RF-like randomness\n\n" + "\n".join(meta_lines), va="top", family="monospace")
    fig.tight_layout()
    fig.savefig(output_dir / "05_sample_metadata.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--outputs-root", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-rx", type=int, default=4)
    args = parser.parse_args()

    experiment_root = Path(__file__).resolve().parent
    data_root = args.data_root or (experiment_root / "data")
    outputs_root = args.outputs_root or (experiment_root / "outputs")
    checkpoint_path = args.checkpoint_path or (outputs_root / "qpsk_rf_step_best.pt")

    seed_everything(args.seed)
    metadata = save_dataset(
        data_root,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        length=args.length,
        seed=args.seed,
        n_rx=args.n_rx,
    )
    plot_dataset_randomness(metadata, outputs_root)

    train_loader = make_loader(data_root / "train", batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(data_root / "val", batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(data_root / "test", batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = IQCNNSeparator(in_ch=2 * args.n_rx, out_ch=4, base_channels=32, dropout=0.0).to(device)
    model, train_hist, val_hist, train_meta = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        device=device,
        lr=8e-4,
        weight_decay=0.0,
        grad_clip=1.0,
        checkpoint_path=str(checkpoint_path),
        normalize_inputs=False,
    )

    val_metrics = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)

    sample_rng = np.random.default_rng(404)
    x_np, y_np, y_alt_np, source_a, source_b, mixture, sample_meta = generate_sample(args.length, sample_rng, n_rx=args.n_rx)
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)
    y = torch.from_numpy(y_np).unsqueeze(0).to(device)
    y_alt = torch.from_numpy(y_alt_np).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)
        aligned = align_to_pit_target(pred, y, y_alt).cpu().numpy()[0]

    plot_sample_results(outputs_root, source_a, source_b, mixture, aligned, sample_meta)

    summary = {
        "device": device,
        "description": "Step 3 RF-like experiment: two pulse-shaped QPSK packets with distinct carrier bands, random timing/phase/CFO/amplitude, fixed receiver channel mixing, and light noise.",
        "n_rx": args.n_rx,
        "data_root": str(data_root),
        "train_history": train_hist,
        "val_history": val_hist,
        "metadata": train_meta,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
        "artifacts": {
            "randomness_plot": str(outputs_root / "00_dataset_randomness.png"),
            "sources_plot": str(outputs_root / "01_clean_sources.png"),
            "mixture_plot": str(outputs_root / "02_received_mixture.png"),
            "outputs_plot": str(outputs_root / "03_model_outputs.png"),
            "frequency_plot": str(outputs_root / "04_frequency_view.png"),
            "metadata_plot": str(outputs_root / "05_sample_metadata.png"),
        },
        "sample_metadata": sample_meta,
    }
    outputs_root.mkdir(parents=True, exist_ok=True)
    with open(outputs_root / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print("QPSK RF step experiment complete")
    print(json.dumps({"val_metrics": val_metrics, "test_metrics": test_metrics, "best_epoch": train_meta["best_epoch"]}, indent=4))
    print(f"Artifacts saved to {outputs_root}")


if __name__ == "__main__":
    main()

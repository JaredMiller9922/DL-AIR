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


class StructuredWaveDataset(Dataset):
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


def raised_cosine_envelope(length, start, stop):
    env = np.zeros(length, dtype=np.float32)
    span = max(stop - start, 8)
    ramp = max(span // 8, 4)
    sustain_start = min(length, start + ramp)
    sustain_stop = max(sustain_start, stop - ramp)

    if start < sustain_start:
        up = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, sustain_start - start, endpoint=False))
        env[start:sustain_start] = up
    env[sustain_start:sustain_stop] = 1.0
    if sustain_stop < stop:
        down = 0.5 + 0.5 * np.cos(np.linspace(0.0, math.pi, stop - sustain_stop, endpoint=False))
        env[sustain_stop:stop] = down
    return env


FIXED_H_A = np.array([1.0 + 0.0j, 0.85 + 0.30j, 1.10 - 0.25j, 0.75 + 0.55j], dtype=np.complex64)
FIXED_H_B = np.array([0.40 + 0.90j, -0.25 + 0.95j, 0.65 + 0.55j, -0.70 + 0.35j], dtype=np.complex64)


def generate_sample(length, rng):
    t = np.arange(length, dtype=np.float32) / length

    freq_a = rng.uniform(5.2, 5.8)
    freq_b = rng.uniform(11.2, 11.8)
    phase_a = rng.uniform(0.0, 2.0 * math.pi)
    phase_b = rng.uniform(0.0, 2.0 * math.pi)
    amp_a = rng.uniform(0.85, 1.15)
    amp_b = rng.uniform(0.70, 1.05)
    chirp_a = rng.uniform(-0.08, 0.08)
    chirp_b = rng.uniform(-0.10, 0.10)

    start_a = int(rng.integers(0, length // 10))
    stop_a = int(rng.integers(8 * length // 10, length))
    start_b = int(rng.integers(length // 12, 2 * length // 10))
    stop_b = int(rng.integers(7 * length // 10, length - length // 20))

    env_a = raised_cosine_envelope(length, start_a, stop_a)
    env_b = raised_cosine_envelope(length, start_b, stop_b)

    phase_track_a = 2.0 * math.pi * (freq_a * t + 0.5 * chirp_a * t * t) + phase_a
    phase_track_b = 2.0 * math.pi * (freq_b * t + 0.5 * chirp_b * t * t) + phase_b
    source_a = amp_a * env_a * np.exp(1j * phase_track_a)
    source_b = amp_b * env_b * np.exp(1j * phase_track_b)

    h_a = FIXED_H_A
    h_b = FIXED_H_B

    mixture = np.zeros((4, length), dtype=np.complex64)
    for rx in range(4):
        mixture[rx] = h_a[rx] * source_a + h_b[rx] * source_b

    noise_std = rng.uniform(0.001, 0.004)
    noise = noise_std * (rng.standard_normal((4, length)) + 1j * rng.standard_normal((4, length)))
    mixture = mixture + noise.astype(np.complex64)

    x = complex_matrix_to_iq_channels(mixture)
    y = stacked_sources_to_iq(source_a.astype(np.complex64), source_b.astype(np.complex64))
    y_alt = stacked_sources_to_iq(source_b.astype(np.complex64), source_a.astype(np.complex64))

    metadata = {
        "freq_a": float(freq_a),
        "freq_b": float(freq_b),
        "amp_a": float(amp_a),
        "amp_b": float(amp_b),
        "chirp_a": float(chirp_a),
        "chirp_b": float(chirp_b),
        "noise_std": float(noise_std),
        "start_a": int(start_a),
        "stop_a": int(stop_a),
        "start_b": int(start_b),
        "stop_b": int(stop_b),
        "h_a_real": np.real(h_a).astype(float).tolist(),
        "h_a_imag": np.imag(h_a).astype(float).tolist(),
        "h_b_real": np.real(h_b).astype(float).tolist(),
        "h_b_imag": np.imag(h_b).astype(float).tolist(),
    }
    return x, y, y_alt, source_a, source_b, mixture, metadata


def save_dataset(root_dir, train_size=768, val_size=192, test_size=192, length=256, seed=17):
    root = Path(root_dir)
    if root.exists():
        shutil.rmtree(root)
    for split in ["train", "val", "test"]:
        (root / split).mkdir(parents=True, exist_ok=True)

    split_sizes = {"train": train_size, "val": val_size, "test": test_size}
    all_metadata = []
    for split_idx, (split_name, split_size) in enumerate(split_sizes.items()):
        rng = np.random.default_rng(seed + split_idx)
        split_meta = []
        for idx in range(split_size):
            x, y, y_alt, _, _, _, meta = generate_sample(length, rng)
            np.savez_compressed(root / split_name / f"sample_{idx:05d}.npz", x=x, y=y, y_alt=y_alt)
            split_meta.append(meta)
            all_metadata.append(meta)
        with open(root / f"{split_name}_metadata.json", "w") as handle:
            json.dump(split_meta, handle, indent=2)

    manifest = {
        "train": train_size,
        "val": val_size,
        "test": test_size,
        "length": length,
        "description": "Two clean complex narrowband sources with random phase, amplitude, chirp, complex receiver gains, and light noise.",
    }
    with open(root / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    return all_metadata


def make_loader(split_dir, batch_size, shuffle):
    dataset = StructuredWaveDataset(split_dir)
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


def plot_metadata_distributions(metadata, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    freq_a = [item["freq_a"] for item in metadata]
    freq_b = [item["freq_b"] for item in metadata]
    noise_std = [item["noise_std"] for item in metadata]
    amp_a = [item["amp_a"] for item in metadata]
    amp_b = [item["amp_b"] for item in metadata]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(freq_a, bins=24, alpha=0.8, label="freq_a")
    axes[0, 0].hist(freq_b, bins=24, alpha=0.8, label="freq_b")
    axes[0, 0].set_title("Source Frequency Ranges")
    axes[0, 0].legend()

    axes[0, 1].hist(noise_std, bins=24, color="tab:red")
    axes[0, 1].set_title("Noise Std Distribution")

    axes[1, 0].hist(amp_a, bins=24, alpha=0.8, label="amp_a")
    axes[1, 0].hist(amp_b, bins=24, alpha=0.8, label="amp_b")
    axes[1, 0].set_title("Amplitude Distribution")
    axes[1, 0].legend()

    axes[1, 1].scatter(freq_a[:300], freq_b[:300], s=10, alpha=0.6)
    axes[1, 1].set_title("Frequency Pair Scatter")
    axes[1, 1].set_xlabel("freq_a")
    axes[1, 1].set_ylabel("freq_b")

    fig.tight_layout()
    fig.savefig(output_dir / "00_dataset_randomness.png")
    plt.close(fig)


def plot_sample_results(output_dir, source_a, source_b, mixture, pred_aligned, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_a = pred_aligned[0] + 1j * pred_aligned[1]
    pred_b = pred_aligned[2] + 1j * pred_aligned[3]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(source_a.real, label="Source A real")
    axes[0].plot(source_b.real, label="Source B real")
    axes[0].set_title("Clean Ground-Truth Source Waves")
    axes[0].legend()
    axes[1].plot(source_a.imag, label="Source A imag")
    axes[1].plot(source_b.imag, label="Source B imag")
    axes[1].set_xlabel("Sample")
    axes[1].set_title("Ground-Truth Imaginary Components")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "01_clean_sources.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(mixture[0].real, label="RX0 real", color="black")
    axes[0].plot(mixture[1].real, label="RX1 real", alpha=0.75)
    axes[0].set_title("Received Mixture Signals (Real Parts)")
    axes[0].legend()
    axes[1].plot(mixture[0].imag, label="RX0 imag", color="black")
    axes[1].plot(mixture[1].imag, label="RX1 imag", alpha=0.75)
    axes[1].set_title("Received Mixture Signals (Imag Parts)")
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
    axes[1].set_xlabel("Normalized Frequency Bin")
    axes[1].set_title("Separated Output Spectra")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "04_frequency_view.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    lines = [
        f"freq_a={metadata['freq_a']:.3f}, freq_b={metadata['freq_b']:.3f}",
        f"amp_a={metadata['amp_a']:.3f}, amp_b={metadata['amp_b']:.3f}",
        f"chirp_a={metadata['chirp_a']:.3f}, chirp_b={metadata['chirp_b']:.3f}",
        f"noise_std={metadata['noise_std']:.4f}",
        f"window_a=[{metadata['start_a']}, {metadata['stop_a']}], window_b=[{metadata['start_b']}, {metadata['stop_b']}]",
        f"h_a_real={np.round(metadata['h_a_real'], 3)}",
        f"h_a_imag={np.round(metadata['h_a_imag'], 3)}",
        f"h_b_real={np.round(metadata['h_b_real'], 3)}",
        f"h_b_imag={np.round(metadata['h_b_imag'], 3)}",
    ]
    ax.axis("off")
    ax.text(0.01, 0.98, "Per-sample randomness used in the received mixture\n\n" + "\n".join(lines), va="top", family="monospace")
    fig.tight_layout()
    fig.savefig(output_dir / "05_sample_metadata.png")
    plt.close(fig)


def main():
    seed_everything(17)
    experiment_root = Path(__file__).resolve().parent
    data_root = experiment_root / "data"
    outputs_root = experiment_root / "outputs"
    checkpoint_path = outputs_root / "structured_rf_bridge_best.pt"

    metadata = save_dataset(data_root, train_size=768, val_size=192, test_size=192, length=256, seed=17)
    plot_metadata_distributions(metadata, outputs_root)

    train_loader = make_loader(data_root / "train", batch_size=32, shuffle=True)
    val_loader = make_loader(data_root / "val", batch_size=32, shuffle=False)
    test_loader = make_loader(data_root / "test", batch_size=32, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = IQCNNSeparator(in_ch=8, out_ch=4, base_channels=32, dropout=0.0).to(device)
    model, train_hist, val_hist, train_meta = train_model(
        model,
        train_loader,
        val_loader,
        epochs=60,
        device=device,
        lr=8e-4,
        weight_decay=0.0,
        grad_clip=1.0,
        checkpoint_path=str(checkpoint_path),
        normalize_inputs=False,
    )

    val_metrics = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)

    sample_rng = np.random.default_rng(777)
    x_np, y_np, y_alt_np, source_a, source_b, mixture, sample_meta = generate_sample(256, sample_rng)
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)
    y = torch.from_numpy(y_np).unsqueeze(0).to(device)
    y_alt = torch.from_numpy(y_alt_np).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)
        aligned = align_to_pit_target(pred, y, y_alt).cpu().numpy()[0]

    plot_sample_results(outputs_root, source_a, source_b, mixture, aligned, sample_meta)

    summary = {
        "device": device,
        "description": "Bridge experiment: clean complex narrowband sources, random per-sample phase/amplitude/chirp, random complex receiver gains, and light additive noise.",
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
        json.dump(summary, handle, indent=4)

    print("Structured RF bridge experiment complete")
    print(json.dumps({"val_metrics": val_metrics, "test_metrics": test_metrics, "best_epoch": train_meta["best_epoch"]}, indent=4))
    print(f"Artifacts saved to {outputs_root}")


if __name__ == "__main__":
    main()

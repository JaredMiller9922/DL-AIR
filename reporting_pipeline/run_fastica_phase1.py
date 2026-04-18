import itertools
import json
from dataclasses import asdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from torch.utils.data import DataLoader
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ExperimentConfig
from utils.data_utils.dataset import SavedRFDataset
from utils.data_utils.generator import MixtureConfig, NoiseConfig, QPSKConfig, RFMixtureGenerator
from utils.model_utils.conversion_helpers import complex_matrix_to_iq_channels, stacked_sources_to_iq
from utils.model_utils.losses import calculate_sdr, mse_loss
from utils.model_utils.symbol_utils import recover_symbols_from_waveform, rrc_taps, symbol_accuracy


def build_qpsk_cfg(cfg: ExperimentConfig) -> QPSKConfig:
    return QPSKConfig(
        n_symbols=cfg.n_symbols,
        samples_per_symbol=cfg.samples_per_symbol,
        rolloff=cfg.rolloff,
        rrc_span_symbols=cfg.rrc_span_symbols,
        normalize_power=cfg.normalize_power,
        num_channels=cfg.num_channels,
    )


def build_noise_cfg(cfg: ExperimentConfig) -> NoiseConfig:
    return NoiseConfig(enabled=cfg.noise_enabled)


def build_mix_cfg(cfg: ExperimentConfig) -> MixtureConfig:
    return MixtureConfig(
        alpha=cfg.alpha,
        snr_db=cfg.snr_db,
        n_rx=cfg.n_rx,
        random_phase=cfg.random_phase,
    )


def create_dataset(cfg: ExperimentConfig, root_dir: Path, seed: int, train_size: int = 256, val_size: int = 64, test_size: int = 256):
    if root_dir.exists():
        for child in root_dir.iterdir():
            if child.is_dir():
                for item in child.glob("*.npz"):
                    item.unlink()
            else:
                child.unlink()
    root_dir.mkdir(parents=True, exist_ok=True)

    generator = RFMixtureGenerator(seed=seed)
    qpsk_soi = build_qpsk_cfg(cfg)
    qpsk_int = build_qpsk_cfg(cfg)
    noise_cfg = build_noise_cfg(cfg)
    mix_cfg = build_mix_cfg(cfg)

    split_sizes = {"train": train_size, "val": val_size, "test": test_size}
    for split_idx, (split, count) in enumerate(split_sizes.items()):
        split_dir = root_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_gen = RFMixtureGenerator(seed=seed + split_idx)
        for idx in range(count):
            ex = split_gen.generate_mixture(qpsk_soi, qpsk_int, noise_cfg, mix_cfg)
            mixture = ex["mixture"]
            if mixture.ndim == 1:
                mixture = mixture[np.newaxis, :]

            x = complex_matrix_to_iq_channels(mixture)
            y = stacked_sources_to_iq(ex["source_a"], ex["source_b"])
            np.savez_compressed(
                split_dir / f"sample_{idx:06d}.npz",
                x=x.astype(np.float32),
                y=y.astype(np.float32),
                symbols_a=np.stack([ex["symbols_a"].real, ex["symbols_a"].imag], axis=0).astype(np.float32),
                symbols_b=np.stack([ex["symbols_b"].real, ex["symbols_b"].imag], axis=0).astype(np.float32),
            )


def load_split(split_dir: Path, batch_size: int) -> DataLoader:
    ds = SavedRFDataset(str(split_dir))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def run_fastica_on_sample(sample_x: np.ndarray) -> np.ndarray:
    X = sample_x.T
    n_components = min(X.shape[1], 4)
    ica = FastICA(
        n_components=n_components,
        algorithm="parallel",
        whiten="unit-variance",
        fun="logcosh",
        random_state=0,
        max_iter=1000,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        S = ica.fit_transform(X).T.astype(np.float32)

    if S.shape[0] < 4:
        padded = np.zeros((4, S.shape[1]), dtype=np.float32)
        padded[: S.shape[0]] = S
        S = padded
    return S


def best_aligned_prediction(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    best_loss = float("inf")
    best_pred = pred.copy()

    for perm in itertools.permutations(range(4)):
        permuted = pred[list(perm)]
        for signs in itertools.product([-1.0, 1.0], repeat=4):
            signed = permuted * np.array(signs, dtype=np.float32)[:, None]
            scaled = np.zeros_like(signed)
            total_loss = 0.0
            for ch in range(4):
                denom = float(np.dot(signed[ch], signed[ch]) + 1e-12)
                scale = float(np.dot(signed[ch], target[ch]) / denom)
                scaled[ch] = scale * signed[ch]
                total_loss += float(np.mean((scaled[ch] - target[ch]) ** 2))
            if total_loss < best_loss:
                best_loss = total_loss
                best_pred = scaled
    return best_pred


def evaluate_fastica(loader: DataLoader, sps: int, rolloff: float, span: int):
    taps = rrc_taps(sps=sps, beta=rolloff, span_symbols=span)
    total_wave_mse = 0.0
    total_sdr = 0.0
    total_symbol_acc_a = 0.0
    total_symbol_acc_b = 0.0
    total_examples = 0
    example = None

    for batch in loader:
        x = batch["x"].numpy()
        y = batch["y"].numpy()
        syms_a = batch["symbols_a"].numpy()
        syms_b = batch["symbols_b"].numpy()
        for i in range(x.shape[0]):
            pred_raw = run_fastica_on_sample(x[i])
            pred = best_aligned_prediction(pred_raw, y[i])

            pred_t = torch.from_numpy(pred).unsqueeze(0)
            y_t = torch.from_numpy(y[i]).unsqueeze(0)
            total_wave_mse += float(mse_loss(pred_t, y_t).item())
            total_sdr += float(calculate_sdr(pred_t, y_t).item())

            pred_a = pred[0] + 1j * pred[1]
            pred_b = pred[2] + 1j * pred[3]
            true_syms_a = syms_a[i][0] + 1j * syms_a[i][1]
            true_syms_b = syms_b[i][0] + 1j * syms_b[i][1]
            rec_a = recover_symbols_from_waveform(pred_a, taps, sps, len(true_syms_a))
            rec_b = recover_symbols_from_waveform(pred_b, taps, sps, len(true_syms_b))
            total_symbol_acc_a += float(symbol_accuracy(rec_a[: len(true_syms_a)], true_syms_a))
            total_symbol_acc_b += float(symbol_accuracy(rec_b[: len(true_syms_b)], true_syms_b))
            total_examples += 1

            if example is None:
                example = {
                    "x": x[i],
                    "y": y[i],
                    "pred": pred,
                }

    return {
        "wave_mse": total_wave_mse / max(1, total_examples),
        "sdr_db": total_sdr / max(1, total_examples),
        "soi_symbol_accuracy": total_symbol_acc_a / max(1, total_examples),
        "int_symbol_accuracy": total_symbol_acc_b / max(1, total_examples),
        "example": example,
    }


def plot_example(example: dict, out_path: Path, title: str):
    x = example["x"]
    y = example["y"]
    pred = example["pred"]
    T = min(200, x.shape[-1])
    t = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, y[0, :T], label="SOI I")
    axes[0].plot(t, y[2, :T], label="INT I")
    axes[0].set_title(f"{title}: Clean Sources")
    axes[0].legend()

    for ch in range(0, x.shape[0], 2):
        axes[1].plot(t, x[ch, :T], label=f"RX{ch//2} I")
    axes[1].set_title(f"{title}: Received Mixture Channels")
    axes[1].legend()

    axes[2].plot(t, y[0, :T], color="black", alpha=0.6, label="True SOI I")
    axes[2].plot(t, pred[0, :T], linestyle="--", label="Pred SOI I")
    axes[2].plot(t, y[2, :T], color="tab:orange", alpha=0.6, label="True INT I")
    axes[2].plot(t, pred[2, :T], linestyle="--", color="tab:red", label="Pred INT I")
    axes[2].set_title(f"{title}: FastICA Outputs")
    axes[2].legend(ncol=2, fontsize=8)
    axes[2].set_xlabel("Sample")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_comparison(summary: dict, out_path: Path):
    labels = ["1-channel", "2-channel"]
    wave_mse = [summary["single_channel"]["wave_mse"], summary["two_channel"]["wave_mse"]]
    soi_acc = [summary["single_channel"]["soi_symbol_accuracy"], summary["two_channel"]["soi_symbol_accuracy"]]
    int_acc = [summary["single_channel"]["int_symbol_accuracy"], summary["two_channel"]["int_symbol_accuracy"]]
    sdr = [summary["single_channel"]["sdr_db"], summary["two_channel"]["sdr_db"]]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].bar(labels, wave_mse, color=["#c96f6f", "#6fa8dc"])
    axes[0, 0].set_title("Wave MSE")
    axes[0, 1].bar(labels, sdr, color=["#c96f6f", "#6fa8dc"])
    axes[0, 1].set_title("SDR (dB)")
    axes[1, 0].bar(labels, soi_acc, color=["#c96f6f", "#6fa8dc"])
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_title("SOI Symbol Accuracy")
    axes[1, 1].bar(labels, int_acc, color=["#c96f6f", "#6fa8dc"])
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Interference Symbol Accuracy")
    for ax in axes.flat:
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("FastICA Phase 1: Single vs Two Channel Validation", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_phase1_config(n_rx: int, dataset_path: str) -> ExperimentConfig:
    return ExperimentConfig(
        mode="phase1_fastica",
        model_name="FastICA",
        dataset_path=dataset_path,
        generate_new_data=True,
        use_on_the_fly_data=False,
        batch_size=32,
        epochs=20,
        lr=1e-3,
        n_symbols=400,
        samples_per_symbol=2,
        rolloff=0.25,
        rrc_span_symbols=12,
        normalize_power=True,
        noise_enabled=True,
        alpha=1.0,
        snr_db=15.0,
        n_rx=n_rx,
        random_phase=True,
    )


def main():
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / "reporting_pipeline" / "outputs" / "phase1_fastica"
    output_root.mkdir(parents=True, exist_ok=True)

    runs = [
        ("single_channel", build_phase1_config(1, str(output_root / "single_channel_data"))),
        ("two_channel", build_phase1_config(2, str(output_root / "two_channel_data"))),
    ]

    summary = {}
    for name, cfg in runs:
        dataset_root = Path(cfg.dataset_path)
        create_dataset(cfg, dataset_root, seed=23 if cfg.n_rx == 1 else 24)
        test_loader = load_split(dataset_root / "test", batch_size=cfg.batch_size)
        metrics = evaluate_fastica(
            test_loader,
            sps=cfg.samples_per_symbol,
            rolloff=cfg.rolloff,
            span=cfg.rrc_span_symbols,
        )
        plot_example(metrics.pop("example"), output_root / f"{name}_fastica_example.png", f"FastICA {name.replace('_', ' ').title()}")
        summary[name] = {
            "config": asdict(cfg),
            **metrics,
        }

    with open(output_root / "phase1_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    plot_comparison(summary, output_root / "fastica_phase1_comparison.png")

    print(json.dumps(summary, indent=2))
    print(f"Saved Phase 1 FastICA outputs to {output_root}")


if __name__ == "__main__":
    main()

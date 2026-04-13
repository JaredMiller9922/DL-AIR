import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.model_utils.losses import align_to_pit_target, calculate_sdr, mse_loss, pit_mse_loss
from utils.model_utils.symbol_utils import QPSK_POINTS, recover_symbols_from_waveform


QPSK_LABELS = {
    0: "(1 + 1j)/sqrt(2)",
    1: "(-1 + 1j)/sqrt(2)",
    2: "(-1 - 1j)/sqrt(2)",
    3: "(1 - 1j)/sqrt(2)",
}


def qpsk_index(samples: np.ndarray) -> np.ndarray:
    dists = np.abs(samples[:, None] - QPSK_POINTS[None, :]) ** 2
    return np.argmin(dists, axis=1).astype(np.int64)


def best_qpsk_label_alignment(pred_labels: np.ndarray, true_labels: np.ndarray):
    best_acc = -1.0
    best_labels = pred_labels
    for rot in range(4):
        rotated = (pred_labels + rot) % 4
        acc = np.mean(rotated == true_labels[: len(rotated)]) if len(rotated) else 0.0
        if acc > best_acc:
            best_acc = acc
            best_labels = rotated
    return best_labels, float(best_acc)


def phase_correct_qpsk(samples: np.ndarray) -> np.ndarray:
    phase_est = np.angle(np.mean(samples ** 4) + 1e-12) / 4.0
    corrected = samples * np.exp(-1j * phase_est)
    power = np.mean(np.abs(corrected) ** 2) + 1e-12
    return corrected / np.sqrt(power)


def remove_carrier(wave: np.ndarray, center_bin: float, phase: float = 0.0) -> np.ndarray:
    n = np.arange(len(wave), dtype=np.float32)
    mixed = wave * np.exp(-1j * (2.0 * np.pi * center_bin * n / len(wave) + phase))
    return mixed


def recover_qpsk_labels_from_waveform(wave: np.ndarray, rrc_taps: np.ndarray, sps: int, n_symbols: int, start: int = 0, center_bin: float = 0.0, phase: float = 0.0) -> np.ndarray:
    baseband = remove_carrier(wave, center_bin=center_bin, phase=phase)
    sample_offset = start + max(0, sps // 2)
    sym_samples = baseband[sample_offset::sps][:n_symbols]
    return qpsk_index(sym_samples)


def evaluate_loader(model, loader, device, rrc_taps, sps):
    model = model.to(device) if hasattr(model, "to") else model
    if hasattr(model, "eval"):
        model.eval()

    total_pit_mse = 0.0
    total_mse = 0.0
    total_sdr = 0.0
    soi_correct = 0
    soi_total = 0
    int_correct = 0
    int_total = 0
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

            pred_np = aligned.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            y_alt_np = y_alt.detach().cpu().numpy()
            labels_a = batch["labels_a"].cpu().numpy()
            labels_b = batch["labels_b"].cpu().numpy()
            starts_a = batch["start_a"].cpu().numpy()
            starts_b = batch["start_b"].cpu().numpy()
            phase_a = batch["phase_a"].cpu().numpy()
            phase_b = batch["phase_b"].cpu().numpy()
            center_bin_a = batch["center_bin_a"].cpu().numpy()
            center_bin_b = batch["center_bin_b"].cpu().numpy()

            for i in range(pred_np.shape[0]):
                use_alt = np.mean((pred_np[i] - y_alt_np[i]) ** 2) < np.mean((pred_np[i] - y_np[i]) ** 2)
                true_labels_a = labels_b[i] if use_alt else labels_a[i]
                true_labels_b = labels_a[i] if use_alt else labels_b[i]
                true_start_a = starts_b[i] if use_alt else starts_a[i]
                true_start_b = starts_a[i] if use_alt else starts_b[i]
                true_phase_a = phase_b[i] if use_alt else phase_a[i]
                true_phase_b = phase_a[i] if use_alt else phase_b[i]
                true_center_bin_a = center_bin_b[i] if use_alt else center_bin_a[i]
                true_center_bin_b = center_bin_a[i] if use_alt else center_bin_b[i]

                wave_a = pred_np[i, 0] + 1j * pred_np[i, 1]
                wave_b = pred_np[i, 2] + 1j * pred_np[i, 3]
                rec_a = recover_qpsk_labels_from_waveform(wave_a, rrc_taps, sps, len(true_labels_a), int(true_start_a), float(true_center_bin_a), float(true_phase_a))
                rec_b = recover_qpsk_labels_from_waveform(wave_b, rrc_taps, sps, len(true_labels_b), int(true_start_b), float(true_center_bin_b), float(true_phase_b))

                n_a = min(len(rec_a), len(true_labels_a))
                n_b = min(len(rec_b), len(true_labels_b))
                _, acc_a = best_qpsk_label_alignment(rec_a[:n_a], true_labels_a[:n_a])
                _, acc_b = best_qpsk_label_alignment(rec_b[:n_b], true_labels_b[:n_b])
                soi_correct += int(round(acc_a * n_a))
                soi_total += n_a
                int_correct += int(round(acc_b * n_b))
                int_total += n_b

            batches += 1

    soi_acc = soi_correct / max(1, soi_total)
    int_acc = int_correct / max(1, int_total)
    avg_acc = 0.5 * (soi_acc + int_acc)
    return {
        "pit_mse": total_pit_mse / max(1, batches),
        "wave_mse": total_mse / max(1, batches),
        "sdr_db": total_sdr / max(1, batches),
        "soi_symbol_accuracy": soi_acc,
        "int_symbol_accuracy": int_acc,
        "avg_symbol_accuracy": avg_acc,
        "soi_symbol_error_rate": 1.0 - soi_acc,
        "int_symbol_error_rate": 1.0 - int_acc,
    }


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def save_csv(path: Path, rows, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_curves(path: Path, sweep_values, model_to_metrics, x_label, metric_key, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    for model_name, metrics in model_to_metrics.items():
        plt.plot(sweep_values, [m[metric_key] for m in metrics], marker="o", label=model_name)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(metric_key.replace("_", " ").title())
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_model_bar(path: Path, rows, metric_key: str, title: str, ylabel: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["model"] for row in rows]
    values = [row[metric_key] for row in rows]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_wave_error_vs_symbol_accuracy(path: Path, rows, title: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    for model_name in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model_name]
        plt.scatter(
            [row["wave_mse"] for row in subset],
            [row["soi_symbol_accuracy"] for row in subset],
            label=model_name,
            s=30,
        )
    plt.xscale("log")
    plt.xlabel("Wave MSE (log scale)")
    plt.ylabel("SOI Symbol Accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_qpsk_alphabet(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(QPSK_POINTS.real, QPSK_POINTS.imag, s=100)
    for idx, point in enumerate(QPSK_POINTS):
        plt.text(point.real + 0.05, point.imag + 0.05, f"{idx}")
    plt.axhline(0.0, color="gray", linewidth=0.8)
    plt.axvline(0.0, color="gray", linewidth=0.8)
    plt.title("QPSK Symbol Alphabet")
    plt.xlabel("In-Phase")
    plt.ylabel("Quadrature")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_example_signals(path: Path, x, y, pred, title_prefix: str, num_samples: int = 200):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x[:, :num_samples]
    y = y[:, :num_samples]
    pred = pred[:, :num_samples]
    t = np.arange(num_samples)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, y[0], label="SOI I")
    axes[0].plot(t, y[2], label="Interference I")
    axes[0].set_title(f"{title_prefix}: Clean Sources")
    axes[0].legend()

    axes[1].plot(t, x[0], label="Mixture I", color="black")
    axes[1].plot(t, x[1], label="Mixture Q", color="gray", linestyle="--")
    axes[1].set_title(f"{title_prefix}: Received Mixture")
    axes[1].legend()

    axes[2].plot(t, y[0], label="True SOI I", color="black", alpha=0.6)
    axes[2].plot(t, pred[0], label="Pred SOI I", linestyle="--")
    axes[2].plot(t, y[2], label="True Int I", color="tab:orange", alpha=0.6)
    axes[2].plot(t, pred[2], label="Pred Int I", linestyle="--", color="tab:red")
    axes[2].set_title(f"{title_prefix}: Separated Outputs")
    axes[2].legend(ncol=2, fontsize=8)
    axes[2].set_xlabel("Sample")

    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

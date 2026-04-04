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


def phase_correct_qpsk(samples: np.ndarray) -> np.ndarray:
    phase_est = np.angle(np.mean(samples ** 4) + 1e-12) / 4.0
    corrected = samples * np.exp(-1j * phase_est)
    power = np.mean(np.abs(corrected) ** 2) + 1e-12
    return corrected / np.sqrt(power)


def recover_qpsk_labels_from_waveform(wave: np.ndarray, rrc_taps: np.ndarray, sps: int, n_symbols: int) -> np.ndarray:
    rec = recover_symbols_from_waveform(wave, rrc_taps, sps, n_symbols)
    rec = phase_correct_qpsk(rec)
    return qpsk_index(rec)


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
            labels_a = batch["labels_a"].cpu().numpy()
            labels_b = batch["labels_b"].cpu().numpy()

            for i in range(pred_np.shape[0]):
                wave_a = pred_np[i, 0] + 1j * pred_np[i, 1]
                wave_b = pred_np[i, 2] + 1j * pred_np[i, 3]
                rec_a = recover_qpsk_labels_from_waveform(wave_a, rrc_taps, sps, len(labels_a[i]))
                rec_b = recover_qpsk_labels_from_waveform(wave_b, rrc_taps, sps, len(labels_b[i]))

                n_a = min(len(rec_a), len(labels_a[i]))
                n_b = min(len(rec_b), len(labels_b[i]))
                soi_correct += int(np.sum(rec_a[:n_a] == labels_a[i][:n_a]))
                soi_total += n_a
                int_correct += int(np.sum(rec_b[:n_b] == labels_b[i][:n_b]))
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

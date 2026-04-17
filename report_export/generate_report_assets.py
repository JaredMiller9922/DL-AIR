import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FINAL_REPORT = ROOT / "Final Report"
FIGURES = FINAL_REPORT / "figures"
EXPORT = ROOT / "report_export" / "overleaf_ready"


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def read_json(path: Path):
    return json.loads(path.read_text())


def create_single_channel_reference_bar() -> None:
    logs_root = ROOT / "logs"
    models = [
        ("Hybrid", "Hybrid_results.json"),
        ("LSTM", "LSTM_results.json"),
        ("IQ-CNN", "IQ_CNN_results.json"),
        ("HTDemucs", "HTDemucs_results.json"),
    ]
    names = []
    values = []
    for label, file_name in models:
        data = read_json(logs_root / file_name)
        names.append(label)
        values.append(data["final_mse"])

    plt.figure(figsize=(7.5, 4.5))
    bars = plt.bar(names, values, color=["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"])
    plt.ylabel("Wave MSE")
    plt.title("Single-Channel Reference Wave Error Comparison")
    plt.yscale("log")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value * 1.15, f"{value:.2e}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "single_channel_reference_wave_comparison.png", dpi=300)
    plt.close()


def create_noise_progress_plots() -> None:
    logs_root = ROOT / "experiments" / "single_channel_threshold_run" / "noise_sweep" / "outputs" / "noise_sweep_single_rx" / "logs"
    variances = [0.00, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    tags = [f"var_{v:.4f}".replace(".", "p") for v in variances]
    models = ["Hybrid", "LSTM", "IQ_CNN"]
    labels = {"Hybrid": "Hybrid", "LSTM": "LSTM", "IQ_CNN": "IQ-CNN"}
    colors = {"Hybrid": "#2ca02c", "LSTM": "#1f77b4", "IQ_CNN": "#ff7f0e"}

    best_mse = {m: [] for m in models}
    sym_acc = {m: [] for m in models}
    sdr = {m: [] for m in models}
    for tag in tags:
        for model in models:
            data = read_json(logs_root / tag / f"{model}_results.json")
            best_mse[model].append(data["best_val_mse"])
            sym_acc[model].append(data["final_symbol_accuracy"])
            sdr[model].append(data["final_sdr_db"])

    plt.figure(figsize=(7.5, 4.5))
    for model in models:
        plt.plot(variances, best_mse[model], marker="o", label=labels[model], color=colors[model])
    plt.xlabel("Noise Variance")
    plt.ylabel("Best Validation MSE")
    plt.title("Single-Channel Noise Sweep Trend")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "noise_variance_vs_best_val_mse_current.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7.5, 4.5))
    for model in models:
        plt.plot(variances, sym_acc[model], marker="o", label=labels[model], color=colors[model])
    plt.xlabel("Noise Variance")
    plt.ylabel("Final Symbol Accuracy")
    plt.title("Single-Channel Noise Sweep Symbol Trend")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "noise_variance_vs_symbol_accuracy_current.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7.5, 4.5))
    for model in models:
        plt.plot(variances, sdr[model], marker="o", label=labels[model], color=colors[model])
    plt.xlabel("Noise Variance")
    plt.ylabel("Final SDR (dB)")
    plt.title("Single-Channel Noise Sweep SDR Trend")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "noise_variance_vs_sdr_current.png", dpi=300)
    plt.close()


def copy_selected_figures() -> None:
    alpha_root = ROOT / "experiments" / "single_channel_threshold_run" / "alpha_sweep" / "outputs" / "alpha_sweep_single_rx"
    threshold_root = ROOT / "experiments" / "single_channel_threshold_run" / "outputs" / "single_rx_threshold_20260413_210903"
    noise_root = ROOT / "experiments" / "single_channel_threshold_run" / "noise_sweep" / "outputs" / "noise_sweep_single_rx"
    visual_root = ROOT / "visualizations"

    copies = {
        alpha_root / "plots" / "alpha_vs_relative_mse.png": FIGURES / "alpha_vs_relative_mse.png",
        alpha_root / "plots" / "alpha_vs_sdr_db.png": FIGURES / "alpha_vs_sdr_db.png",
        alpha_root / "plots" / "LSTM_knee_val_curves.png": FIGURES / "LSTM_knee_val_curves.png",
        alpha_root / "qualitative" / "alpha_0p80" / "IQ_CNN_alpha_0.80_separation.png": FIGURES / "IQ_CNN_alpha_0p80_separation.png",
        alpha_root / "qualitative" / "alpha_1p00" / "IQ_CNN_alpha_1.00_separation.png": FIGURES / "IQ_CNN_alpha_1p00_separation.png",
        threshold_root / "plots" / "htdemucs_train_val.png": FIGURES / "htdemucs_train_val.png",
        visual_root / "HTDemucs_separation.png": FIGURES / "HTDemucs_separation.png",
        visual_root / "HTDemucs_SourceA_symbol_recovery.png": FIGURES / "HTDemucs_SourceA_symbol_recovery.png",
        visual_root / "HTDemucs_SourceB_symbol_recovery.png": FIGURES / "HTDemucs_SourceB_symbol_recovery.png",
        noise_root / "plots" / "Hybrid_var_0p0800_separation.png": FIGURES / "Hybrid_var_0p0800_separation.png",
        noise_root / "plots" / "LSTM_var_0p0800_separation.png": FIGURES / "LSTM_var_0p0800_separation.png",
        noise_root / "plots" / "IQ_CNN_var_0p0800_separation.png": FIGURES / "IQ_CNN_var_0p0800_separation.png",
    }
    for src, dst in copies.items():
        if src.exists():
            copy(src, dst)


def export_overleaf_folder() -> None:
    if EXPORT.exists():
        shutil.rmtree(EXPORT)
    EXPORT.mkdir(parents=True, exist_ok=True)

    copy(FINAL_REPORT / "main.tex", EXPORT / "main.tex")
    copy(FINAL_REPORT / "IEEEtran.cls", EXPORT / "IEEEtran.cls")
    copy(FINAL_REPORT / "IEEEtran.bst", EXPORT / "IEEEtran.bst")
    copy(FINAL_REPORT / "validation mse.png", EXPORT / "validation mse.png")
    if (FINAL_REPORT / "references").exists():
        shutil.copytree(FINAL_REPORT / "references", EXPORT / "references")
    if (FINAL_REPORT / "figures").exists():
        shutil.copytree(FINAL_REPORT / "figures", EXPORT / "figures")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    create_single_channel_reference_bar()
    create_noise_progress_plots()
    copy_selected_figures()
    export_overleaf_folder()


if __name__ == "__main__":
    main()

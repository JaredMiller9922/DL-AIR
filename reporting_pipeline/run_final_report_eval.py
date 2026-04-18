import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reporting_pipeline.config import make_default_config
from reporting_pipeline.data import ensure_sweep_eval_data, ensure_training_data, make_loader
from reporting_pipeline.metrics import (
    QPSK_LABELS,
    evaluate_loader,
    plot_example_signals,
    plot_metric_curves,
    plot_model_bar,
    plot_qpsk_alphabet,
    plot_wave_error_vs_symbol_accuracy,
    save_csv,
    save_json,
)
from reporting_pipeline.model_registry import active_specs, load_trained_model
from reporting_pipeline.training import train_model
from utils.model_utils.symbol_utils import rrc_taps


def ensure_checkpoint(spec, config, train_loader, val_loader, device):
    if spec.kind != "learned":
        return None

    checkpoint_path = config.checkpoints_dir / spec.checkpoint_name

    if checkpoint_path.exists() and not config.force_retrain:
        return checkpoint_path

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model = spec.builder().to(device)
    train_kwargs = dict(spec.train_kwargs or {})
    train_kwargs.setdefault("optimizer_name", "adamw")
    train_kwargs.setdefault("scheduler_name", "plateau")
    train_kwargs.setdefault("scheduler_patience", config.scheduler_patience)

    train_model(
        model,
        train_loader,
        val_loader,
        epochs=config.train_epochs,
        device=device,
        lr=1e-3,
        weight_decay=config.weight_decay,
        grad_clip=config.grad_clip,
        checkpoint_path=str(checkpoint_path),
        checkpoint_extra={"model": spec.name, "pipeline": "final_report"},
        early_stopping_patience=config.early_stopping_patience,
        use_amp=config.use_amp,
        **train_kwargs,
    )
    return checkpoint_path


def evaluate_specs(specs, dataset_dir_map, config, device, sweep_name, sweep_values):
    taps = rrc_taps(config.samples_per_symbol, config.rolloff, config.rrc_span_symbols)
    rows = []
    model_to_metrics = {spec.name: [] for spec in specs}

    train_dir, val_dir = ensure_training_data(config)
    train_loader = make_loader(train_dir, config.batch_size, shuffle=True)
    val_loader = make_loader(val_dir, config.batch_size, shuffle=False)

    for spec in specs:
        ensure_checkpoint(spec, config, train_loader, val_loader, device)
        model = load_trained_model(spec, config.checkpoints_dir, device)

        for sweep_value in sweep_values:
            eval_dir = dataset_dir_map[sweep_value]
            loader = make_loader(eval_dir, config.batch_size, shuffle=False)
            metrics = evaluate_loader(model, loader, device, taps, config.samples_per_symbol)
            metrics.update({"model": spec.name, sweep_name: sweep_value})
            rows.append(metrics)
            model_to_metrics[spec.name].append(metrics)

    return rows, model_to_metrics


def write_summary_tables(config, alpha_rows, snr_rows):
    common_headers = [
        "model",
        "pit_mse",
        "wave_mse",
        "sdr_db",
        "soi_symbol_accuracy",
        "int_symbol_accuracy",
        "avg_symbol_accuracy",
        "soi_symbol_error_rate",
        "int_symbol_error_rate",
    ]
    save_csv(config.tables_dir / "alpha_robustness.csv", alpha_rows, headers=["alpha", *common_headers])
    save_csv(config.tables_dir / "snr_robustness.csv", snr_rows, headers=["snr_db", *common_headers])

    best_alpha_rows = []
    for model_name in sorted({row["model"] for row in alpha_rows}):
        subset = [row for row in alpha_rows if row["model"] == model_name and row["alpha"] == 1.0]
        if subset:
            best_alpha_rows.append(subset[0])
    save_csv(config.tables_dir / "model_comparison_reference.csv", best_alpha_rows, headers=["alpha", *common_headers])


def write_failure_summary(config, alpha_rows, snr_rows):
    threshold = 0.95
    summary = {"threshold_symbol_accuracy": threshold, "alpha_failure": {}, "snr_failure": {}}
    for model_name in sorted({row["model"] for row in alpha_rows}):
        alpha_subset = sorted([row for row in alpha_rows if row["model"] == model_name], key=lambda x: x["alpha"])
        fail_alpha = next((row["alpha"] for row in alpha_subset if row["soi_symbol_accuracy"] < threshold), None)
        summary["alpha_failure"][model_name] = fail_alpha

        snr_subset = sorted([row for row in snr_rows if row["model"] == model_name], key=lambda x: x["snr_db"], reverse=True)
        fail_snr = next((row["snr_db"] for row in snr_subset if row["soi_symbol_accuracy"] < threshold), None)
        summary["snr_failure"][model_name] = fail_snr
    save_json(config.json_dir / "failure_thresholds.json", summary)


def write_symbol_alphabet(config):
    rows = [{"label": k, "constellation": v} for k, v in QPSK_LABELS.items()]
    save_json(config.json_dir / "qpsk_symbol_alphabet.json", rows)
    with open(config.tables_dir / "qpsk_symbol_alphabet.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "constellation"])
        writer.writeheader()
        writer.writerows(rows)


def save_latex_table(path: Path, caption: str, label: str, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    def esc(value: str) -> str:
        return value.replace("_", r"\_")
    with open(path, "w") as handle:
        cols = "|" + "c|" * len(headers)
        handle.write("\\begin{table}[t]\n\\centering\n")
        handle.write(f"\\caption{{{esc(caption)}}}\n")
        handle.write(f"\\label{{{label}}}\n")
        handle.write(f"\\begin{{tabular}}{{{cols}}}\n\\hline\n")
        handle.write(" & ".join(esc(header) for header in headers) + " \\\\ \n\\hline\n")
        for row in rows:
            handle.write(" & ".join(esc(cell) for cell in row) + " \\\\ \n")
        handle.write("\\hline\n\\end{tabular}\n\\end{table}\n")


def write_report_tables(config, alpha_rows, snr_rows):
    reference_rows = [row for row in alpha_rows if row["alpha"] == 1.0]
    reference_rows.sort(key=lambda row: row["avg_symbol_accuracy"], reverse=True)
    save_latex_table(
        config.tables_dir / "model_comparison_reference.tex",
        "Reference comparison on the fixed single-channel evaluation set.",
        "tab:model_comparison_reference",
        ["Model", "PIT-MSE", "Wave MSE", "SDR", "SOI Acc", "Int Acc"],
        [
            [
                row["model"],
                f"{row['pit_mse']:.4f}",
                f"{row['wave_mse']:.4f}",
                f"{row['sdr_db']:.2f}",
                f"{row['soi_symbol_accuracy']:.3f}",
                f"{row['int_symbol_accuracy']:.3f}",
            ]
            for row in reference_rows
        ],
    )

    alpha_summary = []
    for model_name in sorted({row["model"] for row in alpha_rows}):
        subset = [row for row in alpha_rows if row["model"] == model_name]
        best = max(subset, key=lambda row: row["alpha"] if row["soi_symbol_accuracy"] >= 0.95 else -1)
        alpha_summary.append([model_name, f"{best['alpha']:.2f}", f"{best['soi_symbol_accuracy']:.3f}", f"{best['wave_mse']:.4f}"])
    save_latex_table(
        config.tables_dir / "alpha_robustness_summary.tex",
        "Largest tested interference level before substantial SOI symbol degradation.",
        "tab:alpha_robustness",
        ["Model", "Alpha", "SOI Acc", "Wave MSE"],
        alpha_summary,
    )

    snr_summary = []
    for model_name in sorted({row["model"] for row in snr_rows}):
        subset = [row for row in snr_rows if row["model"] == model_name]
        kept = [row for row in subset if row["soi_symbol_accuracy"] >= 0.95]
        best = min(kept, key=lambda row: row["snr_db"]) if kept else min(subset, key=lambda row: row["snr_db"])
        snr_summary.append([model_name, f"{best['snr_db']:.1f}", f"{best['soi_symbol_accuracy']:.3f}", f"{best['wave_mse']:.4f}"])
    save_latex_table(
        config.tables_dir / "noise_robustness_summary.tex",
        "Lowest tested SNR before substantial SOI symbol degradation.",
        "tab:noise_robustness",
        ["Model", "SNR (dB)", "SOI Acc", "Wave MSE"],
        snr_summary,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a tiny smoke-test version of the final report evaluation")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config = make_default_config(project_root)
    if args.quick:
        config.train_examples = 128
        config.val_examples = 64
        config.eval_examples = 64
        config.train_epochs = 2
        config.early_stopping_patience = 2
        config.alpha_sweep = (0.5, 1.0)
        config.snr_sweep_db = (20.0, 10.0)
        config.active_models = ("FastICA", "Linear", "IQ_CNN")
    config.force_retrain = args.force_retrain

    device = config.device if config.device == "cpu" or torch.cuda.is_available() else "cpu"
    specs = active_specs(config.active_models, in_ch=config.in_ch)
    config.outputs_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    config.json_dir.mkdir(parents=True, exist_ok=True)
    if config.force_retrain and config.datasets_dir.exists():
        shutil.rmtree(config.datasets_dir)

    alpha_dirs, snr_dirs = ensure_sweep_eval_data(config)

    alpha_rows, alpha_curves = evaluate_specs(specs, alpha_dirs, config, device, "alpha", config.alpha_sweep)
    snr_rows, snr_curves = evaluate_specs(specs, snr_dirs, config, device, "snr_db", config.snr_sweep_db)

    write_summary_tables(config, alpha_rows, snr_rows)
    write_failure_summary(config, alpha_rows, snr_rows)
    write_symbol_alphabet(config)
    write_report_tables(config, alpha_rows, snr_rows)
    plot_qpsk_alphabet(config.figures_dir / "qpsk_symbol_alphabet.png")

    plot_metric_curves(config.figures_dir / "alpha_sweep_soi_symbol_accuracy.png", list(config.alpha_sweep), alpha_curves, "alpha", "soi_symbol_accuracy", "SOI Symbol Accuracy vs Interference Strength")
    plot_metric_curves(config.figures_dir / "alpha_sweep_interference_symbol_accuracy.png", list(config.alpha_sweep), alpha_curves, "alpha", "int_symbol_accuracy", "Interference Symbol Accuracy vs Interference Strength")
    plot_metric_curves(config.figures_dir / "alpha_sweep_wave_mse.png", list(config.alpha_sweep), alpha_curves, "alpha", "wave_mse", "Wave Recovery Error vs Interference Strength")
    plot_metric_curves(config.figures_dir / "noise_sweep_soi_symbol_accuracy.png", list(config.snr_sweep_db), snr_curves, "SNR (dB)", "soi_symbol_accuracy", "SOI Symbol Accuracy vs Noise Level")
    plot_metric_curves(config.figures_dir / "noise_sweep_interference_symbol_accuracy.png", list(config.snr_sweep_db), snr_curves, "SNR (dB)", "int_symbol_accuracy", "Interference Symbol Accuracy vs Noise Level")
    plot_metric_curves(config.figures_dir / "noise_sweep_wave_mse.png", list(config.snr_sweep_db), snr_curves, "SNR (dB)", "wave_mse", "Wave Recovery Error vs Noise Level")
    plot_model_bar(config.figures_dir / "model_comparison_reference_wave_mse.png", [row for row in alpha_rows if row["alpha"] == 1.0], "wave_mse", "Reference Wave Error Comparison", "Wave MSE")
    plot_model_bar(config.figures_dir / "model_comparison_reference_soi_accuracy.png", [row for row in alpha_rows if row["alpha"] == 1.0], "soi_symbol_accuracy", "Reference SOI Symbol Accuracy Comparison", "SOI Symbol Accuracy")
    plot_wave_error_vs_symbol_accuracy(config.figures_dir / "wave_error_vs_symbol_accuracy_alpha.png", alpha_rows, "Wave Error vs SOI Symbol Accuracy (Alpha Sweep)")
    plot_wave_error_vs_symbol_accuracy(config.figures_dir / "wave_error_vs_symbol_accuracy_noise.png", snr_rows, "Wave Error vs SOI Symbol Accuracy (Noise Sweep)")

    reference_rows = [row for row in alpha_rows if row["alpha"] == 1.0 and row["model"] != "FastICA"]
    best_ref = max(reference_rows, key=lambda row: row["avg_symbol_accuracy"])
    best_spec = next(spec for spec in specs if spec.name == best_ref["model"])
    best_model = load_trained_model(best_spec, config.checkpoints_dir, device)
    reference_loader = make_loader(alpha_dirs[1.0], batch_size=1, shuffle=False)
    sample = next(iter(reference_loader))
    with torch.no_grad():
        pred = best_model(sample["x"].to(device)).cpu().numpy()[0]
    x_np = sample["x"][0].numpy()
    y_np = sample["y"][0].numpy()
    plot_example_signals(config.figures_dir / "pipeline_example_best_model.png", x_np, y_np, pred, title_prefix=best_ref["model"])

    fastica_spec = next(spec for spec in specs if spec.name == "FastICA")
    fastica_model = load_trained_model(fastica_spec, config.checkpoints_dir, device)
    with torch.no_grad():
        fastica_pred = fastica_model(sample["x"].to(device)).cpu().numpy()[0]
    plot_example_signals(config.figures_dir / "pipeline_example_fastica.png", x_np, y_np, fastica_pred, title_prefix="FastICA")

    summary = {
        "device": device,
        "active_models": list(config.active_models),
        "alpha_sweep": list(config.alpha_sweep),
        "snr_sweep_db": list(config.snr_sweep_db),
        "learned_models_use_trained_checkpoints": True,
        "learned_models_train_if_checkpoint_missing": True,
    }
    save_json(config.json_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Final report evaluation artifacts saved to {config.outputs_dir}")


if __name__ == "__main__":
    main()

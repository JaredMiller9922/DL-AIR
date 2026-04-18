import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reporting_pipeline.baselines import FastICABaseline
from reporting_pipeline.metrics import plot_example_signals, plot_model_bar, save_json
from reporting_pipeline.model_registry import active_specs, load_trained_model
from reporting_pipeline.repo_data import (
    estimate_input_channels,
    generate_fixed_splits,
    make_loader,
    make_phase2_config,
    symbols_to_complex,
)
from reporting_pipeline.training import train_model
from utils.model_utils.losses import align_to_pit_target, calculate_sdr, mse_loss, pit_mse_loss
from utils.model_utils.symbol_utils import recover_symbols_from_waveform, rrc_taps, symbol_accuracy


def ensure_checkpoint(spec, cfg, train_loader, val_loader, checkpoint_dir: Path, device: str):
    if spec.kind != "learned":
        return None

    ckpt = checkpoint_dir / spec.checkpoint_name
    if ckpt.exists():
        return ckpt

    ckpt.parent.mkdir(parents=True, exist_ok=True)
    model = spec.builder().to(device)
    train_kwargs = dict(spec.train_kwargs or {})
    train_model(
        model,
        train_loader,
        val_loader,
        epochs=cfg.epochs,
        device=device,
        lr=cfg.lr,
        weight_decay=1e-4,
        grad_clip=1.0,
        checkpoint_path=str(ckpt),
        checkpoint_extra={"phase": "phase2_two_channel_reference", "model": spec.name},
        early_stopping_patience=6,
        use_amp=True,
        **train_kwargs,
    )
    return ckpt


def evaluate_model_on_loader(model, loader, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    taps = rrc_taps(cfg.samples_per_symbol, cfg.rolloff, cfg.rrc_span_symbols)
    total_pit_mse = 0.0
    total_wave_mse = 0.0
    total_sdr = 0.0
    total_soi_acc = 0.0
    total_int_acc = 0.0
    count = 0
    example = None

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_alt = batch["y_alt"].to(device)
            pred = model(x)
            aligned = align_to_pit_target(pred, y, y_alt)

            total_pit_mse += pit_mse_loss(pred, y, y_alt).item()
            total_wave_mse += mse_loss(pred, aligned).item()
            total_sdr += calculate_sdr(pred, aligned).item()

            pred_np = aligned.detach().cpu().numpy()
            syms_a = batch["symbols_a"].numpy()
            syms_b = batch["symbols_b"].numpy()

            for i in range(pred_np.shape[0]):
                pred_a = pred_np[i, 0] + 1j * pred_np[i, 1]
                pred_b = pred_np[i, 2] + 1j * pred_np[i, 3]
                true_a = symbols_to_complex(syms_a[i])
                true_b = symbols_to_complex(syms_b[i])
                rec_a = recover_symbols_from_waveform(pred_a, taps, cfg.samples_per_symbol, len(true_a))
                rec_b = recover_symbols_from_waveform(pred_b, taps, cfg.samples_per_symbol, len(true_b))
                total_soi_acc += float(symbol_accuracy(rec_a[: len(true_a)], true_a))
                total_int_acc += float(symbol_accuracy(rec_b[: len(true_b)], true_b))
                count += 1

                if example is None:
                    example = {
                        "x": batch["x"][i].numpy(),
                        "y": batch["y"][i].numpy(),
                        "pred": pred_np[i],
                    }

    return {
        "pit_mse": total_pit_mse / max(1, len(loader)),
        "wave_mse": total_wave_mse / max(1, len(loader)),
        "sdr_db": total_sdr / max(1, len(loader)),
        "soi_symbol_accuracy": total_soi_acc / max(1, count),
        "int_symbol_accuracy": total_int_acc / max(1, count),
        "avg_symbol_accuracy": 0.5 * ((total_soi_acc / max(1, count)) + (total_int_acc / max(1, count))),
        "example": example,
    }


def save_reference_outputs(output_dir: Path, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "model_comparison_reference.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "pit_mse", "wave_mse", "sdr_db", "soi_symbol_accuracy", "int_symbol_accuracy", "avg_symbol_accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    tex_path = output_dir / "model_comparison_reference.tex"
    with open(tex_path, "w") as handle:
        handle.write("\\begin{table}[t]\n\\centering\n")
        handle.write("\\caption{Two-channel reference comparison using the main repo generator.}\n")
        handle.write("\\label{tab:phase2_two_channel_reference}\n")
        handle.write("\\begin{tabular}{|c|c|c|c|c|c|}\n\\hline\n")
        handle.write("Model & PIT-MSE & Wave MSE & SDR & SOI Acc & Int Acc \\\\ \n\\hline\n")
        for row in rows:
            model_name = row["model"].replace("_", r"\_")
            handle.write(
                f"{model_name} & {row['pit_mse']:.4f} & {row['wave_mse']:.4f} & {row['sdr_db']:.2f} & {row['soi_symbol_accuracy']:.3f} & {row['int_symbol_accuracy']:.3f} \\\\ \n"
            )
        handle.write("\\hline\n\\end{tabular}\n\\end{table}\n")


def main():
    project_root = PROJECT_ROOT
    output_root = project_root / "reporting_pipeline" / "outputs" / "phase2_two_channel_reference"
    dataset_root = output_root / "data"
    checkpoints_root = output_root / "checkpoints"
    figures_root = output_root / "figures"
    tables_root = output_root / "tables"
    json_root = output_root / "json"

    cfg = make_phase2_config(str(dataset_root))
    generate_fixed_splits(cfg, dataset_root, train_size=4096, val_size=512, test_size=1024, seed=31)

    train_loader = make_loader(dataset_root / "train", cfg.batch_size, shuffle=True)
    val_loader = make_loader(dataset_root / "val", cfg.batch_size, shuffle=False)
    test_loader = make_loader(dataset_root / "test", cfg.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    specs = active_specs(("FastICA", "Tiny", "Linear", "Hybrid", "LSTM", "IQ_CNN"), in_ch=estimate_input_channels(cfg))

    rows = []
    examples = {}
    for spec in specs:
        ensure_checkpoint(spec, cfg, train_loader, val_loader, checkpoints_root, device)
        model = load_trained_model(spec, checkpoints_root, device)
        metrics = evaluate_model_on_loader(model, test_loader, cfg)
        examples[spec.name] = metrics.pop("example")
        metrics["model"] = spec.name
        rows.append(metrics)

    rows.sort(key=lambda row: row["avg_symbol_accuracy"], reverse=True)
    save_reference_outputs(tables_root, rows)
    save_json(json_root / "reference_summary.json", {"config": asdict(cfg), "results": rows})

    plot_model_bar(figures_root / "reference_wave_mse.png", rows, "wave_mse", "Two-Channel Reference Wave MSE", "Wave MSE")
    plot_model_bar(figures_root / "reference_soi_accuracy.png", rows, "soi_symbol_accuracy", "Two-Channel Reference SOI Symbol Accuracy", "SOI Symbol Accuracy")

    best_learned = next(row for row in rows if row["model"] != "FastICA")
    plot_example_signals(figures_root / "best_learned_example.png", examples[best_learned["model"]]["x"], examples[best_learned["model"]]["y"], examples[best_learned["model"]]["pred"], title_prefix=best_learned["model"])
    plot_example_signals(figures_root / "fastica_example.png", examples["FastICA"]["x"], examples["FastICA"]["y"], examples["FastICA"]["pred"], title_prefix="FastICA")

    print(json.dumps({"config": asdict(cfg), "results": rows}, indent=2))
    print(f"Saved Phase 2 outputs to {output_root}")

if __name__ == "__main__":
    main()

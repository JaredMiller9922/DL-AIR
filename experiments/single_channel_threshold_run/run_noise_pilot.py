import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ExperimentConfig
from sweep_common import (
    BeautifulRFPlotter,
    final_evaluate,
    instantiate_fast_models,
    make_loader,
    rrc_taps,
    sampled_epoch_indices,
    seed_everything,
    train_fixed_epochs,
    write_csv_rows,
)
from utils.data_utils.generator import QPSKConfig, RFMixtureGenerator
from utils.model_utils.conversion_helpers import complex_matrix_to_iq_channels, stacked_sources_to_iq


def variance_tag(variance: float) -> str:
    return f"var_{variance:.4f}".replace(".", "p")


def build_qpsk_cfg() -> QPSKConfig:
    return QPSKConfig(
        n_symbols=ExperimentConfig.n_symbols,
        samples_per_symbol=ExperimentConfig.samples_per_symbol,
        rolloff=ExperimentConfig.rolloff,
        rrc_span_symbols=ExperimentConfig.rrc_span_symbols,
        normalize_power=ExperimentConfig.normalize_power,
        num_channels=ExperimentConfig.num_channels,
    )


def generate_split(split_dir: Path, split_size: int, variance: float, seed: int) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    gen = RFMixtureGenerator(seed=seed)
    qpsk_cfg = build_qpsk_cfg()
    alpha = ExperimentConfig.alpha

    for idx in range(split_size):
        s_soi, s_soi_symbols, _ = gen.generate_qpsk(qpsk_cfg)
        s_int, s_int_symbols, _ = gen.generate_qpsk(qpsk_cfg)
        clean = s_soi + alpha * s_int
        noise = np.sqrt(variance / 2.0) * (
            gen.rng.standard_normal(clean.shape) + 1j * gen.rng.standard_normal(clean.shape)
        )
        mixture = (clean + noise).astype(np.complex64)[np.newaxis, :]

        np.savez_compressed(
            split_dir / f"sample_{idx:06d}.npz",
            x=complex_matrix_to_iq_channels(mixture).astype(np.float32),
            y=stacked_sources_to_iq(s_soi.astype(np.complex64), s_int.astype(np.complex64)).astype(np.float32),
            symbols_a=np.stack([s_soi_symbols.real, s_soi_symbols.imag], axis=0).astype(np.float32),
            symbols_b=np.stack([s_int_symbols.real, s_int_symbols.imag], axis=0).astype(np.float32),
            alpha=np.array(alpha, dtype=np.float32),
            noise_variance=np.array(variance, dtype=np.float32),
        )


def generate_dataset(data_root: Path, variance: float, train_size: int, val_size: int, seed: int) -> Path:
    target = data_root / variance_tag(variance)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    generate_split(target / "train", train_size, variance, seed=seed)
    generate_split(target / "val", val_size, variance, seed=seed + 1)
    with open(target / "manifest.json", "w") as handle:
        json.dump(
            {
                "noise_variance": variance,
                "alpha": ExperimentConfig.alpha,
                "n_rx": ExperimentConfig.n_rx,
                "num_channels": ExperimentConfig.num_channels,
                "train_size": train_size,
                "val_size": val_size,
            },
            handle,
            indent=2,
        )
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small variance-based noise pilot")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--variances", nargs="+", type=float, default=[0.8, 0.16])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=ExperimentConfig.lr)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    run_name = args.run_name or f"noise_pilot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root = PROJECT_ROOT / "experiments" / "single_channel_threshold_run" / "noise_pilot" / run_name
    data_root = root / "data"
    plots_root = root / "plots"
    logs_root = root / "logs"
    checkpoints_root = root / "checkpoints"
    for path in [data_root, plots_root, logs_root, checkpoints_root]:
        path.mkdir(parents=True, exist_ok=True)

    taps = rrc_taps(ExperimentConfig.samples_per_symbol, ExperimentConfig.rolloff, ExperimentConfig.rrc_span_symbols)
    summary = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "variances": args.variances,
        "config_snapshot": {
            "alpha": ExperimentConfig.alpha,
            "n_rx": ExperimentConfig.n_rx,
            "num_channels": ExperimentConfig.num_channels,
            "epochs": args.epochs,
            "train_size": args.train_size,
            "val_size": args.val_size,
        },
        "results": {},
    }

    for idx, variance in enumerate(args.variances):
        print(f"=== Noise variance {variance:.4f} ===")
        split_root = generate_dataset(data_root, variance, args.train_size, args.val_size, seed=args.seed + 10 * idx)
        train_loader, train_ds = make_loader(split_root / "train", batch_size=args.batch_size, shuffle=True)
        val_loader, val_ds = make_loader(split_root / "val", batch_size=args.batch_size, shuffle=False)
        variance_plotter = BeautifulRFPlotter(save_dir=str(plots_root))
        variance_key = variance_tag(variance)
        variance_summary = {
            "noise_variance": variance,
            "data_root": str(split_root),
            "dataset_sizes": {"train": len(train_ds), "val": len(val_ds)},
            "models": {},
        }

        for model_name, model in instantiate_fast_models(device).items():
            print(f"--- {model_name} @ variance={variance:.4f} ---")
            checkpoint_dir = checkpoints_root / variance_key
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            trained = train_fixed_epochs(
                model_name=model_name,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                lr=args.lr,
                epochs=args.epochs,
                checkpoint_path=checkpoint_dir / f"{model_name}_best.pt",
            )
            metrics = final_evaluate(
                trained["model"],
                val_loader,
                device,
                variance_plotter,
                f"{model_name}_{variance_key}",
                taps,
                ExperimentConfig.samples_per_symbol,
                create_pipeline_plot=False,
            )
            keep = sampled_epoch_indices(trained["epochs"])
            model_summary = {
                "best_epoch": trained["best_epoch"],
                "best_val_mse": trained["best_val_mse"],
                "stop_epoch": trained["stop_epoch"],
                "stop_reason": trained["stop_reason"],
                "epochs_full": trained["epochs"],
                "train_history_full": trained["train_history"],
                "val_history_full": trained["val_history"],
                "sampled_epochs": [trained["epochs"][i] for i in keep],
                "sampled_train_history": [trained["train_history"][i] for i in keep],
                "sampled_val_history": [trained["val_history"][i] for i in keep],
                **metrics,
            }
            variance_summary["models"][model_name] = model_summary

        summary["results"][variance_key] = variance_summary

    with open(root / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    rows = []
    for variance_key, variance_summary in summary["results"].items():
        for model_name, model_summary in variance_summary["models"].items():
            rows.append(
                {
                    "noise_variance": variance_summary["noise_variance"],
                    "model": model_name,
                    "best_epoch": model_summary["best_epoch"],
                    "best_val_mse": model_summary["best_val_mse"],
                    "final_val_mse": model_summary["final_mse"],
                    "final_sdr_db": model_summary["final_sdr_db"],
                    "final_symbol_accuracy": model_summary["final_symbol_accuracy"],
                }
            )
    write_csv_rows(root / "noise_pilot_table.csv", rows)

    print("\nWave-recovery summary:")
    for variance_key, variance_summary in summary["results"].items():
        print(f"variance={variance_summary['noise_variance']:.4f}")
        for model_name, model_summary in variance_summary["models"].items():
            print(
                f"  {model_name}: best_val_mse={model_summary['best_val_mse']:.6f}, "
                f"final_sdr={model_summary['final_sdr_db']:.2f} dB, "
                f"sym_acc={model_summary['final_symbol_accuracy']:.4f}"
            )
    print(f"Artifacts saved to {root}")


if __name__ == "__main__":
    main()

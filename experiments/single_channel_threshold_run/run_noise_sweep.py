import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
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


DEFAULT_VARIANCES = [0.00, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20]
KNEE_VARIANCES = [0.00, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20]
QUALITATIVE_VARIANCES = [0.00, 0.08, 0.16]


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


def get_curve_arrays(summary_results: dict, metric_key: str, model_name: str, variances: list[float]) -> list[float]:
    values = []
    for variance in variances:
        values.append(summary_results[variance_tag(variance)]["models"][model_name][metric_key])
    return values


def metadata_label(alpha: float, epochs: int) -> str:
    return (
        f"n_rx={ExperimentConfig.n_rx}, num_channels={ExperimentConfig.num_channels}, "
        f"alpha={alpha}, noise sweep variance, epochs/model={epochs}"
    )


def plot_main_sweep(plots_dir: Path, summary_results: dict, variances: list[float], epochs: int) -> None:
    label = metadata_label(ExperimentConfig.alpha, epochs)
    models = ["Hybrid", "LSTM", "IQ_CNN"]

    plt.figure(figsize=(11, 6))
    for model_name in models:
        plt.plot(variances, get_curve_arrays(summary_results, "best_val_mse", model_name, variances), marker="o", label=model_name)
    plt.title(f"Single-RX Noise Sweep: Best Validation MSE\n{label}")
    plt.xlabel("Noise Variance")
    plt.ylabel("Best Validation MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "noise_variance_vs_best_val_mse.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    for model_name in models:
        plt.plot(variances, get_curve_arrays(summary_results, "final_mse", model_name, variances), marker="o", label=model_name)
    plt.title(f"Single-RX Noise Sweep: Final Validation MSE\n{label}")
    plt.xlabel("Noise Variance")
    plt.ylabel("Final Validation MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "noise_variance_vs_final_val_mse.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    for model_name in models:
        plt.plot(variances, get_curve_arrays(summary_results, "final_symbol_accuracy", model_name, variances), marker="o", label=model_name)
    plt.title(f"Single-RX Noise Sweep: Symbol Accuracy\n{label}")
    plt.xlabel("Noise Variance")
    plt.ylabel("Final Symbol Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "noise_variance_vs_symbol_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    for model_name in models:
        plt.plot(variances, get_curve_arrays(summary_results, "final_sdr_db", model_name, variances), marker="o", label=model_name)
    plt.title(f"Single-RX Noise Sweep: SDR\n{label}")
    plt.xlabel("Noise Variance")
    plt.ylabel("Final SDR (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "noise_variance_vs_sdr_db.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    for model_name in models:
        base = summary_results[variance_tag(0.0)]["models"][model_name]["best_val_mse"]
        rel = [summary_results[variance_tag(variance)]["models"][model_name]["best_val_mse"] / max(base, 1e-12) for variance in variances]
        plt.plot(variances, rel, marker="o", label=model_name)
    plt.title(f"Single-RX Noise Sweep: Relative MSE vs variance=0.0\n{label}")
    plt.xlabel("Noise Variance")
    plt.ylabel("Relative Best Validation MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "noise_variance_vs_relative_mse.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    for model_name in models:
        plt.plot(variances, get_curve_arrays(summary_results, "best_epoch", model_name, variances), marker="o", label=model_name)
    plt.title(f"Single-RX Noise Sweep: Best Epoch\n{label}")
    plt.xlabel("Noise Variance")
    plt.ylabel("Best Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "noise_variance_vs_best_epoch.png", dpi=300)
    plt.close()


def plot_knee_curves(plots_dir: Path, summary_results: dict, epochs: int) -> None:
    label = metadata_label(ExperimentConfig.alpha, epochs)
    available_knees = [variance for variance in KNEE_VARIANCES if variance_tag(variance) in summary_results]
    for model_name in ["Hybrid", "LSTM", "IQ_CNN"]:
        plt.figure(figsize=(11, 6))
        for variance in available_knees:
            result = summary_results[variance_tag(variance)]["models"][model_name]
            plt.plot(result["sampled_epochs"], result["sampled_val_history"], marker="o", label=f"var={variance:.2f}")
        plt.title(f"{model_name} Noise Knee Curves (Validation MSE)\n{label}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation MSE")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{model_name}_noise_knee_val_curves.png", dpi=300)
        plt.close()


def run_qualitative_for_anchor(
    qualitative_root: Path,
    data_root: Path,
    variance: float,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> None:
    variance_data_root = data_root / variance_tag(variance)
    variance_out = qualitative_root / variance_tag(variance)
    variance_out.mkdir(parents=True, exist_ok=True)

    train_loader, _ = make_loader(variance_data_root / "train", batch_size=batch_size, shuffle=True)
    val_loader, _ = make_loader(variance_data_root / "val", batch_size=batch_size, shuffle=False)
    plotter = BeautifulRFPlotter(save_dir=str(variance_out))
    taps = rrc_taps(ExperimentConfig.samples_per_symbol, ExperimentConfig.rolloff, ExperimentConfig.rrc_span_symbols)

    create_pipeline = True
    for model_name, model in instantiate_fast_models(device).items():
        checkpoint_dir = variance_out / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trained = train_fixed_epochs(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=lr,
            epochs=epochs,
            checkpoint_path=checkpoint_dir / f"{model_name}_best.pt",
        )
        final_evaluate(
            trained["model"],
            val_loader,
            device,
            plotter,
            f"{model_name}_var_{variance:.4f}",
            taps,
            ExperimentConfig.samples_per_symbol,
            create_pipeline_plot=create_pipeline,
        )
        create_pipeline = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full variance-based noise sweep for single-channel models")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--variances", nargs="+", type=float, default=DEFAULT_VARIANCES)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--lr", type=float, default=ExperimentConfig.lr)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--regenerate-data", action="store_true")
    parser.add_argument("--data-root", default="experiments/single_channel_threshold_run/noise_sweep/data")
    parser.add_argument("--outputs-root", default="experiments/single_channel_threshold_run/noise_sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    run_name = args.run_name or f"noise_sweep_single_rx_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_root = PROJECT_ROOT / args.outputs_root
    run_dir = base_root / "outputs" / run_name
    dirs = {
        "run_dir": run_dir,
        "plots": run_dir / "plots",
        "logs": run_dir / "logs",
        "checkpoints": run_dir / "checkpoints",
        "qualitative": run_dir / "qualitative",
        "data": run_dir / "data",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    data_root = PROJECT_ROOT / args.data_root
    data_root.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    if args.regenerate_data:
        for idx, variance in enumerate(args.variances):
            print(f"Generating dataset for variance={variance:.4f}")
            generate_dataset(data_root, variance, train_size=args.train_size, val_size=args.val_size, seed=args.seed + idx)

    taps = rrc_taps(ExperimentConfig.samples_per_symbol, ExperimentConfig.rolloff, ExperimentConfig.rrc_span_symbols)
    summary = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "variances": args.variances,
        "config_snapshot": {
            "alpha": ExperimentConfig.alpha,
            "n_rx": ExperimentConfig.n_rx,
            "num_channels": ExperimentConfig.num_channels,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "train_size": args.train_size,
            "val_size": args.val_size,
        },
        "results": {},
    }

    for variance in args.variances:
        variance_key = variance_tag(variance)
        variance_data_root = data_root / variance_key
        if not variance_data_root.exists():
            raise ValueError(f"Missing dataset for {variance_key}. Run with --regenerate-data or generate datasets first.")

        variance_dirs = {
            "logs": dirs["logs"] / variance_key,
            "checkpoints": dirs["checkpoints"] / variance_key,
        }
        for path in variance_dirs.values():
            path.mkdir(parents=True, exist_ok=True)

        train_loader, train_ds = make_loader(variance_data_root / "train", batch_size=args.batch_size, shuffle=True)
        val_loader, val_ds = make_loader(variance_data_root / "val", batch_size=args.batch_size, shuffle=False)
        plotter = BeautifulRFPlotter(save_dir=str(dirs["plots"]))
        variance_summary = {
            "noise_variance": variance,
            "data_root": str(variance_data_root),
            "dataset_sizes": {"train": len(train_ds), "val": len(val_ds)},
            "models": {},
        }

        for model_name, model in instantiate_fast_models(device).items():
            print(f"--- {model_name} @ variance={variance:.4f} ---")
            trained = train_fixed_epochs(
                model_name=model_name,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                lr=args.lr,
                epochs=args.epochs,
                checkpoint_path=variance_dirs["checkpoints"] / f"{model_name}_best.pt",
            )
            metrics = final_evaluate(
                trained["model"],
                val_loader,
                device,
                plotter,
                f"{model_name}_{variance_key}",
                taps,
                ExperimentConfig.samples_per_symbol,
                create_pipeline_plot=False,
            )

            keep_indices = sampled_epoch_indices(trained["epochs"])
            model_summary = {
                "best_epoch": trained["best_epoch"],
                "best_val_mse": trained["best_val_mse"],
                "stop_epoch": trained["stop_epoch"],
                "stop_reason": trained["stop_reason"],
                "epochs_full": trained["epochs"],
                "train_history_full": trained["train_history"],
                "val_history_full": trained["val_history"],
                "sampled_epochs": [trained["epochs"][i] for i in keep_indices],
                "sampled_train_history": [trained["train_history"][i] for i in keep_indices],
                "sampled_val_history": [trained["val_history"][i] for i in keep_indices],
                **metrics,
            }
            variance_summary["models"][model_name] = model_summary
            with open(variance_dirs["logs"] / f"{model_name}_results.json", "w") as handle:
                json.dump(model_summary, handle, indent=2)

        summary["results"][variance_key] = variance_summary

    with open(dirs["run_dir"] / "summary.json", "w") as handle:
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
                    "stop_epoch": model_summary["stop_epoch"],
                    "stop_reason": model_summary["stop_reason"],
                }
            )
    write_csv_rows(dirs["run_dir"] / "noise_sweep_table.csv", rows)

    plot_main_sweep(dirs["plots"], summary["results"], args.variances, args.epochs)
    plot_knee_curves(dirs["plots"], summary["results"], args.epochs)

    for variance in QUALITATIVE_VARIANCES:
        if variance in args.variances:
            print(f"Generating qualitative anchor plots for variance={variance:.4f}")
            run_qualitative_for_anchor(dirs["qualitative"], data_root, variance, args.epochs, args.batch_size, args.lr, device)

    print(f"Artifacts saved to {dirs['run_dir']}")


if __name__ == "__main__":
    main()

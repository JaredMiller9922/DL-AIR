import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep_common import (
    PROJECT_ROOT,
    BeautifulRFPlotter,
    ExperimentConfig,
    alpha_tag,
    build_run_dirs,
    final_evaluate,
    generate_dataset_for_alpha,
    instantiate_fast_models,
    make_loader,
    make_metadata_label,
    rrc_taps,
    sampled_epoch_indices,
    seed_everything,
    train_fixed_epochs,
    write_sweep_table,
)


DEFAULT_ALPHAS = [0.80, 0.85, 0.90, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.07, 1.10]
KNEE_ALPHAS = [0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10]
QUALITATIVE_ALPHAS = [0.80, 1.00, 1.05]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full alpha sweep for single-channel models")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--lr", type=float, default=ExperimentConfig.lr)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--regenerate-data", action="store_true")
    parser.add_argument("--data-root", default="experiments/single_channel_threshold_run/alpha_sweep/data")
    parser.add_argument("--plots-root", default="experiments/single_channel_threshold_run/alpha_sweep")
    return parser.parse_args()


def get_curve_arrays(summary_results: dict, metric_key: str, model_name: str, alphas: list[float]) -> list[float]:
    values = []
    for alpha in alphas:
        values.append(summary_results[alpha_tag(alpha)]["models"][model_name][metric_key])
    return values


def plot_main_sweep(plots_dir: Path, summary_results: dict, alphas: list[float], epochs: int) -> None:
    label = make_metadata_label(epochs)
    models = ["Hybrid", "LSTM", "IQ_CNN"]

    plt.figure(figsize=(11, 6))
    for model_name in models:
        plt.plot(alphas, get_curve_arrays(summary_results, "best_val_mse", model_name, alphas), marker="o", label=model_name)
    plt.title(f"Single-RX Alpha Sweep: Best Validation MSE\n{label}")
    plt.xlabel("alpha")
    plt.ylabel("Best Validation MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "alpha_vs_best_val_mse.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    for model_name in models:
        plt.plot(alphas, get_curve_arrays(summary_results, "final_mse", model_name, alphas), marker="o", label=model_name)
    plt.title(f"Single-RX Alpha Sweep: Final Validation MSE\n{label}")
    plt.xlabel("alpha")
    plt.ylabel("Final Validation MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "alpha_vs_final_val_mse.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    for model_name in models:
        plt.plot(alphas, get_curve_arrays(summary_results, "final_symbol_accuracy", model_name, alphas), marker="o", label=model_name)
    plt.title(f"Single-RX Alpha Sweep: Symbol Accuracy\n{label}")
    plt.xlabel("alpha")
    plt.ylabel("Final Symbol Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "alpha_vs_symbol_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    for model_name in models:
        plt.plot(alphas, get_curve_arrays(summary_results, "final_sdr_db", model_name, alphas), marker="o", label=model_name)
    plt.title(f"Single-RX Alpha Sweep: SDR\n{label}")
    plt.xlabel("alpha")
    plt.ylabel("Final SDR (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "alpha_vs_sdr_db.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    for model_name in models:
        base = summary_results[alpha_tag(0.80)]["models"][model_name]["best_val_mse"]
        rel = [summary_results[alpha_tag(alpha)]["models"][model_name]["best_val_mse"] / max(base, 1e-12) for alpha in alphas]
        plt.plot(alphas, rel, marker="o", label=model_name)
    plt.title(f"Single-RX Alpha Sweep: Relative MSE vs alpha=0.80\n{label}")
    plt.xlabel("alpha")
    plt.ylabel("Relative Best Validation MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "alpha_vs_relative_mse.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    for model_name in models:
        plt.plot(alphas, get_curve_arrays(summary_results, "best_epoch", model_name, alphas), marker="o", label=model_name)
    plt.title(f"Single-RX Alpha Sweep: Best Epoch\n{label}")
    plt.xlabel("alpha")
    plt.ylabel("Best Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "alpha_vs_best_epoch.png", dpi=300)
    plt.close()


def plot_knee_curves(plots_dir: Path, summary_results: dict, epochs: int) -> None:
    label = make_metadata_label(epochs)
    available_knees = [alpha for alpha in KNEE_ALPHAS if alpha_tag(alpha) in summary_results]
    for model_name in ["Hybrid", "LSTM", "IQ_CNN"]:
        plt.figure(figsize=(11, 6))
        for alpha in available_knees:
            result = summary_results[alpha_tag(alpha)]["models"][model_name]
            plt.plot(result["sampled_epochs"], result["sampled_val_history"], marker="o", label=f"alpha={alpha:.2f}")
        plt.title(f"{model_name} Knee Curves (Validation MSE)\n{label}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation MSE")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{model_name}_knee_val_curves.png", dpi=300)
        plt.close()


def run_qualitative_for_anchor(
    qualitative_root: Path,
    data_root: Path,
    alpha: float,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> None:
    alpha_data_root = data_root / alpha_tag(alpha)
    alpha_out = qualitative_root / alpha_tag(alpha)
    alpha_out.mkdir(parents=True, exist_ok=True)

    train_loader, _ = make_loader(alpha_data_root / "train", batch_size=batch_size, shuffle=True)
    val_loader, _ = make_loader(alpha_data_root / "val", batch_size=batch_size, shuffle=False)
    plotter = BeautifulRFPlotter(save_dir=str(alpha_out))
    taps = rrc_taps(ExperimentConfig.samples_per_symbol, ExperimentConfig.rolloff, ExperimentConfig.rrc_span_symbols)

    create_pipeline = True
    for model_name, model in instantiate_fast_models(device).items():
        trained = train_fixed_epochs(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=lr,
            epochs=epochs,
            checkpoint_path=alpha_out / f"{model_name}_best.pt",
        )
        final_evaluate(
            trained["model"],
            val_loader,
            device,
            plotter,
            f"{model_name}_alpha_{alpha:.2f}",
            taps,
            ExperimentConfig.samples_per_symbol,
            create_pipeline_plot=create_pipeline,
        )
        create_pipeline = False


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    run_name = args.run_name or f"alpha_sweep_single_rx_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_root = PROJECT_ROOT / args.plots_root
    dirs = build_run_dirs(base_root, run_name)
    data_root = PROJECT_ROOT / args.data_root
    data_root.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    if args.regenerate_data:
        for idx, alpha in enumerate(args.alphas):
            print(f"Generating dataset for alpha={alpha:.2f}")
            generate_dataset_for_alpha(data_root, alpha, train_size=args.train_size, val_size=args.val_size, seed=args.seed + idx)

    taps = rrc_taps(ExperimentConfig.samples_per_symbol, ExperimentConfig.rolloff, ExperimentConfig.rrc_span_symbols)
    summary = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "alphas": args.alphas,
        "config_snapshot": {
            "n_rx": ExperimentConfig.n_rx,
            "num_channels": ExperimentConfig.num_channels,
            "use_on_the_fly_data": ExperimentConfig.use_on_the_fly_data,
            "noise_enabled": ExperimentConfig.noise_enabled,
            "snr_db": ExperimentConfig.snr_db,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "train_size": args.train_size,
            "val_size": args.val_size,
        },
        "results": {},
    }

    for alpha in args.alphas:
        alpha_key = alpha_tag(alpha)
        alpha_data_root = data_root / alpha_key
        if not alpha_data_root.exists():
            raise ValueError(f"Missing dataset for {alpha_key}. Run with --regenerate-data or generate datasets first.")

        alpha_dirs = {
            "logs": dirs["logs"] / alpha_key,
            "checkpoints": dirs["checkpoints"] / alpha_key,
        }
        for path in alpha_dirs.values():
            path.mkdir(parents=True, exist_ok=True)

        train_loader, train_ds = make_loader(alpha_data_root / "train", batch_size=args.batch_size, shuffle=True)
        val_loader, val_ds = make_loader(alpha_data_root / "val", batch_size=args.batch_size, shuffle=False)
        plotter = BeautifulRFPlotter(save_dir=str(dirs["plots"]))
        alpha_summary = {
            "alpha": alpha,
            "data_root": str(alpha_data_root),
            "dataset_sizes": {"train": len(train_ds), "val": len(val_ds)},
            "models": {},
        }

        for model_name, model in instantiate_fast_models(device).items():
            print(f"--- {model_name} @ alpha={alpha:.2f} ---")
            trained = train_fixed_epochs(
                model_name=model_name,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                lr=args.lr,
                epochs=args.epochs,
                checkpoint_path=alpha_dirs["checkpoints"] / f"{model_name}_best.pt",
            )
            metrics = final_evaluate(
                trained["model"],
                val_loader,
                device,
                plotter,
                f"{model_name}_{alpha_key}",
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
            alpha_summary["models"][model_name] = model_summary
            with open(alpha_dirs["logs"] / f"{model_name}_results.json", "w") as handle:
                json.dump(model_summary, handle, indent=2)

        summary["results"][alpha_key] = alpha_summary

    with open(dirs["run_dir"] / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    write_sweep_table(dirs["run_dir"] / "alpha_sweep_table.csv", summary["results"])
    plot_main_sweep(dirs["plots"], summary["results"], args.alphas, args.epochs)
    plot_knee_curves(dirs["plots"], summary["results"], args.epochs)

    for alpha in QUALITATIVE_ALPHAS:
        if alpha in args.alphas:
            print(f"Generating qualitative anchor plots for alpha={alpha:.2f}")
            run_qualitative_for_anchor(dirs["qualitative"], data_root, alpha, args.epochs, args.batch_size, args.lr, device)

    print(f"Artifacts saved to {dirs['run_dir']}")


if __name__ == "__main__":
    main()

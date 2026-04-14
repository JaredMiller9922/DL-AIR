import argparse
import copy
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ExperimentConfig
from networks.htdemucs import RFHTDemucsWrapper
from networks.hybrid_separator import HybridSeparator
from networks.iq_cnn_separator import IQCNNSeparator
from networks.lstm_separator import LSTMSeparator
from utils.data_utils.dataset import SavedRFDataset
from utils.model_utils.losses import calculate_sdr, mse_loss
from utils.model_utils.symbol_utils import recover_symbols_from_waveform, rrc_taps, symbol_accuracy
from utils.plot_utils.plotting_utils import BeautifulRFPlotter


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SavedRFDatasetWithSwap(Dataset):
    def __init__(self, split_dir: str):
        self.ds = SavedRFDataset(split_dir)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        sample = self.ds[idx]
        y = sample["y"]
        sample["y_alt"] = torch.cat([y[2:4], y[0:2]], dim=0)
        return sample


def make_loader(split_dir: Path, batch_size: int, shuffle: bool) -> tuple[DataLoader, Dataset]:
    ds = SavedRFDatasetWithSwap(str(split_dir))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, ds


def choose_target(pred: torch.Tensor, y: torch.Tensor, y_alt: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, pred.ndim))
    loss_main = ((pred - y) ** 2).mean(dim=dims)
    loss_alt = ((pred - y_alt) ** 2).mean(dim=dims)
    use_alt = loss_alt < loss_main
    view_shape = [pred.shape[0]] + [1] * (pred.ndim - 1)
    return torch.where(use_alt.view(*view_shape), y_alt, y)


def evaluate_loss(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_alt = batch["y_alt"].to(device)
            pred = model(x)
            target = choose_target(pred, y, y_alt)
            total += mse_loss(pred, target).item()
    return total / len(loader)


def final_evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: str,
    plotter: BeautifulRFPlotter,
    model_name: str,
    taps: np.ndarray,
    sps: int,
    create_pipeline_plot: bool,
) -> dict:
    model.eval()
    total_mse = 0.0
    total_sdr = 0.0
    total_sym_acc = 0.0
    total_examples = 0

    with torch.no_grad():
        first_batch = next(iter(val_loader))
        x0 = first_batch["x"].to(device)
        y0 = first_batch["y"].to(device)
        y0_alt = first_batch["y_alt"].to(device)
        pred0 = model(x0)
        target0 = choose_target(pred0, y0, y0_alt)

        if create_pipeline_plot:
            plotter.plot_data_pipeline(x0, target0)
        plotter.plot_separation_performance(target0, pred0, model_name=model_name)

        pred0_np = pred0.cpu().numpy()
        target0_np = target0.cpu().numpy()
        y0_np = y0.cpu().numpy()
        y0_alt_np = y0_alt.cpu().numpy()
        true_sym_a = first_batch["symbols_a"].cpu().numpy()
        true_sym_b = first_batch["symbols_b"].cpu().numpy()

        pred_a0 = pred0_np[:, 0, :] + 1j * pred0_np[:, 1, :]
        pred_b0 = pred0_np[:, 2, :] + 1j * pred0_np[:, 3, :]
        true_a_c = true_sym_a[:, 0, :] + 1j * true_sym_a[:, 1, :]
        true_b_c = true_sym_b[:, 0, :] + 1j * true_sym_b[:, 1, :]

        i = 0
        use_swapped = np.mean((target0_np[i] - y0_alt_np[i]) ** 2) < np.mean((target0_np[i] - y0_np[i]) ** 2)
        ref_sym_a = true_b_c[i] if use_swapped else true_a_c[i]
        ref_sym_b = true_a_c[i] if use_swapped else true_b_c[i]

        rec_a = recover_symbols_from_waveform(pred_a0[i], taps, sps, len(ref_sym_a))
        rec_b = recover_symbols_from_waveform(pred_b0[i], taps, sps, len(ref_sym_b))
        plotter.plot_symbol_recovery(ref_sym_a, rec_a, model_name=f"{model_name}_SourceA")
        plotter.plot_symbol_recovery(ref_sym_b, rec_b, model_name=f"{model_name}_SourceB")

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_alt = batch["y_alt"].to(device)
            pred = model(x)
            target = choose_target(pred, y, y_alt)

            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()
            y_np = y.cpu().numpy()
            y_alt_np = y_alt.cpu().numpy()
            true_sym_a = batch["symbols_a"].cpu().numpy()
            true_sym_b = batch["symbols_b"].cpu().numpy()

            pred_a = pred_np[:, 0, :] + 1j * pred_np[:, 1, :]
            pred_b = pred_np[:, 2, :] + 1j * pred_np[:, 3, :]
            true_a_c = true_sym_a[:, 0, :] + 1j * true_sym_a[:, 1, :]
            true_b_c = true_sym_b[:, 0, :] + 1j * true_sym_b[:, 1, :]

            for i in range(pred_a.shape[0]):
                use_swapped = np.mean((target_np[i] - y_alt_np[i]) ** 2) < np.mean((target_np[i] - y_np[i]) ** 2)
                ref_sym_a = true_b_c[i] if use_swapped else true_a_c[i]
                ref_sym_b = true_a_c[i] if use_swapped else true_b_c[i]

                rec_a = recover_symbols_from_waveform(pred_a[i], taps, sps, len(ref_sym_a))
                rec_b = recover_symbols_from_waveform(pred_b[i], taps, sps, len(ref_sym_b))
                n_a = min(len(rec_a), len(ref_sym_a))
                n_b = min(len(rec_b), len(ref_sym_b))
                acc_a = symbol_accuracy(rec_a[:n_a], ref_sym_a[:n_a])
                acc_b = symbol_accuracy(rec_b[:n_b], ref_sym_b[:n_b])
                total_sym_acc += 0.5 * (acc_a + acc_b)
                total_examples += 1

            total_mse += mse_loss(pred, target).item()
            total_sdr += calculate_sdr(pred, target).item()

    return {
        "final_mse": total_mse / len(val_loader),
        "final_sdr_db": total_sdr / len(val_loader),
        "final_symbol_accuracy": total_sym_acc / total_examples if total_examples else 0.0,
    }


def sampled_epoch_indices(epochs: list[int]) -> list[int]:
    keep = []
    for idx, epoch in enumerate(epochs):
        if epoch <= 10 or epoch % 10 == 0 or idx == len(epochs) - 1:
            keep.append(idx)
    deduped = []
    seen = set()
    for idx in keep:
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    return deduped


def plot_model_curves(run_plots_dir: Path, results: dict, metadata_label: str) -> None:
    fast_models = [name for name in ["Hybrid", "LSTM", "IQ_CNN"] if name in results]

    plt.figure(figsize=(11, 6))
    for name in fast_models:
        plt.plot(results[name]["sampled_epochs"], results[name]["sampled_train_history"], label=f"{name} Train")
        plt.plot(results[name]["sampled_epochs"], results[name]["sampled_val_history"], linestyle="--", label=f"{name} Val")
    plt.title(f"Single-Channel Fast Models\n{metadata_label}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(run_plots_dir / "fast_models_train_val.png", dpi=300)
    plt.close()

    if "HTDemucs" in results:
        plt.figure(figsize=(11, 6))
        plt.plot(results["HTDemucs"]["sampled_epochs"], results["HTDemucs"]["sampled_train_history"], label="HTDemucs Train")
        plt.plot(results["HTDemucs"]["sampled_epochs"], results["HTDemucs"]["sampled_val_history"], linestyle="--", label="HTDemucs Val")
        plt.title(f"Single-Channel HTDemucs\n{metadata_label}")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_plots_dir / "htdemucs_train_val.png", dpi=300)
        plt.close()

    plt.figure(figsize=(11, 6))
    for name, result in results.items():
        plt.plot(result["sampled_epochs"], result["sampled_val_history"], label=name)
    plt.yscale("log")
    plt.title(f"All Models Validation MSE (Log Scale)\n{metadata_label}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_plots_dir / "all_models_val_logscale.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    for name, result in results.items():
        plt.plot(result["sampled_epochs"], result["sampled_val_history"], label=name)
    plt.title(f"All Models Validation MSE (Linear Scale)\n{metadata_label}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_plots_dir / "all_models_val_linear.png", dpi=300)
    plt.close()


def instantiate_models(device: str) -> dict:
    return {
        "Hybrid": {
            "model": HybridSeparator(in_ch=2, out_ch=4).to(device),
            "threshold": 1e-4,
            "max_epochs": 40,
        },
        "LSTM": {
            "model": LSTMSeparator(in_ch=2, out_ch=4).to(device),
            "threshold": 1e-4,
            "max_epochs": 40,
        },
        "IQ_CNN": {
            "model": IQCNNSeparator(in_ch=2, out_ch=4).to(device),
            "threshold": 1e-4,
            "max_epochs": 40,
        },
        "HTDemucs": {
            "model": RFHTDemucsWrapper(in_ch=2, out_ch=4).to(device),
            "threshold": 1e-3,
            "max_epochs": 150,
        },
    }


def train_until_threshold(
    model_name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    lr: float,
    threshold: float,
    max_epochs: int,
    checkpoint_path: Path,
) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_history = []
    val_history = []
    epochs = []

    epoch0_train = evaluate_loss(model, train_loader, device)
    epoch0_val = evaluate_loss(model, val_loader, device)
    train_history.append(epoch0_train)
    val_history.append(epoch0_val)
    epochs.append(0)

    best_val = epoch0_val
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stop_reason = None
    if epoch0_val <= threshold:
        stop_reason = "threshold_reached"
        torch.save({"epoch": 0, "model": best_state, "best_val_mse": best_val}, checkpoint_path)
        return {
            "model": model,
            "train_history": train_history,
            "val_history": val_history,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_val_mse": best_val,
            "stop_epoch": 0,
            "stop_reason": stop_reason,
        }

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_train = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_alt = batch["y_alt"].to(device)
            pred = model(x)
            target = choose_target(pred, y, y_alt)
            loss = mse_loss(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_train += loss.item()

        avg_train = total_train / len(train_loader)
        avg_val = evaluate_loss(model, val_loader, device)
        train_history.append(avg_train)
        val_history.append(avg_val)
        epochs.append(epoch)

        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"epoch": epoch, "model": best_state, "best_val_mse": best_val}, checkpoint_path)

        print(
            f"[{model_name}] Epoch {epoch}/{max_epochs} | "
            f"Train MSE {avg_train:.8f} | Val MSE {avg_val:.8f} | Best {best_val:.8f}"
        )

        if avg_val <= threshold:
            stop_reason = "threshold_reached"
            break

    if stop_reason is None:
        stop_reason = "max_epochs_reached"

    model.load_state_dict(best_state)
    return {
        "model": model,
        "train_history": train_history,
        "val_history": val_history,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "stop_epoch": epochs[-1],
        "stop_reason": stop_reason,
    }


def build_run_dirs(base_dir: Path, run_name: str) -> dict:
    run_dir = base_dir / "outputs" / run_name
    dirs = {
        "run_dir": run_dir,
        "plots": run_dir / "plots",
        "logs": run_dir / "logs",
        "checkpoints": run_dir / "checkpoints",
        "snapshots": run_dir / "snapshots",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated single-channel threshold run")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--lr", type=float, default=ExperimentConfig.lr)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline-threshold", type=float, default=1e-4)
    parser.add_argument("--demucs-threshold", type=float, default=1e-3)
    parser.add_argument("--baseline-max-epochs", type=int, default=40)
    parser.add_argument("--demucs-max-epochs", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    run_name = args.run_name or f"single_rx_threshold_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_root = PROJECT_ROOT / "experiments" / "single_channel_threshold_run"
    dirs = build_run_dirs(experiment_root, run_name)
    data_root = PROJECT_ROOT / args.data_root

    train_loader, train_ds = make_loader(data_root / "train", batch_size=args.batch_size, shuffle=True)
    val_loader, val_ds = make_loader(data_root / "val", batch_size=args.batch_size, shuffle=False)

    first_batch = next(iter(train_loader))
    in_ch = int(first_batch["x"].shape[1])
    if in_ch != 2:
        raise ValueError(f"Expected single-channel input with 2 IQ channels, got in_ch={in_ch}")

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    plotter = BeautifulRFPlotter(save_dir=str(dirs["plots"]))
    taps = rrc_taps(ExperimentConfig.samples_per_symbol, ExperimentConfig.rolloff, ExperimentConfig.rrc_span_symbols)

    models = instantiate_models(device)
    for name in ["Hybrid", "LSTM", "IQ_CNN"]:
        models[name]["threshold"] = args.baseline_threshold
        models[name]["max_epochs"] = args.baseline_max_epochs
    models["HTDemucs"]["threshold"] = args.demucs_threshold
    models["HTDemucs"]["max_epochs"] = args.demucs_max_epochs

    metadata_label = (
        f"n_rx={ExperimentConfig.n_rx}, num_channels={ExperimentConfig.num_channels}, "
        f"alpha={ExperimentConfig.alpha}, snr_db={ExperimentConfig.snr_db}, "
        f"saved_data=True"
    )

    summary = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "device": device,
        "config_snapshot": {
            "n_rx": ExperimentConfig.n_rx,
            "num_channels": ExperimentConfig.num_channels,
            "use_on_the_fly_data": ExperimentConfig.use_on_the_fly_data,
            "alpha": ExperimentConfig.alpha,
            "snr_db": ExperimentConfig.snr_db,
            "noise_enabled": ExperimentConfig.noise_enabled,
            "samples_per_symbol": ExperimentConfig.samples_per_symbol,
            "n_symbols": ExperimentConfig.n_symbols,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
        "thresholds": {
            "baseline_models": args.baseline_threshold,
            "HTDemucs": args.demucs_threshold,
        },
        "max_epochs": {
            "baseline_models": args.baseline_max_epochs,
            "HTDemucs": args.demucs_max_epochs,
        },
        "dataset_sizes": {
            "train": len(train_ds),
            "val": len(val_ds),
        },
        "results": {},
    }

    create_pipeline_plot = True
    for model_name in ["Hybrid", "LSTM", "IQ_CNN", "HTDemucs"]:
        model_info = models[model_name]
        checkpoint_path = dirs["checkpoints"] / f"{model_name}_best.pt"
        print(f"--- Training {model_name} ---")
        trained = train_until_threshold(
            model_name=model_name,
            model=model_info["model"],
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=args.lr,
            threshold=model_info["threshold"],
            max_epochs=model_info["max_epochs"],
            checkpoint_path=checkpoint_path,
        )

        metrics = final_evaluate(
            trained["model"],
            val_loader,
            device,
            plotter,
            model_name,
            taps,
            ExperimentConfig.samples_per_symbol,
            create_pipeline_plot=create_pipeline_plot,
        )
        create_pipeline_plot = False

        keep_indices = sampled_epoch_indices(trained["epochs"])
        model_summary = {
            "threshold": model_info["threshold"],
            "max_epochs": model_info["max_epochs"],
            "best_epoch": trained["best_epoch"],
            "stop_epoch": trained["stop_epoch"],
            "stop_reason": trained["stop_reason"],
            "best_val_mse": trained["best_val_mse"],
            "train_history_full": trained["train_history"],
            "val_history_full": trained["val_history"],
            "epochs_full": trained["epochs"],
            "sampled_epochs": [trained["epochs"][i] for i in keep_indices],
            "sampled_train_history": [trained["train_history"][i] for i in keep_indices],
            "sampled_val_history": [trained["val_history"][i] for i in keep_indices],
            "checkpoint_path": str(checkpoint_path),
            **metrics,
        }
        summary["results"][model_name] = model_summary

        with open(dirs["logs"] / f"{model_name}_results.json", "w") as handle:
            json.dump(model_summary, handle, indent=2)

        if trained["stop_reason"] == "threshold_reached":
            print(
                f"[{model_name}] Stopped at epoch {trained['stop_epoch']}: threshold_reached "
                f"(val_mse={trained['best_val_mse']:.8f} <= {model_info['threshold']:.8f})"
            )
        else:
            print(f"[{model_name}] Stopped at epoch {trained['stop_epoch']}: max_epochs_reached")

    plot_model_curves(dirs["plots"], summary["results"], metadata_label)

    with open(dirs["run_dir"] / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print("\nFinal summary:")
    for model_name in ["Hybrid", "LSTM", "IQ_CNN", "HTDemucs"]:
        result = summary["results"][model_name]
        print(
            f"{model_name}: best_val_mse={result['best_val_mse']:.8f}, "
            f"stop_epoch={result['stop_epoch']}, "
            f"stop_reason={result['stop_reason']}, "
            f"sym_acc={result['final_symbol_accuracy']:.4f}"
        )
    print(f"Artifacts saved to {dirs['run_dir']}")


if __name__ == "__main__":
    main()

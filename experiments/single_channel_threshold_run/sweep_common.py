import copy
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ExperimentConfig
from networks.hybrid_separator import HybridSeparator
from networks.iq_cnn_separator import IQCNNSeparator
from networks.lstm_separator import LSTMSeparator
from utils.data_utils.dataset import SavedRFDataset, SyntheticRFDataset
from utils.data_utils.generator import MixtureConfig, NoiseConfig, QPSKConfig, RFMixtureGenerator
from utils.model_utils.losses import calculate_sdr, mse_loss
from utils.model_utils.symbol_utils import recover_symbols_from_waveform, rrc_taps, symbol_accuracy
from utils.plot_utils.plotting_utils import BeautifulRFPlotter


def alpha_tag(alpha: float) -> str:
    return f"alpha_{alpha:.2f}".replace(".", "p")


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


def instantiate_fast_models(device: str) -> dict:
    return {
        "Hybrid": HybridSeparator(in_ch=2, out_ch=4).to(device),
        "LSTM": LSTMSeparator(in_ch=2, out_ch=4).to(device),
        "IQ_CNN": IQCNNSeparator(in_ch=2, out_ch=4).to(device),
    }


def train_fixed_epochs(
    model_name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    lr: float,
    epochs: int,
    checkpoint_path: Path,
) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_history = []
    val_history = []
    epoch_points = []

    epoch0_train = evaluate_loss(model, train_loader, device)
    epoch0_val = evaluate_loss(model, val_loader, device)
    train_history.append(epoch0_train)
    val_history.append(epoch0_val)
    epoch_points.append(0)

    best_val = epoch0_val
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    torch.save({"epoch": 0, "model": best_state, "best_val_mse": best_val}, checkpoint_path)

    for epoch in range(1, epochs + 1):
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
        epoch_points.append(epoch)

        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"epoch": epoch, "model": best_state, "best_val_mse": best_val}, checkpoint_path)

        print(f"[{model_name}] Epoch {epoch}/{epochs} | Train MSE {avg_train:.8f} | Val MSE {avg_val:.8f} | Best {best_val:.8f}")

    model.load_state_dict(best_state)
    return {
        "model": model,
        "train_history": train_history,
        "val_history": val_history,
        "epochs": epoch_points,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "stop_epoch": epoch_points[-1],
        "stop_reason": "fixed_budget",
    }


def build_run_dirs(base_dir: Path, run_name: str) -> dict:
    run_dir = base_dir / "outputs" / run_name
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
    return dirs


def write_csv_rows(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_generator_configs(alpha: float, noise_enabled: bool | None = None, snr_db: float | None = None) -> tuple:
    qpsk_cfg_soi = QPSKConfig(
        n_symbols=ExperimentConfig.n_symbols,
        samples_per_symbol=ExperimentConfig.samples_per_symbol,
        rolloff=ExperimentConfig.rolloff,
        rrc_span_symbols=ExperimentConfig.rrc_span_symbols,
        normalize_power=ExperimentConfig.normalize_power,
        num_channels=ExperimentConfig.num_channels,
    )
    qpsk_cfg_int = QPSKConfig(
        n_symbols=ExperimentConfig.n_symbols,
        samples_per_symbol=ExperimentConfig.samples_per_symbol,
        rolloff=ExperimentConfig.rolloff,
        rrc_span_symbols=ExperimentConfig.rrc_span_symbols,
        normalize_power=ExperimentConfig.normalize_power,
        num_channels=ExperimentConfig.num_channels,
    )
    noise_cfg = NoiseConfig(enabled=ExperimentConfig.noise_enabled if noise_enabled is None else noise_enabled)
    mix_cfg = MixtureConfig(
        alpha=alpha,
        snr_db=ExperimentConfig.snr_db if snr_db is None else snr_db,
        n_rx=1,
        random_phase=ExperimentConfig.random_phase,
    )
    return qpsk_cfg_soi, qpsk_cfg_int, noise_cfg, mix_cfg


def save_split(dataset: SyntheticRFDataset, split_dir: Path, split_size: int) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(split_size):
        sample = dataset[idx]
        save_dict = {
            "x": sample["x"].numpy().astype(np.float32),
            "y": sample["y"].numpy().astype(np.float32),
            "symbols_a": sample["symbols_a"].numpy().astype(np.float32),
            "symbols_b": sample["symbols_b"].numpy().astype(np.float32),
            "alpha": np.array(dataset.mix_cfg.alpha, dtype=np.float32),
            "snr_db": np.array(dataset.mix_cfg.snr_db, dtype=np.float32),
        }
        np.savez_compressed(split_dir / f"sample_{idx:06d}.npz", **save_dict)


def generate_dataset_for_alpha(data_root: Path, alpha: float, train_size: int, val_size: int, seed: int) -> Path:
    target_root = data_root / alpha_tag(alpha)
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    qpsk_cfg_soi, qpsk_cfg_int, noise_cfg, mix_cfg = build_generator_configs(alpha)
    dataset = SyntheticRFDataset(
        num_examples=1,
        generator=RFMixtureGenerator(seed=seed),
        qpsk_cfg_soi=qpsk_cfg_soi,
        qpsk_cfg_int=qpsk_cfg_int,
        noise_cfg=noise_cfg,
        mix_cfg=mix_cfg,
    )
    save_split(dataset, target_root / "train", train_size)
    save_split(dataset, target_root / "val", val_size)

    manifest = {
        "alpha": alpha,
        "train_size": train_size,
        "val_size": val_size,
        "n_rx": 1,
        "num_channels": ExperimentConfig.num_channels,
        "noise_enabled": noise_cfg.enabled,
        "snr_db": mix_cfg.snr_db,
        "samples_per_symbol": ExperimentConfig.samples_per_symbol,
        "n_symbols": ExperimentConfig.n_symbols,
    }
    with open(target_root / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    return target_root


def write_sweep_table(csv_path: Path, summary_results: dict) -> None:
    rows = []
    for alpha_key, alpha_result in summary_results.items():
        alpha_value = alpha_result["alpha"]
        for model_name, model_result in alpha_result["models"].items():
            rows.append(
                {
                    "alpha": alpha_value,
                    "model": model_name,
                    "best_epoch": model_result["best_epoch"],
                    "best_val_mse": model_result["best_val_mse"],
                    "final_val_mse": model_result["final_mse"],
                    "final_sdr_db": model_result["final_sdr_db"],
                    "final_symbol_accuracy": model_result["final_symbol_accuracy"],
                    "stop_epoch": model_result["stop_epoch"],
                    "stop_reason": model_result["stop_reason"],
                }
            )
    write_csv_rows(csv_path, rows)


def make_metadata_label(epochs: int) -> str:
    return (
        f"n_rx={ExperimentConfig.n_rx}, num_channels={ExperimentConfig.num_channels}, "
        f"noise_enabled={ExperimentConfig.noise_enabled}, snr_db={ExperimentConfig.snr_db}, "
        f"epochs/model={epochs}"
    )

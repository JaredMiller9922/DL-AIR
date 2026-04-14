import json
import random
import shutil
import sys
import argparse
from dataclasses import asdict
from pathlib import Path

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
from utils.data_utils.generator import MixtureConfig, NoiseConfig, QPSKConfig, RFMixtureGenerator
from utils.model_utils.conversion_helpers import complex_matrix_to_iq_channels, stacked_sources_to_iq
from utils.model_utils.losses import mse_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InMemoryDataset(Dataset):
    def __init__(self, samples: list[dict[str, torch.Tensor]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def generate_saved_split(split_dir: Path, split_size: int, alpha: float, seed: int) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    gen = RFMixtureGenerator(seed=seed)
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
    noise_cfg = NoiseConfig(enabled=ExperimentConfig.noise_enabled)
    mix_cfg = MixtureConfig(
        alpha=alpha,
        snr_db=ExperimentConfig.snr_db,
        n_rx=1,
        random_phase=ExperimentConfig.random_phase,
    )

    for idx in range(split_size):
        ex = gen.generate_mixture(qpsk_cfg_soi=qpsk_cfg_soi, qpsk_cfg_int=qpsk_cfg_int, noise_cfg=noise_cfg, mix_cfg=mix_cfg)
        mixture = ex["mixture"]
        if mixture.ndim == 1:
            mixture = mixture[np.newaxis, :]

        x = complex_matrix_to_iq_channels(mixture).astype(np.float32)
        y = stacked_sources_to_iq(ex["source_a"], ex["source_b"]).astype(np.float32)
        symbols_a = np.stack([ex["symbols_a"].real, ex["symbols_a"].imag], axis=0).astype(np.float32)
        symbols_b = np.stack([ex["symbols_b"].real, ex["symbols_b"].imag], axis=0).astype(np.float32)

        np.savez_compressed(
            split_dir / f"sample_{idx:06d}.npz",
            x=x,
            y=y,
            symbols_a=symbols_a,
            symbols_b=symbols_b,
            alpha=np.array(alpha, dtype=np.float32),
        )


def build_alpha_dataset(root: Path, alpha: float, train_size: int, val_size: int) -> Path:
    alpha_dir = root / f"alpha_{alpha:.2f}".replace(".", "p")
    if alpha_dir.exists():
        shutil.rmtree(alpha_dir)
    alpha_dir.mkdir(parents=True, exist_ok=True)

    generate_saved_split(alpha_dir / "train", train_size, alpha, seed=1000 + int(alpha * 1000))
    generate_saved_split(alpha_dir / "val", val_size, alpha, seed=2000 + int(alpha * 1000))

    with open(alpha_dir / "manifest.json", "w") as handle:
        json.dump({"alpha": alpha, "train_size": train_size, "val_size": val_size, "n_rx": 1}, handle, indent=2)
    return alpha_dir


def load_split(split_dir: Path) -> InMemoryDataset:
    samples = []
    for file_path in sorted(split_dir.glob("*.npz")):
        data = np.load(file_path)
        samples.append(
            {
                "x": torch.from_numpy(data["x"]).float(),
                "y": torch.from_numpy(data["y"]).float(),
            }
        )
    return InMemoryDataset(samples)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            total += mse_loss(pred, y).item()
    return total / len(loader)


def train_short(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: str, epochs: int, lr: float) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_hist = [evaluate(model, train_loader, device)]
    val_hist = [evaluate(model, val_loader, device)]

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            loss = mse_loss(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()

        train_hist.append(total / len(train_loader))
        val_hist.append(evaluate(model, val_loader, device))

    return {
        "train_history": train_hist,
        "val_history": val_hist,
        "best_val_mse": float(min(val_hist)),
        "final_val_mse": float(val_hist[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.80, 0.90, 1.00, 1.10])
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--run-name", type=str, default="default")
    args = parser.parse_args()

    seed_everything(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_root = PROJECT_ROOT / "experiments" / "single_channel_threshold_run" / "alpha_pilot" / args.run_name
    data_root = experiment_root / "data"
    output_root = experiment_root / "outputs"
    data_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    alphas = args.alphas
    train_size = args.train_size
    val_size = args.val_size
    epochs = args.epochs
    lr = ExperimentConfig.lr
    batch_size = 32

    summary = {
        "config_snapshot": asdict(ExperimentConfig(mode="train", model_name="pilot")),
        "pilot_setup": {
            "alphas": alphas,
            "train_size": train_size,
            "val_size": val_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
        },
        "results": {},
    }

    for alpha in alphas:
        print(f"=== Alpha {alpha:.2f} ===")
        alpha_dir = build_alpha_dataset(data_root, alpha, train_size, val_size)
        train_ds = load_split(alpha_dir / "train")
        val_ds = load_split(alpha_dir / "val")
        train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False)

        alpha_results = {}
        models = {
            "Hybrid": HybridSeparator(in_ch=2, out_ch=4).to(device),
            "LSTM": LSTMSeparator(in_ch=2, out_ch=4).to(device),
            "IQ_CNN": IQCNNSeparator(in_ch=2, out_ch=4).to(device),
        }

        for model_name, model in models.items():
            print(f"--- {model_name} @ alpha={alpha:.2f} ---")
            result = train_short(model, train_loader, val_loader, device, epochs=epochs, lr=lr)
            alpha_results[model_name] = result
            print(
                f"{model_name}: epoch0_val={result['val_history'][0]:.6f}, "
                f"best_val={result['best_val_mse']:.6f}, final_val={result['final_val_mse']:.6f}"
            )

        summary["results"][f"{alpha:.2f}"] = alpha_results

    with open(output_root / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Pilot summary saved to {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from networks.iq_cnn_separator import IQCNNSeparator
from utils.data_utils.generator import MixtureConfig, NoiseConfig, QPSKConfig, RFMixtureGenerator
from utils.model_utils.conversion_helpers import complex_matrix_to_iq_channels, stacked_sources_to_iq
from utils.model_utils.losses import mse_loss


@dataclass
class CheckConfig:
    train_size: int = 2000
    val_size: int = 400
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-3
    alpha: float = 1.0
    n_rx: int = 1
    noise_enabled: bool = False
    snr_db: float = 25.0
    n_symbols: int = 400
    sps: int = 2
    rolloff: float = 0.25
    span: int = 12
    normalize_power: bool = True


class InMemoryDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(size: int, seed: int, phase_shift_deg: float, cfg: CheckConfig):
    gen = RFMixtureGenerator(seed=seed)
    qpsk_cfg = QPSKConfig(
        n_symbols=cfg.n_symbols,
        samples_per_symbol=cfg.sps,
        rolloff=cfg.rolloff,
        rrc_span_symbols=cfg.span,
        normalize_power=cfg.normalize_power,
        num_channels=cfg.n_rx,
    )
    noise_cfg = NoiseConfig(enabled=cfg.noise_enabled, snr_db=cfg.snr_db)
    mix_cfg = MixtureConfig(
        alpha=cfg.alpha,
        snr_db=cfg.snr_db,
        n_rx=cfg.n_rx,
        random_phase=False,
        phase_shift_deg=0,
        interference_phase_shift=phase_shift_deg,
    )
    samples = []
    for _ in range(size):
        ex = gen.generate_mixture(qpsk_cfg, qpsk_cfg, noise_cfg, mix_cfg)
        mixture = ex["mixture"]
        if mixture.ndim == 1:
            mixture = mixture[np.newaxis, :]
        x = complex_matrix_to_iq_channels(mixture).astype(np.float32)
        y = stacked_sources_to_iq(ex["source_a"], ex["source_b"]).astype(np.float32)
        samples.append({"x": torch.from_numpy(x), "y": torch.from_numpy(y)})
    return InMemoryDataset(samples)


def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            total += mse_loss(pred, y).item()
    return total / len(loader)


def train_case(phase_shift_deg: float, cfg: CheckConfig, device: str):
    train_ds = build_dataset(cfg.train_size, 10, phase_shift_deg, cfg)
    val_ds = build_dataset(cfg.val_size, 20, phase_shift_deg, cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    model = IQCNNSeparator(in_ch=2, out_ch=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train_hist = [evaluate(model, train_loader, device)]
    val_hist = [evaluate(model, val_loader, device)]
    for _ in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            loss = mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        train_hist.append(evaluate(model, train_loader, device))
        val_hist.append(evaluate(model, val_loader, device))
    return {
        "phase_shift_deg": phase_shift_deg,
        "epoch0_val_mse": val_hist[0],
        "best_val_mse": min(val_hist),
        "final_val_mse": val_hist[-1],
        "val_history": val_hist,
    }


def main():
    cfg = CheckConfig()
    seed_everything(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    phases = [0, 15, 30, 45, 90, 180]
    results = [train_case(phase, cfg, device) for phase in phases]
    print(json.dumps({"config": cfg.__dict__, "results": results}, indent=2))


if __name__ == "__main__":
    main()

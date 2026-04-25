import json
import random
import sys
from dataclasses import dataclass, asdict
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
from utils.model_utils.symbol_utils import rrc_taps, recover_symbols_from_waveform, symbol_accuracy


@dataclass
class CheckConfig:
    train_size: int = 1200
    val_size: int = 240
    batch_size: int = 32
    epochs: int = 8
    lr: float = 1e-3
    alpha: float = 1.0
    n_rx: int = 4
    noise_enabled: bool = False
    snr_db: float = 25.0
    sigma2: float | None = None
    n_symbols: int = 100
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
    noise_cfg = NoiseConfig(enabled=cfg.noise_enabled, snr_db=cfg.snr_db, sigma2=cfg.sigma2)
    mix_cfg = MixtureConfig(
        alpha=cfg.alpha,
        snr_db=cfg.snr_db,
        n_rx=cfg.n_rx,
        random_phase=False,
        phase_shift_deg=phase_shift_deg,
        interference_phase_shift=0,
    )

    samples = []
    for _ in range(size):
        ex = gen.generate_mixture(qpsk_cfg, qpsk_cfg, noise_cfg, mix_cfg)
        x = complex_matrix_to_iq_channels(ex["mixture"]).astype(np.float32)
        y = stacked_sources_to_iq(ex["source_a"], ex["source_b"]).astype(np.float32)
        symbols_a = np.stack([ex["symbols_a"].real, ex["symbols_a"].imag], axis=0).astype(np.float32)
        symbols_b = np.stack([ex["symbols_b"].real, ex["symbols_b"].imag], axis=0).astype(np.float32)
        samples.append(
            {
                "x": torch.from_numpy(x),
                "y": torch.from_numpy(y),
                "symbols_a": torch.from_numpy(symbols_a),
                "symbols_b": torch.from_numpy(symbols_b),
            }
        )
    return InMemoryDataset(samples)


def evaluate_symbol_accuracy(model, loader, device: str, sps: int, rolloff: float, span: int):
    model.eval()
    taps = rrc_taps(sps=sps, beta=rolloff, span_symbols=span)
    total_sym_acc = 0.0
    total_examples = 0
    total_mse = 0.0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            total_mse += mse_loss(pred, y).item()
            pred_np = pred.cpu().numpy()
            true_sym_a = batch["symbols_a"].numpy()
            true_sym_b = batch["symbols_b"].numpy()

            pred_a = pred_np[:, 0, :] + 1j * pred_np[:, 1, :]
            pred_b = pred_np[:, 2, :] + 1j * pred_np[:, 3, :]
            true_sym_a_c = true_sym_a[:, 0, :] + 1j * true_sym_a[:, 1, :]
            true_sym_b_c = true_sym_b[:, 0, :] + 1j * true_sym_b[:, 1, :]

            for i in range(pred_a.shape[0]):
                rec_a = recover_symbols_from_waveform(pred_a[i], taps, sps, len(true_sym_a_c[i]))
                rec_b = recover_symbols_from_waveform(pred_b[i], taps, sps, len(true_sym_b_c[i]))

                n_aa = min(len(rec_a), len(true_sym_a_c[i]))
                n_bb = min(len(rec_b), len(true_sym_b_c[i]))
                direct_score = 0.5 * (
                    symbol_accuracy(rec_a[:n_aa], true_sym_a_c[i][:n_aa])
                    + symbol_accuracy(rec_b[:n_bb], true_sym_b_c[i][:n_bb])
                )

                n_ab = min(len(rec_a), len(true_sym_b_c[i]))
                n_ba = min(len(rec_b), len(true_sym_a_c[i]))
                swap_score = 0.5 * (
                    symbol_accuracy(rec_a[:n_ab], true_sym_b_c[i][:n_ab])
                    + symbol_accuracy(rec_b[:n_ba], true_sym_a_c[i][:n_ba])
                )

                total_sym_acc += max(direct_score, swap_score)
                total_examples += 1

    return {
        "val_mse": total_mse / max(1, len(loader)),
        "val_symbol_accuracy": total_sym_acc / max(1, total_examples),
    }


def train_case(phase_shift_deg: float, cfg: CheckConfig, device: str):
    train_ds = build_dataset(cfg.train_size, 10, phase_shift_deg, cfg)
    val_ds = build_dataset(cfg.val_size, 20, phase_shift_deg, cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = IQCNNSeparator(in_ch=2 * cfg.n_rx, out_ch=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

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

    metrics = evaluate_symbol_accuracy(model, val_loader, device, cfg.sps, cfg.rolloff, cfg.span)
    metrics["phase_shift_deg"] = phase_shift_deg
    return metrics


def main():
    cfg = CheckConfig()
    seed_everything(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    phases = [5, 10, 15, 30, 45, 90]
    results = [train_case(phase, cfg, device) for phase in phases]
    print(json.dumps({"config": asdict(cfg), "device": device, "results": results}, indent=2))


if __name__ == "__main__":
    main()

import json
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from networks.hybrid_separator import HybridSeparator
from networks.iq_cnn_separator import IQCNNSeparator
from networks.lstm_separator import LSTMSeparator
from utils.data_utils.generator import MixtureConfig, NoiseConfig, QPSKConfig, RFMixtureGenerator
from utils.model_utils.conversion_helpers import complex_matrix_to_iq_channels, stacked_sources_to_iq
from utils.model_utils.losses import align_to_pit_target, mse_loss, pit_mse_loss, pit_sdr
from utils.model_utils.symbol_utils import recover_symbols_from_waveform, rrc_taps, symbol_accuracy


@dataclass
class SanityConfig:
    model_name: str = "Hybrid"
    train_size: int = 4096
    val_size: int = 512
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-3
    n_rx: int = 4
    alpha: float = 1.0
    noise_enabled: bool = False
    snr_db: float = 100.0
    n_symbols: int = 400
    samples_per_symbol: int = 2
    rolloff: float = 0.25
    rrc_span_symbols: int = 12
    seed: int = 0
    fixed_mixing: bool = True


class PrecomputedRFDataset(Dataset):
    def __init__(self, num_examples: int, generator, qpsk_cfg_soi, qpsk_cfg_int, noise_cfg, mix_cfg):
        self.samples = []
        fixed_h = np.array(
            [
                [1.0 + 0.0j, 0.35 + 0.15j],
                [0.20 - 0.10j, 1.0 + 0.0j],
                [0.75 + 0.10j, -0.30 + 0.25j],
                [-0.15 + 0.40j, 0.60 - 0.20j],
            ],
            dtype=np.complex64,
        )[: mix_cfg.n_rx]
        fixed_h /= np.linalg.norm(fixed_h, axis=0, keepdims=True) + 1e-12

        for _ in range(num_examples):
            if mix_cfg.random_phase:
                ex = generator.generate_mixture(
                    qpsk_cfg_soi=qpsk_cfg_soi,
                    qpsk_cfg_int=qpsk_cfg_int,
                    noise_cfg=noise_cfg,
                    mix_cfg=mix_cfg,
                )
            else:
                source_a, symbols_a, _ = generator.generate_qpsk(qpsk_cfg_soi)
                source_b, symbols_b, _ = generator.generate_qpsk(qpsk_cfg_int)
                sources = np.vstack([source_a, mix_cfg.alpha * source_b])
                mixture = fixed_h @ sources
                if noise_cfg.enabled:
                    mixture = mixture + generator.generate_noise(mixture, mix_cfg.snr_db)
                ex = {
                    "mixture": mixture.astype(np.complex64),
                    "source_a": source_a.astype(np.complex64),
                    "source_b": source_b.astype(np.complex64),
                    "symbols_a": symbols_a.astype(np.complex64),
                    "symbols_b": symbols_b.astype(np.complex64),
                }

            mixture = ex["mixture"]
            if mixture.ndim == 1:
                mixture = mixture[np.newaxis, :]

            y = stacked_sources_to_iq(ex["source_a"], ex["source_b"])
            y_alt = np.concatenate([y[2:4], y[0:2]], axis=0).astype(np.float32)

            self.samples.append(
                {
                    "x": torch.from_numpy(complex_matrix_to_iq_channels(mixture)).float(),
                    "y": torch.from_numpy(y).float(),
                    "y_alt": torch.from_numpy(y_alt).float(),
                    "symbols_a": torch.from_numpy(
                        np.stack([ex["symbols_a"].real, ex["symbols_a"].imag], axis=0).astype(np.float32)
                    ),
                    "symbols_b": torch.from_numpy(
                        np.stack([ex["symbols_b"].real, ex["symbols_b"].imag], axis=0).astype(np.float32)
                    ),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_dataset(num_examples: int, seed: int, cfg: SanityConfig):
    generator = RFMixtureGenerator(seed=seed)
    qpsk_cfg_soi = QPSKConfig(
        n_symbols=cfg.n_symbols,
        samples_per_symbol=cfg.samples_per_symbol,
        rolloff=cfg.rolloff,
        rrc_span_symbols=cfg.rrc_span_symbols,
    )
    qpsk_cfg_int = QPSKConfig(
        n_symbols=cfg.n_symbols,
        samples_per_symbol=cfg.samples_per_symbol,
        rolloff=cfg.rolloff,
        rrc_span_symbols=cfg.rrc_span_symbols,
    )
    noise_cfg = NoiseConfig(enabled=cfg.noise_enabled)
    mix_cfg = MixtureConfig(alpha=cfg.alpha, snr_db=cfg.snr_db, n_rx=cfg.n_rx, random_phase=not cfg.fixed_mixing)
    return PrecomputedRFDataset(num_examples, generator, qpsk_cfg_soi, qpsk_cfg_int, noise_cfg, mix_cfg)


def evaluate(model, loader, device, taps, sps):
    model.eval()
    total_pit_mse = 0.0
    total_aligned_mse = 0.0
    total_sdr = 0.0
    total_symbol_acc = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_alt = batch["y_alt"].to(device)
            pred = model(x)

            total_pit_mse += pit_mse_loss(pred, y, y_alt).item()
            aligned_y = align_to_pit_target(pred, y, y_alt)
            total_aligned_mse += mse_loss(pred, aligned_y).item()
            total_sdr += pit_sdr(pred, y, y_alt).item()

            pred_np = pred.cpu().numpy()
            true_a = batch["symbols_a"].cpu().numpy()
            true_b = batch["symbols_b"].cpu().numpy()

            pred_a = pred_np[:, 0, :] + 1j * pred_np[:, 1, :]
            pred_b = pred_np[:, 2, :] + 1j * pred_np[:, 3, :]
            true_a_c = true_a[:, 0, :] + 1j * true_a[:, 1, :]
            true_b_c = true_b[:, 0, :] + 1j * true_b[:, 1, :]

            batch_use_alt = (((pred - y_alt) ** 2).mean(dim=tuple(range(1, pred.ndim))) < ((pred - y) ** 2).mean(dim=tuple(range(1, pred.ndim)))).cpu().numpy()

            for i in range(pred.shape[0]):
                ref_a = true_b_c[i] if batch_use_alt[i] else true_a_c[i]
                ref_b = true_a_c[i] if batch_use_alt[i] else true_b_c[i]
                rec_a = recover_symbols_from_waveform(pred_a[i], taps, sps, len(ref_a))
                rec_b = recover_symbols_from_waveform(pred_b[i], taps, sps, len(ref_b))
                n_a = min(len(rec_a), len(ref_a))
                n_b = min(len(rec_b), len(ref_b))
                acc_a = symbol_accuracy(rec_a[:n_a], ref_a[:n_a])
                acc_b = symbol_accuracy(rec_b[:n_b], ref_b[:n_b])
                total_symbol_acc += 0.5 * (acc_a + acc_b)
                total_examples += 1

    return {
        "pit_mse": total_pit_mse / len(loader),
        "aligned_mse": total_aligned_mse / len(loader),
        "pit_sdr_db": total_sdr / len(loader),
        "symbol_accuracy": total_symbol_acc / total_examples,
    }


def main():
    cfg = SanityConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = build_dataset(cfg.train_size, cfg.seed, cfg)
    val_ds = build_dataset(cfg.val_size, cfg.seed + 1, cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    in_ch = 2 * cfg.n_rx
    if cfg.model_name == "IQ_CNN":
        model = IQCNNSeparator(in_ch=in_ch, out_ch=4).to(device)
    elif cfg.model_name == "LSTM":
        model = LSTMSeparator(in_ch=in_ch, out_ch=4).to(device)
    elif cfg.model_name == "Hybrid":
        model = HybridSeparator(in_ch=in_ch, out_ch=4).to(device)
    else:
        raise ValueError(f"Unsupported model_name: {cfg.model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    taps = rrc_taps(cfg.samples_per_symbol, cfg.rolloff, cfg.rrc_span_symbols)

    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_train = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_alt = batch["y_alt"].to(device)
            pred = model(x)
            loss = pit_mse_loss(pred, y, y_alt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train += loss.item()

        train_metrics = evaluate(model, train_loader, device, taps, cfg.samples_per_symbol)
        val_metrics = evaluate(model, val_loader, device, taps, cfg.samples_per_symbol)
        row = {
            "epoch": epoch,
            "train_pit_mse": train_metrics["pit_mse"],
            "val_pit_mse": val_metrics["pit_mse"],
            "train_symbol_accuracy": train_metrics["symbol_accuracy"],
            "val_symbol_accuracy": val_metrics["symbol_accuracy"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}: "
            f"train PIT-MSE {train_metrics['pit_mse']:.6f} | "
            f"val PIT-MSE {val_metrics['pit_mse']:.6f} | "
            f"train sym acc {train_metrics['symbol_accuracy']:.4f} | "
            f"val sym acc {val_metrics['symbol_accuracy']:.4f}"
        )

    final_train = evaluate(model, train_loader, device, taps, cfg.samples_per_symbol)
    final_val = evaluate(model, val_loader, device, taps, cfg.samples_per_symbol)
    summary = {
        "config": cfg.__dict__,
        "model": cfg.model_name,
        "train": final_train,
        "val": final_val,
        "history": history,
    }
    print("Final summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

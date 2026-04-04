import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from reporting_pipeline.config import ReportEvalConfig
from utils.model_utils.conversion_helpers import complex_matrix_to_iq_channels, stacked_sources_to_iq
from utils.model_utils.symbol_utils import QPSK_POINTS, rrc_taps


FIXED_H_A = np.array([1.0 + 0.0j, 0.75 + 0.25j, 1.10 - 0.20j, 0.70 + 0.45j], dtype=np.complex64)
FIXED_H_B = np.array([0.35 + 0.80j, -0.15 + 0.95j, 0.55 + 0.50j, -0.60 + 0.30j], dtype=np.complex64)


class ReportDataset(Dataset):
    def __init__(self, split_dir: Path):
        self.files = sorted(split_dir.glob("*.npz"))
        if not self.files:
            raise ValueError(f"No .npz files found in {split_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx])
        sample = {
            "x": torch.from_numpy(data["x"]).float(),
            "y": torch.from_numpy(data["y"]).float(),
            "y_alt": torch.from_numpy(data["y_alt"]).float(),
            "symbols_a": torch.from_numpy(data["symbols_a"]).float(),
            "symbols_b": torch.from_numpy(data["symbols_b"]).float(),
            "labels_a": torch.from_numpy(data["labels_a"]).long(),
            "labels_b": torch.from_numpy(data["labels_b"]).long(),
        }
        return sample


def make_loader(split_dir: Path, batch_size: int, shuffle: bool):
    dataset = ReportDataset(split_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _qpsk_symbols(num_symbols: int, rng: np.random.Generator):
    labels = rng.integers(0, 4, size=num_symbols, endpoint=False)
    return QPSK_POINTS[labels], labels.astype(np.int64)


def _pulse_shape(symbols: np.ndarray, taps: np.ndarray, sps: int):
    up = np.zeros(len(symbols) * sps, dtype=np.complex64)
    up[::sps] = symbols
    shaped = np.convolve(up, taps, mode="same")
    power = np.mean(np.abs(shaped) ** 2) + 1e-12
    return (shaped / np.sqrt(power)).astype(np.complex64)


def _place_packet(packet: np.ndarray, frame_len: int, start: int):
    out = np.zeros(frame_len, dtype=np.complex64)
    stop = min(frame_len, start + len(packet))
    out[start:stop] = packet[: stop - start]
    return out


def _apply_carrier(sig: np.ndarray, center_bin: float, cfo_bin: float, phase: float):
    n = np.arange(len(sig), dtype=np.float32)
    carrier = np.exp(1j * (2.0 * math.pi * (center_bin + cfo_bin) * n / len(sig) + phase))
    return (sig * carrier).astype(np.complex64)


def _snr_db_to_noise_std(signal: np.ndarray, snr_db: float) -> float:
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / max(snr_linear, 1e-12)
    return float(np.sqrt(noise_power / 2.0))


def _build_source(num_symbols: int, frame_len: int, sps: int, taps: np.ndarray, center_bin: float, rng: np.random.Generator):
    symbols, labels = _qpsk_symbols(num_symbols, rng)
    packet = _pulse_shape(symbols, taps, sps)
    start_max = max(1, frame_len - len(packet))
    start = int(rng.integers(0, start_max, endpoint=False))
    amp = rng.uniform(0.9, 1.1)
    cfo = rng.uniform(-0.15, 0.15)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    framed = _place_packet(packet, frame_len, start)
    wave = amp * _apply_carrier(framed, center_bin=center_bin, cfo_bin=cfo, phase=phase)
    return wave.astype(np.complex64), symbols.astype(np.complex64), labels, {
        "amp": float(amp),
        "cfo": float(cfo),
        "phase": float(phase),
        "start": int(start),
        "center_bin": float(center_bin),
    }


def _make_example(config: ReportEvalConfig, alpha: float, snr_db: float, rng: np.random.Generator, taps: np.ndarray):
    source_a, symbols_a, labels_a, meta_a = _build_source(
        config.qpsk_symbols_soi,
        config.frame_len,
        config.samples_per_symbol,
        taps,
        center_bin=14.0,
        rng=rng,
    )
    source_b, symbols_b, labels_b, meta_b = _build_source(
        config.qpsk_symbols_int,
        config.frame_len,
        config.samples_per_symbol,
        taps,
        center_bin=36.0,
        rng=rng,
    )

    mixture = np.zeros((config.n_rx, config.frame_len), dtype=np.complex64)
    for rx in range(config.n_rx):
        mixture[rx] = FIXED_H_A[rx] * source_a + alpha * FIXED_H_B[rx] * source_b

    noise_std = _snr_db_to_noise_std(mixture, snr_db)
    noise = noise_std * (rng.standard_normal(mixture.shape) + 1j * rng.standard_normal(mixture.shape))
    mixture = mixture + noise.astype(np.complex64)

    save_dict = {
        "x": complex_matrix_to_iq_channels(mixture),
        "y": stacked_sources_to_iq(source_a, source_b),
        "y_alt": stacked_sources_to_iq(source_b, source_a),
        "symbols_a": np.stack([symbols_a.real, symbols_a.imag], axis=0).astype(np.float32),
        "symbols_b": np.stack([symbols_b.real, symbols_b.imag], axis=0).astype(np.float32),
        "labels_a": labels_a,
        "labels_b": labels_b,
        "alpha": np.array(alpha, dtype=np.float32),
        "snr_db": np.array(snr_db, dtype=np.float32),
    }
    meta = {
        "alpha": float(alpha),
        "snr_db": float(snr_db),
        "noise_std": float(noise_std),
        "source_a": meta_a,
        "source_b": meta_b,
        "channel_a_real": np.real(FIXED_H_A).astype(float).tolist(),
        "channel_a_imag": np.imag(FIXED_H_A).astype(float).tolist(),
        "channel_b_real": np.real(FIXED_H_B).astype(float).tolist(),
        "channel_b_imag": np.imag(FIXED_H_B).astype(float).tolist(),
    }
    return save_dict, meta


def generate_dataset(root_dir: Path, num_examples: int, config: ReportEvalConfig, alpha: float, snr_db: float, seed: int):
    root_dir.mkdir(parents=True, exist_ok=True)
    taps = rrc_taps(config.samples_per_symbol, config.rolloff, config.rrc_span_symbols)
    rng = np.random.default_rng(seed)
    metadata = []
    for idx in range(num_examples):
        example, meta = _make_example(config, alpha=alpha, snr_db=snr_db, rng=rng, taps=taps)
        np.savez_compressed(root_dir / f"sample_{idx:06d}.npz", **example)
        metadata.append(meta)

    with open(root_dir / "metadata.json", "w") as handle:
        json.dump(metadata[: min(16, len(metadata))], handle, indent=2)


def ensure_training_data(config: ReportEvalConfig):
    train_dir = config.datasets_dir / "train_reference"
    val_dir = config.datasets_dir / "val_reference"
    manifest_path = config.datasets_dir / "manifest.json"
    if not train_dir.exists() or not val_dir.exists():
        generate_dataset(train_dir, config.train_examples, config, config.train_alpha, config.train_snr_db, config.seed)
        generate_dataset(val_dir, config.val_examples, config, config.train_alpha, config.train_snr_db, config.seed + 1)
        with open(manifest_path, "w") as handle:
            json.dump({"config": asdict(config)}, handle, indent=2, default=str)
    return train_dir, val_dir


def ensure_sweep_eval_data(config: ReportEvalConfig):
    alpha_dirs = {}
    snr_dirs = {}
    for idx, alpha in enumerate(config.alpha_sweep):
        eval_dir = config.datasets_dir / f"alpha_{str(alpha).replace('.', 'p')}"
        if not eval_dir.exists():
            generate_dataset(eval_dir, config.eval_examples, config, alpha=alpha, snr_db=config.train_snr_db, seed=config.seed + 100 + idx)
        alpha_dirs[alpha] = eval_dir

    for idx, snr_db in enumerate(config.snr_sweep_db):
        eval_dir = config.datasets_dir / f"snr_{str(snr_db).replace('.', 'p').replace('-', 'm')}"
        if not eval_dir.exists():
            generate_dataset(eval_dir, config.eval_examples, config, alpha=config.train_alpha, snr_db=snr_db, seed=config.seed + 200 + idx)
        snr_dirs[snr_db] = eval_dir

    return alpha_dirs, snr_dirs

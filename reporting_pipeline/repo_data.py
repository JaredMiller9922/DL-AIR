import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import ExperimentConfig
from utils.data_utils.dataset import SavedRFDataset, SyntheticRFDataset
from utils.data_utils.generator import MixtureConfig, NoiseConfig, QPSKConfig, RFMixtureGenerator


def build_qpsk_config(cfg: ExperimentConfig) -> QPSKConfig:
    return QPSKConfig(
        n_symbols=cfg.n_symbols,
        samples_per_symbol=cfg.samples_per_symbol,
        rolloff=cfg.rolloff,
        rrc_span_symbols=cfg.rrc_span_symbols,
        normalize_power=cfg.normalize_power,
        num_channels=cfg.num_channels,
    )


def build_noise_config(cfg: ExperimentConfig) -> NoiseConfig:
    return NoiseConfig(enabled=cfg.noise_enabled)


def build_mixture_config(cfg: ExperimentConfig) -> MixtureConfig:
    return MixtureConfig(
        alpha=cfg.alpha,
        snr_db=cfg.snr_db,
        n_rx=cfg.n_rx,
        random_phase=cfg.random_phase,
    )


def generate_fixed_splits(cfg: ExperimentConfig, root_dir: Path, train_size: int, val_size: int, test_size: int, seed: int) -> None:
    generator = RFMixtureGenerator(seed=seed)
    dataset = SyntheticRFDataset(
        num_examples=train_size + val_size + test_size,
        generator=generator,
        qpsk_cfg_soi=build_qpsk_config(cfg),
        qpsk_cfg_int=build_qpsk_config(cfg),
        noise_cfg=build_noise_config(cfg),
        mix_cfg=build_mixture_config(cfg),
    )
    dataset.save_splits(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        root_dir=str(root_dir),
        overwrite=True,
    )
    with open(root_dir / "phase2_manifest.json", "w") as handle:
        json.dump({"config": asdict(cfg), "train_size": train_size, "val_size": val_size, "test_size": test_size}, handle, indent=2, default=str)


class RepoSavedRFDataset(Dataset):
    """Wraps SavedRFDataset and adds y_alt for PIT training/eval."""

    def __init__(self, split_dir: Path):
        self.ds = SavedRFDataset(str(split_dir))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        sample = self.ds[idx]
        y = sample["y"]
        y_alt = torch.cat([y[2:4], y[0:2]], dim=0)
        out = {
            "x": sample["x"],
            "y": y,
            "y_alt": y_alt,
        }
        if "symbols_a" in sample:
            out["symbols_a"] = sample["symbols_a"]
        if "symbols_b" in sample:
            out["symbols_b"] = sample["symbols_b"]
        return out


def make_loader(split_dir: Path, batch_size: int, shuffle: bool) -> DataLoader:
    ds = RepoSavedRFDataset(split_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def make_phase2_config(dataset_path: str) -> ExperimentConfig:
    return ExperimentConfig(
        mode="phase2_two_channel_reference",
        model_name="reporting_pipeline",
        dataset_path=dataset_path,
        generate_new_data=True,
        use_on_the_fly_data=False,
        batch_size=32,
        epochs=20,
        lr=1e-3,
        n_symbols=400,
        samples_per_symbol=2,
        rolloff=0.25,
        rrc_span_symbols=12,
        normalize_power=True,
        noise_enabled=True,
        alpha=1.0,
        snr_db=15.0,
        n_rx=2,
        random_phase=True,
    )


def estimate_input_channels(cfg: ExperimentConfig) -> int:
    return 2 * cfg.n_rx


def symbols_to_complex(symbols_2ch: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(symbols_2ch, torch.Tensor):
        symbols_2ch = symbols_2ch.numpy()
    return symbols_2ch[0] + 1j * symbols_2ch[1]

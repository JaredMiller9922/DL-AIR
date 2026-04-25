import os
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from utils.model_utils.conversion_helpers import *
from torch.utils.data import DataLoader
from config import ExperimentConfig

def make_loader(data_dir, batch_size=4, shuffle=False):
    ds = SavedRFDataset(data_dir)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    return loader, ds

class SyntheticRFDataset(Dataset):
    """
    On-the-fly synthetic RF dataset.

    Returns each sample as:
        x: (2, T)
        mixture: (T,) complex


    where:
      x = I/Q channels for the received mixture
      y = [srcA_I, srcA_Q, srcB_I, srcB_Q]
    """

    def __init__(
        self,
        num_examples: int,
        generator,
        qpsk_cfg_soi,
        qpsk_cfg_int,
        noise_cfg,
        mix_cfg,
        custom_symbols: Optional[str] = None,
        source_a_cfg=None,
        source_b_cfg=None,
        family_cfg=None,
        return_meta: bool = False,
    ):
        self.num_examples = num_examples
        self.generator = generator
        self.qpsk_cfg_soi = qpsk_cfg_soi
        self.qpsk_cfg_int = qpsk_cfg_int
        self.noise_cfg = noise_cfg
        self.mix_cfg = mix_cfg
        self.custom_symbols = custom_symbols
        self.source_a_cfg = source_a_cfg
        self.source_b_cfg = source_b_cfg
        self.family_cfg = family_cfg
        self.return_meta = return_meta

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int):
        ex = self.generator.generate_mixture(
            qpsk_cfg_soi=self.qpsk_cfg_soi,
            qpsk_cfg_int=self.qpsk_cfg_int,
            noise_cfg = self.noise_cfg,
            mix_cfg=self.mix_cfg,
            soi_message=self.custom_symbols,
            source_a_cfg=self.source_a_cfg,
            source_b_cfg=self.source_b_cfg,
            family_cfg=self.family_cfg,
        )

        mixture = ex["mixture"]      # (n_rx, T) complex
        source_a = ex["source_a"]    # (T,) complex
        source_b = ex["source_b"]    # (T,) complex

        # 1 channel case
        if self.mix_cfg.n_rx == 1:
            x = complex_to_2ch(mixture)     # (2*n_rx, T)
        else:
            x = complex_matrix_to_iq_channels(mixture)
        y = stacked_sources_to_iq(source_a, source_b)  # (4, T)

        sample = {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "symbols_a": torch.from_numpy(
                np.stack([ex["symbols_a"].real, ex["symbols_a"].imag], axis=0).astype(np.float32)
            ),
            "symbols_b": torch.from_numpy(
                np.stack([ex["symbols_b"].real, ex["symbols_b"].imag], axis=0).astype(np.float32)
            ),
            # "meta": ex["meta"],
        }

        if self.return_meta:
            sample["meta"] = ex["meta"]

        return sample

    def save_splits(
        self,
        train_size: int,
        val_size: int,
        test_size: int,
        root_dir: str = "data",
        overwrite: bool = False,
    ) -> None:
        """
        Generate fixed synthetic datasets and save them to disk.

        Output directory structure:
            data/
              train/
              val/
              test/
        """
        root = Path(root_dir)

        if root.exists() and overwrite:
            shutil.rmtree(root)

        root.mkdir(parents=True, exist_ok=True)

        self._save_split(root / "train", train_size)
        self._save_split(root / "val", val_size)
        self._save_split(root / "test", test_size)

        manifest = {
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
        }

        with open(root / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def _save_split(self, split_dir: Path, split_size: int) -> None:
        split_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(split_size):
            sample = self[idx]

            save_dict = {
                "x": sample["x"].numpy().astype(np.float32),
                "y": sample["y"].numpy().astype(np.float32),
            }

            save_dict["alpha"] = np.array(self.mix_cfg.alpha, dtype=np.float32)
            save_dict["snr_db"] = np.array(self.mix_cfg.snr_db, dtype=np.float32)

            save_dict["symbols_a"] = sample["symbols_a"].numpy().astype(np.float32)
            save_dict["symbols_b"] = sample["symbols_b"].numpy().astype(np.float32)

            np.savez_compressed(
                split_dir / f"sample_{idx:06d}.npz",
                **save_dict
            )

class SavedRFDataset(Dataset):
    """
    Loads previously saved synthetic RF examples from disk.
    """

    def __init__(self, split_dir: str):
        self.split_dir = Path(split_dir)
        self.files = sorted(self.split_dir.glob("*.npz"))

        if not self.files:
            raise ValueError(f"No .npz files found in {split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx])

        sample = {
            "x": torch.from_numpy(data["x"]).float(),
            "y": torch.from_numpy(data["y"]).float(),
        }

        if "symbols_a" in data:
            sample["symbols_a"] = torch.from_numpy(data["symbols_a"]).float()
        if "symbols_b" in data:
            sample["symbols_b"] = torch.from_numpy(data["symbols_b"]).float()

        return sample

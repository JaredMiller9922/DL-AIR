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
      x: (2*n_rx, T) float32
      y: (4, T)      float32
      meta: dict

    where:
      x = I/Q channels for the received mixture
      y = [srcA_I, srcA_Q, srcB_I, srcB_Q]
    """

    def __init__(
        self,
        num_examples: int,
        generator,
        qpsk_cfg,
        int_cfg,
        mix_cfg,
        permutation_invariant_targets: bool = False,
    ):
        self.num_examples = num_examples
        self.generator = generator
        self.qpsk_cfg = qpsk_cfg
        self.int_cfg = int_cfg
        self.mix_cfg = mix_cfg
        self.permutation_invariant_targets = permutation_invariant_targets

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int):
        ex = self.generator.generate_example(
            qpsk_cfg=self.qpsk_cfg,
            int_cfg=self.int_cfg,
            mix_cfg=self.mix_cfg,
        )

        mixture = ex["mixture"]      # (n_rx, T) complex
        source_a = ex["source_a"]    # (T,) complex
        source_b = ex["source_b"]    # (T,) complex

        x = complex_matrix_to_iq_channels(mixture)     # (2*n_rx, T)
        y = stacked_sources_to_iq(source_a, source_b)  # (4, T)

        sample = {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "meta": ex["meta"],
        }

        if self.permutation_invariant_targets:
            y_swapped = stacked_sources_to_iq(source_b, source_a)
            sample["y_alt"] = torch.from_numpy(y_swapped)

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
            "permutation_invariant_targets": self.permutation_invariant_targets,
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

            if "y_alt" in sample:
                save_dict["y_alt"] = sample["y_alt"].numpy().astype(np.float32)

            meta = sample["meta"]
            save_dict["alpha"] = np.array(meta.get("alpha", -1), dtype=np.float32)

            snr_db = meta.get("snr_db", None)
            save_dict["snr_db"] = np.array(
                -999.0 if snr_db is None else snr_db,
                dtype=np.float32
            )

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

        if "y_alt" in data:
            sample["y_alt"] = torch.from_numpy(data["y_alt"]).float()

        meta = {}
        if "alpha" in data:
            meta["alpha"] = float(data["alpha"])
        if "snr_db" in data:
            val = float(data["snr_db"])
            meta["snr_db"] = None if val == -999.0 else val

        sample["meta"] = meta
        return sample
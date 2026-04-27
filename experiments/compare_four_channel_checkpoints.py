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
from utils.model_utils.conversion_helpers import complex_matrix_to_iq_channels
from utils.model_utils.symbol_utils import rrc_taps, recover_symbols_from_waveform


@dataclass
class EvalConfig:
    alpha_values: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    sigma2_values: tuple[float, ...] = (0.0, 1.0, 5.0, 10.0, 20.0)
    trials: int = 4
    n_rx: int = 4
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


def count_matches(pred_syms: np.ndarray, true_syms: np.ndarray) -> int:
    n = min(len(pred_syms), len(true_syms))
    return int(np.sum(pred_syms[:n] == true_syms[:n]))


def evaluate_example(model, ex, taps, sps, device):
    x = complex_matrix_to_iq_channels(ex["mixture"]).astype(np.float32)
    x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x_tensor).squeeze(0).cpu().numpy()

    pred_a = pred[0] + 1j * pred[1]
    pred_b = pred[2] + 1j * pred[3]
    rec_a = recover_symbols_from_waveform(pred_a, taps, sps, len(ex["symbols_a"]))
    rec_b = recover_symbols_from_waveform(pred_b, taps, sps, len(ex["symbols_b"]))
    true_a = ex["symbols_a"]
    true_b = ex["symbols_b"]

    direct = count_matches(rec_a, true_a) + count_matches(rec_b, true_b)
    swapped = count_matches(rec_a, true_b) + count_matches(rec_b, true_a)
    if direct >= swapped:
        soi = count_matches(rec_a, true_a)
        intr = count_matches(rec_b, true_b)
    else:
        soi = count_matches(rec_b, true_a)
        intr = count_matches(rec_a, true_b)
    return soi, intr


def evaluate_checkpoint(checkpoint_path: Path, phase_shift_deg: int, cfg: EvalConfig, device: str):
    model = IQCNNSeparator(in_ch=8, out_ch=4).to(device)
    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    taps = rrc_taps(sps=cfg.sps, beta=cfg.rolloff, span_symbols=cfg.span)

    rows = []
    for alpha_idx, alpha in enumerate(cfg.alpha_values):
        for sigma_idx, sigma2 in enumerate(cfg.sigma2_values):
            soi_counts = []
            int_counts = []
            for trial in range(cfg.trials):
                seed = 10000 * alpha_idx + 100 * sigma_idx + trial
                gen = RFMixtureGenerator(seed=seed)
                qpsk_cfg = QPSKConfig(
                    n_symbols=cfg.n_symbols,
                    samples_per_symbol=cfg.sps,
                    rolloff=cfg.rolloff,
                    rrc_span_symbols=cfg.span,
                    normalize_power=cfg.normalize_power,
                    num_channels=cfg.n_rx,
                )
                noise_cfg = NoiseConfig(enabled=True, snr_db=None, sigma2=sigma2)
                mix_cfg = MixtureConfig(
                    alpha=alpha,
                    snr_db=None,
                    n_rx=cfg.n_rx,
                    random_phase=False,
                    phase_shift_deg=phase_shift_deg,
                    interference_phase_shift=0,
                )
                ex = gen.generate_mixture(qpsk_cfg, qpsk_cfg, noise_cfg, mix_cfg)
                soi, intr = evaluate_example(model, ex, taps, cfg.sps, device)
                soi_counts.append(soi)
                int_counts.append(intr)

            rows.append(
                {
                    "alpha": alpha,
                    "sigma2": sigma2,
                    "soi_mean_correct": float(np.mean(soi_counts)),
                    "int_mean_correct": float(np.mean(int_counts)),
                    "soi_mean_incorrect": float(cfg.n_symbols - np.mean(soi_counts)),
                    "int_mean_incorrect": float(cfg.n_symbols - np.mean(int_counts)),
                }
            )
    return rows


def main():
    seed_everything(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = EvalConfig()
    checkpoints = {
        "new": PROJECT_ROOT / "pytorch_models" / "IQ_CNN_4_channel.pt",
        "backup": PROJECT_ROOT / "pytorch_models" / "IQ_CNN_4_channel_backup.pt",
    }
    output = {"config": asdict(cfg), "device": device, "results": {}}
    for label, ckpt in checkpoints.items():
        output["results"][label] = {
            "phase5": evaluate_checkpoint(ckpt, 5, cfg, device),
            "phase10": evaluate_checkpoint(ckpt, 10, cfg, device),
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

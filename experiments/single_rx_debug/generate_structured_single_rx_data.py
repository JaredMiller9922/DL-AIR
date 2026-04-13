import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_utils.conversion_helpers import complex_matrix_to_iq_channels, stacked_sources_to_iq


QPSK_CONSTELLATION = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex64) / np.sqrt(2.0)
PULSE = np.array([0.15, 0.35, 0.35, 0.15], dtype=np.float32)
H_A = np.array([1.0 + 0.0j], dtype=np.complex64)
H_B = np.array([0.35 + 0.80j], dtype=np.complex64)


def qpsk_symbols(num_symbols, rng):
    idx = rng.integers(0, 4, size=num_symbols)
    return QPSK_CONSTELLATION[idx]


def pulse_shape(symbols, sps):
    up = np.zeros(len(symbols) * sps, dtype=np.complex64)
    up[::sps] = symbols
    real = np.convolve(up.real, PULSE, mode="same")
    imag = np.convolve(up.imag, PULSE, mode="same")
    return (real + 1j * imag).astype(np.complex64)


def window_signal(sig, length, start):
    out = np.zeros(length, dtype=np.complex64)
    stop = min(length, start + len(sig))
    out[start:stop] = sig[: stop - start]
    return out


def apply_carrier(sig, base_freq, cfo, phase):
    n = np.arange(len(sig), dtype=np.float32)
    carrier = np.exp(1j * (2.0 * math.pi * (base_freq + cfo) * n / len(sig) + phase))
    return (sig * carrier).astype(np.complex64)


def build_source(length, rng, symbol_range, base_freq):
    sps = 8
    num_symbols = int(rng.integers(symbol_range[0], symbol_range[1] + 1))
    symbols = qpsk_symbols(num_symbols, rng)
    shaped = pulse_shape(symbols, sps)
    start = int(rng.integers(0, max(1, length - len(shaped))))
    framed = window_signal(shaped, length, start)
    amp = rng.uniform(0.8, 1.2)
    cfo = rng.uniform(-0.25, 0.25)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    source = amp * apply_carrier(framed, base_freq=base_freq, cfo=cfo, phase=phase)
    return source.astype(np.complex64), {
        "num_symbols": int(num_symbols),
        "start": int(start),
        "amp": float(amp),
        "cfo": float(cfo),
        "phase": float(phase),
        "base_freq": float(base_freq),
    }


def generate_sample(length, rng):
    source_a, meta_a = build_source(length, rng, symbol_range=(20, 28), base_freq=14.0)
    source_b, meta_b = build_source(length, rng, symbol_range=(18, 24), base_freq=36.0)

    mixture = np.zeros((1, length), dtype=np.complex64)
    mixture[0] = H_A[0] * source_a + H_B[0] * source_b

    noise_std = rng.uniform(0.001, 0.004)
    noise = noise_std * (rng.standard_normal((1, length)) + 1j * rng.standard_normal((1, length)))
    mixture = mixture + noise.astype(np.complex64)

    x = complex_matrix_to_iq_channels(mixture)
    y = stacked_sources_to_iq(source_a, source_b)
    y_alt = stacked_sources_to_iq(source_b, source_a)
    metadata = {
        "source_a": meta_a,
        "source_b": meta_b,
        "noise_std": float(noise_std),
        "n_rx": 1,
    }
    return x, y, y_alt, metadata


def save_dataset(root_dir, train_size=10000, val_size=1000, test_size=1000, length=256, seed=29):
    root = Path(root_dir)
    if root.exists():
        shutil.rmtree(root)
    for split in ["train", "val", "test"]:
        (root / split).mkdir(parents=True, exist_ok=True)

    split_sizes = {"train": train_size, "val": val_size, "test": test_size}
    manifest = {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "n_rx": 1,
        "length": length,
        "description": "Single-RX structured QPSK packets with distinct carrier bands, random timing/phase/CFO/amplitude, fixed mixing, and light noise.",
    }

    for split_idx, (split_name, split_size) in enumerate(split_sizes.items()):
        rng = np.random.default_rng(seed + split_idx)
        for idx in range(split_size):
            x, y, y_alt, _ = generate_sample(length, rng)
            np.savez_compressed(root / split_name / f"sample_{idx:06d}.npz", x=x, y=y, y_alt=y_alt)

    with open(root / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    save_dataset(PROJECT_ROOT / "data" / "structured_single_rx")

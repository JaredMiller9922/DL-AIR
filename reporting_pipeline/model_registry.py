from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from networks.hybrid_separator import HybridSeparator
from networks.iq_cnn_separator import IQCNNSeparator
from networks.linear_separator import LinearSeparator
from networks.lstm_separator import LSTMSeparator
from networks.tiny_separator import TinySeparator
from reporting_pipeline.baselines import FastICABaseline


@dataclass
class ModelSpec:
    name: str
    kind: str
    builder: Callable
    checkpoint_name: str | None = None
    train_kwargs: dict | None = None


def build_registry():
    return {
        "FastICA": ModelSpec(
            name="FastICA",
            kind="baseline",
            builder=lambda: FastICABaseline(),
            checkpoint_name=None,
            train_kwargs=None,
        ),
        "Linear": ModelSpec(
            name="Linear",
            kind="learned",
            builder=lambda: LinearSeparator(in_ch=8, out_ch=4),
            checkpoint_name="linear_separator.pt",
            train_kwargs={"optimizer_name": "adamw", "scheduler_name": "plateau", "scheduler_patience": 4},
        ),
        "Tiny": ModelSpec(
            name="Tiny",
            kind="learned",
            builder=lambda: TinySeparator(in_ch=8, out_ch=4, hidden=64),
            checkpoint_name="tiny_separator.pt",
            train_kwargs={"optimizer_name": "adamw", "scheduler_name": "plateau", "scheduler_patience": 4},
        ),
        "Hybrid": ModelSpec(
            name="Hybrid",
            kind="learned",
            builder=lambda: HybridSeparator(in_ch=8, out_ch=4, hidden=64, num_blocks=4, dropout=0.1),
            checkpoint_name="hybrid_separator.pt",
            train_kwargs={"optimizer_name": "adamw", "scheduler_name": "plateau", "scheduler_patience": 4},
        ),
        "LSTM": ModelSpec(
            name="LSTM",
            kind="learned",
            builder=lambda: LSTMSeparator(in_ch=8, out_ch=4, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.15),
            checkpoint_name="lstm_separator.pt",
            train_kwargs={"optimizer_name": "adamw", "scheduler_name": "plateau", "scheduler_patience": 4},
        ),
        "IQ_CNN": ModelSpec(
            name="IQ_CNN",
            kind="learned",
            builder=lambda: IQCNNSeparator(in_ch=8, out_ch=4, base_channels=48, dropout=0.08),
            checkpoint_name="iq_cnn_separator.pt",
            train_kwargs={"optimizer_name": "adamw", "scheduler_name": "plateau", "scheduler_patience": 4},
        ),
    }


def active_specs(active_names):
    registry = build_registry()
    return [registry[name] for name in active_names if name in registry]


def load_trained_model(spec: ModelSpec, checkpoint_dir: Path, device: str):
    model = spec.builder()
    if spec.kind == "baseline":
        return model

    if spec.checkpoint_name is None:
        raise ValueError(f"No checkpoint configured for learned model {spec.name}")

    checkpoint_path = checkpoint_dir / spec.checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()

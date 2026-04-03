# config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentConfig:
    # Core mode
    mode: str  # "train" or "inference"

    # Model
    model_name: str
    model_path: Optional[str] = None
    dropout: float = 0.0 

    # Training params
    batch_size: int = 16
    epochs: int = 50
    lr: float = 1e-3

    # Signal params
    modulation: str = "QPSK"
    noise_alpha: float = 0.5
    noise_sigma: float = 0.1
    num_symbols: int = 1000
    num_sources: int = 2

    # Data
    dataset_path: str = "./data"
    generate_new_data: bool = False 

    # Advanced
    use_cross_val: bool = False
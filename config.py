# config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
# TODO: Is there a name that is more expressive than ExperimentConfig?
class ExperimentConfig:
    # Core mode
    mode: str  # "train" or "inference"

    # Model
    model_name: str
    model_path: Optional[str] = None
    dropout: float = 0.0 

    # Training params
    batch_size: int = 16
    epochs: int = 100
    lr: float = 1e-3

    # Signal params
    modulation: str = "QPSK"
    # TODO: This variable shouldn't exist mixture = s1 + \alpha * s2 + noise. alpha is defined in MixtureConfig below
    noise_alpha: float = 0.5
    noise_sigma: float = 0.1
    num_symbols: int = 1000
    num_sources: int = 2

    # Data
    dataset_path: str = "./data"
    generate_new_data: bool = False 

    # Advanced
    use_cross_val: bool = False


    # Config parameters that Jared needed

    # ----- QPSKConfig -------- #
    n_symbols: int = 400
    samples_per_symbol: int = 2
    rolloff: float = 0.25
    rrc_span_symbols: int = 12
    normalize_power: bool = True
    num_channels: int = 4

    # ----- NoiseConfig -------- #
    noise_enabled: bool = False

    # ----- MixtureConfig -------- #
    alpha: float = 1.0
    snr_db: Optional[float] = 100 # Default value of 100 makes no changes
    n_rx: int = 4
    random_phase: bool = True
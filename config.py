from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentConfig:
    # --- Core Mode ---
    mode: str = "train"  # "train", "evaluate", or "generate"

    # --- Model Definition ---
    model_name: str = "Hybrid"
    model_path: Optional[str] = None
    dropout: float = 0.0 

    # --- Training Hyperparameters ---
    batch_size: int = 16
    epochs: int = 100
    lr: float = 1e-3

    # Signal params
    modulation: str = "QPSK"

    # Data
    dataset_path: str = "./data"
    generate_new_data: bool = False 

    # Advanced
    use_cross_val: bool = False

    # ----- QPSKConfig -------- #
    num_symbols: int = 400
    n_rx: int = 2
    samples_per_symbol: int = 2
    rolloff: float = 0.25
    rrc_span_symbols: int = 12
    normalize_power: bool = True

    # ----- NoiseConfig -------- #
    noise_enabled: bool = True
    # IMPORTANT: Either snr_db or sigma2 should be set here not both
    snr_db: float = 25
    sigma2: float = None

    # ----- MixtureConfig -------- #
    noise_alpha: float = 1
    random_phase: bool = False
    phase_shift_deg: int = 5
    interference_phase_shift: int = 0

    # --- Data Management ---
    dataset_path: str = "data"
    use_on_the_fly_data: bool = False
    
    # --- Evaluation Specifics ---
    # Captures custom user bits (e.g., "10110011") from the UI for Source A
    custom_symbols: Optional[str] = None

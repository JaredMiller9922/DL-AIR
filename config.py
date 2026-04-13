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
    epochs: int = 50
    lr: float = 1e-3
    use_cross_val: bool = False

    # --- Signal & QPSK Parameters ---
    modulation: str = "QPSK"
    n_symbols: int = 400
    samples_per_symbol: int = 2
    rolloff: float = 0.25
    rrc_span_symbols: int = 12
    normalize_power: bool = True
    num_channels: int = 4

    # --- Mixture & Noise Parameters ---
    alpha: float = 1.0           
    noise_enabled: bool = False
    snr_db: Optional[float] = 100.0 
    n_rx: int = 4
    random_phase: bool = True
    noise_variance: Optional[float] = None

    # --- Data Management ---
    dataset_path: str = "data"
    use_on_the_fly_data: bool = False
    
    # --- Evaluation Specifics ---
    # Captures custom user bits (e.g., "10110011") from the UI for Source A
    custom_symbols: Optional[str] = None
    
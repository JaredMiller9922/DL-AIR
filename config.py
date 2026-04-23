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
    epochs: int = 20
    lr: float = 1e-3

    # Signal params
    modulation: str = "QPSK"

    # Advanced
    use_cross_val: bool = False

    # ----- QPSKConfig -------- #
    num_symbols: int = 400
    n_symbols: Optional[int] = None
    n_rx: int = 2
    samples_per_symbol: int = 2
    rolloff: float = 0.25
    rrc_span_symbols: int = 12
    normalize_power: bool = True

    # Optional time/sampling metadata used by CLI experiments.
    num_channels: int = 4
    fs: float = 1.0
    symbol_rate: Optional[float] = None
    T1: float = 0.0
    T2: Optional[float] = None

    # ----- NoiseConfig -------- #
    noise_enabled: bool = True
    # IMPORTANT: Either snr_db or sigma2/noise_variance should be set here, not both.
    snr_db: Optional[float] = 25.0
    sigma2: Optional[float] = None
    noise_variance: Optional[float] = None

    # ----- MixtureConfig -------- #
    alpha: float = 1.0
    # Legacy name retained for older scripts.
    noise_alpha: float = 0.8
    random_phase: bool = False
    phase_shift_deg: int = 5
    interference_phase_shift: int = 0

    # --- Data Management ---
    dataset_path: str = "data"
    generate_new_data: bool = False
    use_on_the_fly_data: bool = False
    train_examples: int = 2000
    val_examples: int = 200
    test_examples: int = 2000

    # --- Artifact locations ---
    save_dir: str = "visualizations"
    log_dir: str = "logs"
    model_dir: str = "pytorch_models"
    
    # --- Evaluation Specifics ---
    # Captures custom user bits (e.g., "10110011") from the UI for Source A
    custom_symbols: Optional[str] = None

    def __post_init__(self):
        if self.n_symbols is not None:
            self.num_symbols = self.n_symbols
        else:
            self.n_symbols = self.num_symbols

        if self.noise_variance is None and self.sigma2 is not None:
            self.noise_variance = self.sigma2
        elif self.sigma2 is None and self.noise_variance is not None:
            self.sigma2 = self.noise_variance

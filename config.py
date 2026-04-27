from dataclasses import dataclass, replace
from typing import Optional, Tuple

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

    # Expanded source/waveform controls. These default to the original
    # QPSK/QPSK regime so older scripts and UI calls continue to behave the
    # same unless they opt into a new benchmark or family mode.
    source_a_type: str = "QPSK"
    source_b_type: str = "QPSK"
    source_family_mode: bool = False
    source_a_family: Tuple[str, ...] = ("QPSK",)
    source_b_family: Tuple[str, ...] = ("QPSK",)
    modulation_family_sweep: bool = False

    # Advanced
    use_cross_val: bool = False

    # ----- QPSKConfig -------- #
    num_symbols: int = 400
    n_symbols: Optional[int] = None
    n_rx: int = 4
    samples_per_symbol: int = 2
    rolloff: float = 0.25
    rrc_span_symbols: int = 12
    normalize_power: bool = True
    n_symbols_range: Optional[Tuple[int, int]] = None
    samples_per_symbol_choices: Optional[Tuple[int, ...]] = None
    rolloff_range: Optional[Tuple[float, float]] = None

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
    phase_shift_deg: int = 10
    interference_phase_shift: int = 0
    mixing_mode: str = "phase_only"
    random_complex_mixing: bool = False
    timing_offset: int = 0
    timing_offset_range: Optional[Tuple[int, int]] = None
    carrier_offset: float = 0.0
    carrier_offset_range: Optional[Tuple[float, float]] = None
    phase_mismatch_deg: float = 0.0
    phase_mismatch_range_deg: Optional[Tuple[float, float]] = None
    amplitude_imbalance_db: float = 0.0
    amplitude_imbalance_db_range: Optional[Tuple[float, float]] = None
    alpha_range: Optional[Tuple[float, float]] = None
    snr_db_range: Optional[Tuple[float, float]] = None

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

    # --- Benchmark / transfer metadata ---
    benchmark_name: str = "benchmark_qpsk_baseline"
    transfer_eval_target: Optional[str] = None
    mit_aligned: bool = False
    
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

        if self.modulation and self.source_a_type == "QPSK":
            self.source_a_type = self.modulation

        if self.mit_aligned:
            self.n_rx = 4
            self.source_a_type = "QPSK"
            self.source_family_mode = True
            self.mixing_mode = "phase_only"
            self.benchmark_name = self.benchmark_name or "benchmark_mit_aligned"


BENCHMARK_PRESETS = {
    "benchmark_qpsk_baseline": ExperimentConfig(
        benchmark_name="benchmark_qpsk_baseline",
        source_a_type="QPSK",
        source_b_type="QPSK",
        source_family_mode=False,
        n_rx=2,
        num_symbols=400,
        samples_per_symbol=2,
        alpha=1.0,
        noise_alpha=1.0,
        snr_db=25.0,
        mixing_mode="phase_only",
    ),
    "benchmark_modulation_diverse": ExperimentConfig(
        benchmark_name="benchmark_modulation_diverse",
        source_family_mode=True,
        source_a_family=("BPSK", "QPSK", "8PSK", "16QAM"),
        source_b_family=("BPSK", "QPSK", "8PSK", "16QAM", "PAM"),
        n_symbols_range=(1024, 1024),
        samples_per_symbol_choices=(2,),
        rolloff_range=(0.2, 0.35),
        alpha_range=(0.3, 1.5),
        snr_db_range=(15.0, 35.0),
        timing_offset_range=(-2, 2),
        n_rx=2,
        mixing_mode="random_complex",
        random_complex_mixing=True,
        val_examples=64,
        test_examples=64,
    ),
    "benchmark_multichannel_phase_only": ExperimentConfig(
        benchmark_name="benchmark_multichannel_phase_only",
        source_family_mode=True,
        source_a_family=("QPSK",),
        source_b_family=("QPSK", "8PSK", "16QAM"),
        n_symbols_range=(256, 256),
        samples_per_symbol_choices=(2,),
        alpha_range=(0.4, 2.0),
        snr_db_range=(15.0, 35.0),
        n_rx=4,
        mixing_mode="phase_only",
        phase_shift_deg=5,
    ),
    "benchmark_mit_aligned": ExperimentConfig(
        benchmark_name="benchmark_mit_aligned",
        mit_aligned=True,
        source_family_mode=True,
        source_a_family=("QPSK",),
        source_b_family=("QPSK", "BPSK", "8PSK", "MULTITONE", "COLORED_NOISE", "BURSTY"),
        n_symbols_range=(4, 4),
        samples_per_symbol_choices=(16,),
        rolloff_range=(0.25, 0.35),
        alpha_range=(0.2, 2.5),
        snr_db_range=(15.0, 35.0),
        timing_offset_range=(-2, 2),
        carrier_offset_range=(-0.01, 0.01),
        phase_mismatch_range_deg=(-10.0, 10.0),
        amplitude_imbalance_db_range=(-2.0, 2.0),
        n_rx=4,
        mixing_mode="phase_only",
    ),
    "benchmark_interferer_diverse": ExperimentConfig(
        benchmark_name="benchmark_interferer_diverse",
        source_family_mode=True,
        source_a_family=("QPSK",),
        source_b_family=("QPSK", "8PSK", "16QAM", "MULTITONE", "COLORED_NOISE", "CHIRP", "BURSTY"),
        n_symbols_range=(128, 128),
        samples_per_symbol_choices=(2,),
        rolloff_range=(0.2, 0.35),
        alpha_range=(0.2, 2.0),
        snr_db_range=(10.0, 35.0),
        timing_offset_range=(-4, 4),
        carrier_offset_range=(-0.02, 0.02),
        n_rx=2,
        mixing_mode="random_complex",
        random_complex_mixing=True,
    ),
}


BENCHMARK_DESCRIPTIONS = {
    "benchmark_qpsk_baseline": "Original QPSK-vs-QPSK synthetic benchmark with fixed two-receiver phase-only mixing.",
    "benchmark_modulation_diverse": "Mixed digital-modulation benchmark that samples source/interferer waveform families and random complex mixing.",
    "benchmark_multichannel_phase_only": "Four-receiver benchmark with structured phase-only mixing and a wider interferer family.",
    "benchmark_mit_aligned": "MIT-style synthetic mode with four receivers, QPSK SOI, phase-only mixing, and short-frame difficulty sweeps.",
    "benchmark_interferer_diverse": "QPSK SOI benchmark with a broader interferer set including multitone, colored-noise, chirp, and bursty regimes.",
}


def benchmark_config(name: str, **overrides) -> ExperimentConfig:
    if name not in BENCHMARK_PRESETS:
        raise ValueError(f"Unknown benchmark preset '{name}'. Available: {sorted(BENCHMARK_PRESETS)}")
    cfg = replace(BENCHMARK_PRESETS[name])
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(f"ExperimentConfig has no field '{key}'")
        setattr(cfg, key, value)
    cfg.__post_init__()
    return cfg

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReportEvalConfig:
    project_root: Path
    outputs_dir: Path = field(init=False)
    datasets_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    tables_dir: Path = field(init=False)
    json_dir: Path = field(init=False)

    seed: int = 23
    device: str = "cuda"
    frame_len: int = 256
    n_rx: int = 1
    batch_size: int = 64

    train_examples: int = 12000
    val_examples: int = 1500
    eval_examples: int = 1024

    train_alpha: float = 1.0
    train_snr_db: float = 15.0
    alpha_sweep: tuple = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0)
    snr_sweep_db: tuple = (30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0)

    qpsk_symbols_soi: int = 128
    qpsk_symbols_int: int = 128
    samples_per_symbol: int = 2
    rolloff: float = 0.25
    rrc_span_symbols: int = 12

    train_epochs: int = 20
    early_stopping_patience: int = 6
    scheduler_patience: int = 4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_amp: bool = True
    force_retrain: bool = False

    active_models: tuple = ("FastICA", "Tiny", "Linear", "Hybrid", "LSTM", "IQ_CNN")

    @property
    def in_ch(self) -> int:
        return 2 * self.n_rx

    def __post_init__(self):
        self.outputs_dir = self.project_root / "reporting_pipeline" / "outputs"
        self.datasets_dir = self.outputs_dir / "datasets"
        self.checkpoints_dir = self.outputs_dir / "checkpoints"
        self.figures_dir = self.outputs_dir / "figures"
        self.tables_dir = self.outputs_dir / "tables"
        self.json_dir = self.outputs_dir / "json"


def make_default_config(project_root: Path) -> ReportEvalConfig:
    return ReportEvalConfig(project_root=project_root)

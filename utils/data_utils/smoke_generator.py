import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils.dataset import SyntheticRFDataset
from utils.data_utils.generator import (
    MixtureConfig,
    NoiseConfig,
    QPSKConfig,
    RFMixtureGenerator,
    SourceConfig,
    SourceFamilyConfig,
    mit_aligned_family_config,
)
from utils.model_utils.conversion_helpers import complex_matrix_to_iq_channels


def run_smoke_checks() -> None:
    gen = RFMixtureGenerator(seed=123)
    qpsk_cfg = QPSKConfig(n_symbols=16, samples_per_symbol=2)
    noise_cfg = NoiseConfig(enabled=True, snr_db=25)
    mix_cfg = MixtureConfig(alpha=0.8, n_rx=2)

    old = gen.generate_mixture(qpsk_cfg, qpsk_cfg, noise_cfg, mix_cfg)
    assert old["mixture"].shape[0] == 2
    assert old["mixture"].shape[1] == old["source_a"].shape[0]
    assert complex_matrix_to_iq_channels(old["mixture"]).shape[0] == 4

    for source_type in [
        "BPSK",
        "QPSK",
        "8PSK",
        "16QAM",
        "PAM",
        "MULTITONE",
        "COLORED_NOISE",
        "CHIRP",
        "BURSTY",
        "RECORDED",
    ]:
        wave, symbols, meta = gen.generate_source(
            SourceConfig(source_type=source_type, n_symbols=16, samples_per_symbol=2)
        )
        assert wave.ndim == 1 and wave.size > 0
        assert symbols.ndim == 1 and symbols.size > 0
        assert meta["source_type"]

    family_cfg = SourceFamilyConfig(
        source_a_types=("BPSK", "QPSK"),
        source_b_types=("8PSK", "16QAM", "MULTITONE"),
        n_symbols_range=(16, 16),
        samples_per_symbol_choices=(2,),
        alpha_range=(0.5, 1.0),
        snr_db_range=(20.0, 25.0),
        n_rx=4,
        mixing_modes=("phase_only",),
    )
    family = gen.generate_mixture(qpsk_cfg, qpsk_cfg, noise_cfg, MixtureConfig(n_rx=4), family_cfg=family_cfg)
    assert family["mixture"].shape[0] == 4
    assert complex_matrix_to_iq_channels(family["mixture"]).shape[0] == 8

    mit_like = gen.generate_mixture(
        qpsk_cfg,
        qpsk_cfg,
        noise_cfg,
        MixtureConfig(n_rx=4),
        family_cfg=mit_aligned_family_config(),
    )
    assert mit_like["mixture"].shape[0] == 4

    dataset = SyntheticRFDataset(2, gen, qpsk_cfg, qpsk_cfg, noise_cfg, mix_cfg)
    sample = dataset[0]
    assert sample["x"].shape[0] == 4
    assert sample["y"].shape[0] == 4

    print("RF generator smoke checks passed.")


if __name__ == "__main__":
    run_smoke_checks()

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.data_utils.generator import RFMixtureGenerator, QPSKConfig, MixtureConfig, NoiseConfig
from utils.data_utils.dataset import SyntheticRFDataset
from config import ExperimentConfig


if __name__ == "__main__":
    generator = RFMixtureGenerator(seed = 0)

    qpsk_cfg_soi = QPSKConfig(
        n_symbols=ExperimentConfig.n_symbols,
        samples_per_symbol=ExperimentConfig.samples_per_symbol,
        rolloff=ExperimentConfig.rolloff,
        rrc_span_symbols=ExperimentConfig.rrc_span_symbols,
    )

    qpsk_cfg_int = QPSKConfig(
        n_symbols=ExperimentConfig.n_symbols,
        samples_per_symbol=ExperimentConfig.samples_per_symbol,
        rolloff=ExperimentConfig.rolloff,
        rrc_span_symbols=ExperimentConfig.rrc_span_symbols,
    )

    noise_cfg = NoiseConfig(
        enabled=ExperimentConfig.noise_enabled
    )

    mix_cfg = MixtureConfig(
        alpha=ExperimentConfig.alpha,
        snr_db=ExperimentConfig.snr_db,
        n_rx=ExperimentConfig.n_rx,
        random_phase=ExperimentConfig.random_phase,
    )

    gen = RFMixtureGenerator(seed=0)

    dataset = SyntheticRFDataset(
        num_examples=1,   # not used by save_splits directly except __getitem__
        generator=gen,
        qpsk_cfg_soi=qpsk_cfg_soi,
        qpsk_cfg_int=qpsk_cfg_int,
        noise_cfg=noise_cfg,
        mix_cfg=mix_cfg,
    )

    dataset.save_splits(
        train_size=10000,
        val_size=1000,
        test_size=1000,
        root_dir=ExperimentConfig.dataset_path,
        overwrite=True,
    )

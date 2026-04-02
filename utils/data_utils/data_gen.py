import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from generator import RFMixtureGenerator, QPSKConfig, MixtureConfig, NoiseConfig
from utils.data_utils.dataset import SyntheticRFDataset


if __name__ == "__main__":
    generator = RFMixtureGenerator(seed = 0)

    qpsk_cfg_soi = QPSKConfig(
        n_symbols=400,
        samples_per_symbol=2,
        rolloff=0.25,
        rrc_span_symbols=12,
    )

    qpsk_cfg_int = QPSKConfig(
        n_symbols=400,
        samples_per_symbol=2,
        rolloff=0.25,
        rrc_span_symbols=12,
    )

    noise_cfg = NoiseConfig(
        enabled=True
    )

    mix_cfg = MixtureConfig(
        alpha=0.8,
        snr_db=25.0,
    )

    gen = RFMixtureGenerator(seed=0)

    dataset = SyntheticRFDataset(
        num_examples=1,   # not used by save_splits directly except __getitem__
        generator=gen,
        qpsk_cfg_soi=qpsk_cfg_soi,
        qpsk_cfg_int=qpsk_cfg_int,
        noise_cfg=noise_cfg,
        mix_cfg=mix_cfg,
        permutation_invariant_targets=True,
    )

    dataset.save_splits(
        train_size=10,
        val_size=1,
        test_size=1,
        root_dir="data",
        overwrite=True,
    )
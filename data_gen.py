from generator import RFMixtureGenerator, QPSKConfig, InterfererConfig, MixtureConfig
from dataset import SyntheticRFDataset

if __name__ == "__main__":
    generator = RFMixtureGenerator(seed = 0)

    qpsk_cfg = QPSKConfig(
        n_symbols=400,
        samples_per_symbol=2,
        rolloff=0.25,
        rrc_span_symbols=12,
        include_preamble=False,
    )

    int_cfg = InterfererConfig(
        mode="bandlimited_noise",
        fir_len=33,
    )

    mix_cfg = MixtureConfig(
        n_rx=4,
        alpha=0.8,
        snr_db=25.0,
    )

    gen = RFMixtureGenerator(seed=0)

    dataset = SyntheticRFDataset(
        num_examples=1,   # not used by save_splits directly except __getitem__
        generator=gen,
        qpsk_cfg=qpsk_cfg,
        int_cfg=int_cfg,
        mix_cfg=mix_cfg,
        permutation_invariant_targets=True,
    )

    dataset.save_splits(
        train_size=5000,
        val_size=500,
        test_size=500,
        root_dir="data",
        overwrite=True,
    )
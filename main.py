import torch

from train import train_model
from evaluate import ModelEvaluator
from networks.hybrid_separator import HybridSeparator
from networks.lstm_separator import LSTMSeparator
from networks.linear_separator import LinearSeparator
from networks.iq_cnn_separator import IQCNNSeparator
from networks.htdemucs import RFHTDemucsWrapper
from networks.fast_ica import FastICABaseline
from utils.data_utils.dataset import make_loader
from utils.plot_utils.plotting_utils import BeautifulRFPlotter
from cross_validator import GridSearchManager
from utils.model_utils.symbol_utils import rrc_taps
from utils.data_utils.generator import QPSKConfig
from config import ExperimentConfig
from utils.data_utils.generator import RFMixtureGenerator, QPSKConfig, NoiseConfig, MixtureConfig
from utils.data_utils.dataset import SyntheticRFDataset
from torch.utils.data import DataLoader

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if ExperimentConfig.use_on_the_fly_data:
        print("Using on the fly data generation")

        gen = RFMixtureGenerator(seed=0)

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
        )

        train_ds = SyntheticRFDataset(
            num_examples=20000,  # controls "epoch size"
            generator=gen,
            qpsk_cfg_soi=qpsk_cfg_soi,
            qpsk_cfg_int=qpsk_cfg_int,
            noise_cfg=noise_cfg,
            mix_cfg=mix_cfg,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=ExperimentConfig.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        # KEEP validation FIXED
        val_loader, _ = make_loader("data/val", batch_size=ExperimentConfig.batch_size)
    
    else:
        print("Using data stored in data folder")
        train_loader, _ = make_loader("data/train", batch_size=16, shuffle=True)
        val_loader, _ = make_loader("data/val", batch_size=16)
    
    plotter = BeautifulRFPlotter(save_dir="visualizations")

    rrc = rrc_taps(
        sps=ExperimentConfig.samples_per_symbol,
        beta=ExperimentConfig.rolloff,
        span_symbols=ExperimentConfig.rrc_span_symbols,
    )

    evaluator = ModelEvaluator(
        train_loader=train_loader,
        val_loader=val_loader,
        plotter=plotter,
        rrc_taps=rrc,
        sps=ExperimentConfig.samples_per_symbol,
        device=device,
    )

    all_results = {}
    print("WE MADE IT BEFORE THE Separator")

    models_to_test = {
        # "Hybrid": {"model": HybridSeparator(in_ch=4, out_ch=4).to(device), "train": True},
        # "LSTM": {"model": LSTMSeparator(in_ch=4, out_ch=4).to(device), "train": True},
        # "IQ_CNN": {"model": IQCNNSeparator(in_ch=4, out_ch=4).to(device), "train": True},
        # "HTDemucs": {"model": RFHTDemucsWrapper(in_ch=4, out_ch=4).to(device), "train": True},
        "FastICA": {"model": FastICABaseline(), "train": False},

        # "Linear": {"model": LinearSeparator(in_ch=8, out_ch=4).to(device), "train": True},
    }

    print("MOdels were tested")

    for name, entry in models_to_test.items():
        model = entry["model"]
        print(f"--- Training {name} ---")
        if entry["train"]:
            print(f"Model: {name} and parameters: {model.parameters()}")

            save_path = f"pytorch_models/{name}.pt"

            trained_model, t_hist, v_hist = train_model(
                model,
                train_loader,
                val_loader,
                plotter,
                epochs=ExperimentConfig.epochs,
                device=device,
                save_path=save_path
            )
            all_results[name] = evaluator.run_full_evaluation(trained_model, t_hist, v_hist, name)
        else:
            print(f"Model: {name} baseline (no training step)")
            all_results[name] = evaluator.run_full_evaluation(model, [], [], name)

    # Final visual duties
    evaluator.plot_comparison(all_results)
    evaluator.print_latex_table(all_results)

if __name__ == "__main__":
    main()

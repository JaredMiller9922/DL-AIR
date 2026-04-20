import torch
import os
import numpy as np 

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


DO_CROSS_VAL = False

def variance_to_snr_db(signal_power, variance):
    if variance is None or variance <= 0:
        return None
    return 10 * np.log10(signal_power / variance)

def get_model(config, device):
    """Instantiates the model based on the UI dropdown."""
    name = config.model_name.upper()
    if name == "HYBRID": return HybridSeparator(in_ch=8, out_ch=4).to(device) # Add dropout kwarg if supported
    elif name == "LSTM": return LSTMSeparator(in_ch=8, out_ch=4).to(device)
    elif name == "LINEAR": return LinearSeparator(in_ch=8, out_ch=4).to(device)
    elif name == "IQ_CNN": return IQCNNSeparator(in_ch=8, out_ch=4).to(device)
    elif name == "HTDEMUCS": return RFHTDemucsWrapper(in_ch=8, out_ch=4).to(device)
    else: raise ValueError(f"Unknown model: {config.model_name}")

def run_experiment(config: ExperimentConfig, ui_callback=None):
    """Bridge function called by the Streamlit UI."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    estimated_signal_power = 1.0 + config.alpha**2
    
    # 1. Pipeline Setup
    gen = RFMixtureGenerator(seed=0)
    qpsk_cfg = QPSKConfig(
        n_symbols=config.n_symbols, samples_per_symbol=config.samples_per_symbol,
        rolloff=config.rolloff, rrc_span_symbols=config.rrc_span_symbols
    )
    # --- Resolve noise parameters cleanly ---
    if not config.noise_enabled:
        snr_db = None
        variance = None
    else:
        if config.noise_variance is not None:
            variance = config.noise_variance
            snr_db = variance_to_snr_db(estimated_signal_power, variance)
        else:
            snr_db = config.snr_db
            variance = None

    noise_cfg = NoiseConfig(
        enabled=config.noise_enabled,
        snr_db=snr_db,
        variance=variance
    )

    mix_cfg = MixtureConfig(alpha=config.alpha, snr_db=snr_db)


    # 2. Evaluator Setup
    plotter = BeautifulRFPlotter(save_dir="../visualizations")
    rrc = rrc_taps(sps=config.samples_per_symbol, beta=config.rolloff, span_symbols=config.rrc_span_symbols)
    
    val_ds = SyntheticRFDataset(
        num_examples=200, generator=gen, qpsk_cfg_soi=qpsk_cfg, 
        qpsk_cfg_int=qpsk_cfg, noise_cfg=noise_cfg, mix_cfg=mix_cfg
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    
    evaluator = ModelEvaluator(val_loader, plotter, rrc_taps=rrc, sps=config.samples_per_symbol, device=device, log_dir="../logs")
    model = get_model(config, device)

    model.ui_callback = ui_callback

    # 3. Execution
    if config.mode == "evaluate":
        model.load_state_dict(torch.load(config.model_path, map_location=device))
        results = evaluator.run_full_evaluation(model, train_hist=[], val_hist=[], model_name=config.model_name)
        
        # Grab a signal sample for the interactive UI plots
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            pred = model(sample_batch["x"].to(device)).cpu().numpy()
            results["sample_signal"] = pred[0, 0, :] + 1j * pred[0, 1, :] 
        return results

    elif config.mode == "train":
        train_ds = SyntheticRFDataset(
            num_examples=2000, generator=gen, qpsk_cfg_soi=qpsk_cfg, 
            qpsk_cfg_int=qpsk_cfg, noise_cfg=noise_cfg, mix_cfg=mix_cfg
        )
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

        trained_model, t_hist, v_hist = train_model(
            model, train_loader, val_loader, plotter, 
            epochs=config.epochs, device=device, lr=config.lr
            # Add ui_callback=ui_callback to your train_model signature in train.py if you want live charts!
        )
        
        os.makedirs("../pytorch_models", exist_ok=True)
        torch.save(trained_model.state_dict(), f"../pytorch_models/{config.model_name.lower()}_model.pt")
        
        results = evaluator.run_full_evaluation(trained_model, t_hist, v_hist, config.model_name)
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            pred = trained_model(sample_batch["x"].to(device)).cpu().numpy()
            results["sample_signal"] = pred[0, 0, :] + 1j * pred[0, 1, :] 
        return results



def train_and_validate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if ExperimentConfig.use_on_the_fly_data:
        print("Using on the fly data generation")

        gen = RFMixtureGenerator(seed=0)

        qpsk_cfg_soi = QPSKConfig(
            n_symbols=ExperimentConfig.num_symbols,
            samples_per_symbol=ExperimentConfig.samples_per_symbol,
            rolloff=ExperimentConfig.rolloff,
            rrc_span_symbols=ExperimentConfig.rrc_span_symbols,
        )

        qpsk_cfg_int = QPSKConfig(
            n_symbols=ExperimentConfig.num_symbols,
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

    if ExperimentConfig.n_rx == 1:
        models_to_test = {
            "Hybrid": {"model": HybridSeparator(in_ch=ExperimentConfig.n_rx * 2, out_ch=4).to(device), "train": True},
            "LSTM": {"model": LSTMSeparator(in_ch=ExperimentConfig.n_rx * 2, out_ch=4).to(device), "train": True},
            "IQ_CNN": {"model": IQCNNSeparator(in_ch=ExperimentConfig.n_rx * 2, out_ch=4).to(device), "train": True},
            "HTDemucs": {"model": RFHTDemucsWrapper(in_ch=ExperimentConfig.n_rx * 2, out_ch=4).to(device), "train": True},
        }
    else:
        models_to_test = {
            # "FastICA": {"model": FastICABaseline(), "train": False},
            # "Hybrid": {"model": HybridSeparator(in_ch=ExperimentConfig.n_rx * 2, out_ch=4).to(device), "train": True},
            # "LSTM": {"model": LSTMSeparator(in_ch=ExperimentConfig.n_rx * 2, out_ch=4).to(device), "train": True},
            "IQ_CNN": {"model": IQCNNSeparator(in_ch=ExperimentConfig.n_rx * 2, out_ch=4).to(device), "train": True},
            # "HTDemucs": {"model": RFHTDemucsWrapper(in_ch=ExperimentConfig.n_rx * 2, out_ch=4).to(device), "train": True},
        }

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
    import argparse

    parser = argparse.ArgumentParser(description="RF Separation Pipeline")

    # --- Core Mode ---
    parser.add_argument("--mode", choices=["train", "evaluate", "generate"], default=None)

    # --- Model Definition ---
    parser.add_argument("--model", default="Hybrid")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)

    # --- Training ---
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_cross_val", action="store_true")

    # --- Signal / QPSK ---
    parser.add_argument("--modulation", type=str, default="QPSK")
    parser.add_argument("--n_symbols", type=int, default=4)
    parser.add_argument("--samples_per_symbol", type=int, default=2)
    parser.add_argument("--rolloff", type=float, default=0.25)
    parser.add_argument("--rrc_span_symbols", type=int, default=12)
    parser.add_argument("--normalize_power", action="store_true")
    parser.add_argument("--num_channels", type=int, default=4)

    # --- Time / Sampling ---
    parser.add_argument("--fs", type=float, default=1.0)
    parser.add_argument("--symbol_rate", type=float, default=None)
    parser.add_argument("--T1", type=float, default=0.0)
    parser.add_argument("--T2", type=float, default=None)

    # --- Mixture / Noise ---
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--snr_db", type=float, default=100.0)
    parser.add_argument("--n_rx", type=int, default=4)
    parser.add_argument("--random_phase", action="store_true")
    parser.add_argument("--noise_var", type=float, default=None)

    # --- Data ---
    parser.add_argument("--dataset_path", type=str, default="data")
    parser.add_argument("--use_on_the_fly_data", action="store_true")

    # --- Evaluation ---
    parser.add_argument("--custom_symbols", type=str, default=None)

    args = parser.parse_args()

    # 👉 If no CLI args → run original behavior
    if args.mode is None:
        train_and_validate()
    else:
        config = ExperimentConfig(
            mode=args.mode,
            model_name=args.model,
            model_path=args.model_path,
            dropout=args.dropout,

            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            use_cross_val=args.use_cross_val,

            modulation=args.modulation,
            n_symbols=args.n_symbols,
            samples_per_symbol=args.samples_per_symbol,
            rolloff=args.rolloff,
            rrc_span_symbols=args.rrc_span_symbols,
            normalize_power=args.normalize_power,
            num_channels=args.num_channels,

            fs=args.fs,
            symbol_rate=args.symbol_rate,
            T1=args.T1,
            T2=args.T2,

            alpha=args.alpha,
            noise_enabled=args.noise,
            snr_db=args.snr_db,
            n_rx=args.n_rx,
            random_phase=args.random_phase,
            noise_variance=args.noise_var,

            dataset_path=args.dataset_path,
            use_on_the_fly_data=args.use_on_the_fly_data,

            custom_symbols=args.custom_symbols,
        )

        results = run_experiment(config)

        if results:
            print("\n=== RESULTS ===")
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    print(f"{k}: {v}")

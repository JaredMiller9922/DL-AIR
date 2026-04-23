from pathlib import Path

import numpy as np
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
from utils.model_utils.symbol_utils import rrc_taps
from config import ExperimentConfig
from utils.data_utils.generator import (
    RFMixtureGenerator,
    QPSKConfig,
    NoiseConfig,
    MixtureConfig,
    SourceConfig,
    SourceFamilyConfig,
    mit_aligned_family_config,
)
from utils.data_utils.dataset import SyntheticRFDataset
from torch.utils.data import DataLoader


DO_CROSS_VAL = False
ROOT_DIR = Path(__file__).resolve().parent
LEARNED_MODEL_NAMES = ("Hybrid", "LSTM", "Linear", "IQ_CNN", "HTDemucs")
BASELINE_MODEL_NAMES = ("FastICA",)


def normalize_model_name(model_name):
    lookup = {
        "hybrid": "Hybrid",
        "lstm": "LSTM",
        "linear": "Linear",
        "iq_cnn": "IQ_CNN",
        "iqcnn": "IQ_CNN",
        "htdemucs": "HTDemucs",
        "rfhtdemucs": "HTDemucs",
        "fastica": "FastICA",
        "fast_ica": "FastICA",
    }
    key = str(model_name).replace("-", "_").replace(" ", "_").lower()
    if key not in lookup:
        raise ValueError(f"Unknown model: {model_name}")
    return lookup[key]


def supported_model_names(n_rx=2, include_baselines=True):
    names = list(LEARNED_MODEL_NAMES)
    if include_baselines and n_rx >= 2:
        names = list(BASELINE_MODEL_NAMES) + names
    return names


def model_is_trainable(model_name):
    return normalize_model_name(model_name) in LEARNED_MODEL_NAMES


def resolve_path(path):
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT_DIR / resolved
    return resolved


def checkpoint_candidates(model_name, model_dir=None):
    name = normalize_model_name(model_name)
    if not model_is_trainable(name):
        return []

    model_root = resolve_path(model_dir or "pytorch_models")
    report_root = ROOT_DIR / "reporting_pipeline" / "outputs" / "checkpoints"
    lower = name.lower()
    report_names = {
        "Hybrid": "hybrid_separator.pt",
        "LSTM": "lstm_separator.pt",
        "Linear": "linear_separator.pt",
        "IQ_CNN": "iq_cnn_separator.pt",
        "HTDemucs": "htdemucs_separator.pt",
    }
    names = [
        f"{lower}_model.pt",
        f"{name}_model.pt",
        f"{name}.pt",
        f"{lower}.pt",
    ]
    candidates = [model_root / filename for filename in names]
    report_name = report_names.get(name)
    if report_name:
        candidates.append(report_root / report_name)
    return candidates


def default_checkpoint_path(model_name, model_dir=None):
    candidates = checkpoint_candidates(model_name, model_dir)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def load_model_weights(model, model_path, device):
    resolved = resolve_path(model_path)
    if resolved is None:
        raise ValueError("A model_path is required for learned model evaluation.")
    if not resolved.exists():
        raise FileNotFoundError(f"No checkpoint found at {resolved}")

    checkpoint = torch.load(str(resolved), map_location=device)
    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if state_dict is None and isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        state_dict = checkpoint

    if any(str(key).startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    return resolved

def variance_to_snr_db(signal_power, variance):
    if variance is None or variance <= 0:
        return None
    return 10 * np.log10(signal_power / variance)

def get_model(config, device):
    """Instantiates the model based on the UI dropdown."""
    name = normalize_model_name(config.model_name)
    in_ch = config.n_rx * 2
    dropout = config.dropout
    if name == "FastICA":
        return FastICABaseline().to(device)
    if name == "Hybrid":
        return HybridSeparator(in_ch=in_ch, out_ch=4, dropout=dropout).to(device)
    if name == "LSTM":
        return LSTMSeparator(in_ch=in_ch, out_ch=4, dropout=dropout).to(device)
    if name == "Linear":
        return LinearSeparator(in_ch=in_ch, out_ch=4).to(device)
    if name == "IQ_CNN":
        return IQCNNSeparator(in_ch=in_ch, out_ch=4, dropout=dropout).to(device)
    if name == "HTDemucs":
        return RFHTDemucsWrapper(in_ch=in_ch, out_ch=4).to(device)
    raise ValueError(f"Unknown model: {config.model_name}")


def artifact_dir_for(config):
    save_dir = resolve_path(config.save_dir)
    if save_dir.name.endswith("-channel"):
        return save_dir
    return save_dir / f"{config.n_rx}-channel"


def build_signal_configs(config):
    estimated_signal_power = 1.0 + config.alpha**2
    qpsk_cfg = QPSKConfig(
        n_symbols=config.num_symbols,
        samples_per_symbol=config.samples_per_symbol,
        rolloff=config.rolloff,
        rrc_span_symbols=config.rrc_span_symbols,
        normalize_power=config.normalize_power,
        num_channels=config.n_rx,
    )

    if not config.noise_enabled:
        snr_db = None
        sigma2 = None
    elif config.noise_variance is not None:
        sigma2 = config.noise_variance
        snr_db = variance_to_snr_db(estimated_signal_power, sigma2)
    else:
        snr_db = config.snr_db
        sigma2 = None

    noise_cfg = NoiseConfig(
        enabled=config.noise_enabled,
        snr_db=snr_db,
        sigma2=sigma2,
    )
    mix_cfg = MixtureConfig(
        alpha=config.alpha,
        snr_db=snr_db,
        n_rx=config.n_rx,
        random_phase=config.random_phase,
        phase_shift_deg=config.phase_shift_deg,
        interference_phase_shift=config.interference_phase_shift,
        mixing_mode=config.mixing_mode,
        random_complex_mixing=config.random_complex_mixing,
        timing_offset=config.timing_offset,
        carrier_offset=config.carrier_offset,
        phase_mismatch_deg=config.phase_mismatch_deg,
        amplitude_imbalance_db=config.amplitude_imbalance_db,
    )
    return qpsk_cfg, noise_cfg, mix_cfg, snr_db


def build_family_config(config):
    if not config.source_family_mode and not config.mit_aligned:
        return None
    if config.mit_aligned:
        family_cfg = mit_aligned_family_config()
    else:
        family_cfg = SourceFamilyConfig()

    family_cfg.source_a_types = tuple(config.source_a_family or (config.source_a_type,))
    family_cfg.source_b_types = tuple(config.source_b_family or (config.source_b_type,))
    family_cfg.n_symbols_range = config.n_symbols_range or (config.num_symbols, config.num_symbols)
    family_cfg.samples_per_symbol_choices = config.samples_per_symbol_choices or (config.samples_per_symbol,)
    family_cfg.rolloff_range = config.rolloff_range or (config.rolloff, config.rolloff)
    family_cfg.rrc_span_symbols = config.rrc_span_symbols
    family_cfg.alpha_range = config.alpha_range or (config.alpha, config.alpha)
    family_cfg.snr_db_range = config.snr_db_range or (config.snr_db, config.snr_db)
    family_cfg.timing_offset_range = config.timing_offset_range or (config.timing_offset, config.timing_offset)
    family_cfg.carrier_offset_range = config.carrier_offset_range or (config.carrier_offset, config.carrier_offset)
    family_cfg.phase_mismatch_range_deg = config.phase_mismatch_range_deg or (
        config.phase_mismatch_deg,
        config.phase_mismatch_deg,
    )
    family_cfg.amplitude_imbalance_db_range = config.amplitude_imbalance_db_range or (
        config.amplitude_imbalance_db,
        config.amplitude_imbalance_db,
    )
    family_cfg.mixing_modes = (config.mixing_mode,)
    family_cfg.n_rx = config.n_rx
    family_cfg.normalize_power = config.normalize_power
    family_cfg.mit_aligned = config.mit_aligned
    return family_cfg


def build_synthetic_loader(config, num_examples, generator, qpsk_cfg, noise_cfg, mix_cfg, shuffle=False):
    source_a_cfg = SourceConfig(
        source_type=config.source_a_type,
        n_symbols=config.num_symbols,
        samples_per_symbol=config.samples_per_symbol,
        rolloff=config.rolloff,
        rrc_span_symbols=config.rrc_span_symbols,
        normalize_power=config.normalize_power,
        num_channels=config.n_rx,
    )
    source_b_cfg = SourceConfig(
        source_type=config.source_b_type,
        n_symbols=config.num_symbols,
        samples_per_symbol=config.samples_per_symbol,
        rolloff=config.rolloff,
        rrc_span_symbols=config.rrc_span_symbols,
        normalize_power=config.normalize_power,
        num_channels=config.n_rx,
    )
    family_cfg = build_family_config(config)
    dataset = SyntheticRFDataset(
        num_examples=num_examples,
        generator=generator,
        qpsk_cfg_soi=qpsk_cfg,
        qpsk_cfg_int=qpsk_cfg,
        noise_cfg=noise_cfg,
        mix_cfg=mix_cfg,
        custom_symbols=config.custom_symbols,
        source_a_cfg=source_a_cfg,
        source_b_cfg=source_b_cfg,
        family_cfg=family_cfg,
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle)


def collect_artifacts(config, model_name):
    plot_dir = artifact_dir_for(config)
    snapshot_prefixes = {
        "Hybrid": "HybridSeparator",
        "LSTM": "LSTMSeparator",
        "Linear": "LinearSeparator",
        "IQ_CNN": "IQCNNSeparator",
        "HTDemucs": "RFHTDemucsWrapper",
    }
    snapshot_prefix = snapshot_prefixes.get(model_name, model_name)
    return {
        "plot_dir": str(plot_dir),
        "data_pipeline": str(plot_dir / "data_pipeline_waves.png"),
        "separation": str(plot_dir / f"{model_name}_separation.png"),
        "source_a_symbols": str(plot_dir / f"{model_name}_SourceA_symbol_recovery.png"),
        "source_b_symbols": str(plot_dir / f"{model_name}_SourceB_symbol_recovery.png"),
        "training_snapshots": [
            str(path)
            for path in sorted(plot_dir.glob(f"{snapshot_prefix}_Train_Epoch_*_separation.png"))
        ],
    }

def run_experiment(config: ExperimentConfig, ui_callback=None):
    """Bridge function called by the Streamlit UI."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.model_name = normalize_model_name(config.model_name)
    if config.model_name == "FastICA" and config.n_rx < 2:
        raise ValueError("FastICA needs at least 2 receive antennas (4 I/Q channels).")

    # 1. Pipeline Setup
    gen = RFMixtureGenerator(seed=0)
    qpsk_cfg, noise_cfg, mix_cfg, resolved_snr_db = build_signal_configs(config)
    plot_dir = artifact_dir_for(config)
    log_dir = resolve_path(config.log_dir)
    model_dir = resolve_path(config.model_dir)
    plotter = BeautifulRFPlotter(save_dir=str(plot_dir))
    rrc = rrc_taps(
        sps=config.samples_per_symbol,
        beta=config.rolloff,
        span_symbols=config.rrc_span_symbols,
    )

    val_loader = build_synthetic_loader(
        config,
        num_examples=config.val_examples,
        generator=gen,
        qpsk_cfg=qpsk_cfg,
        noise_cfg=noise_cfg,
        mix_cfg=mix_cfg,
        shuffle=False,
    )

    if config.mode == "train":
        train_loader = build_synthetic_loader(
            config,
            num_examples=config.train_examples,
            generator=gen,
            qpsk_cfg=qpsk_cfg,
            noise_cfg=noise_cfg,
            mix_cfg=mix_cfg,
            shuffle=True,
        )
    else:
        train_loader = val_loader

    evaluator = ModelEvaluator(
        train_loader=train_loader,
        val_loader=val_loader,
        plotter=plotter,
        rrc_taps=rrc,
        sps=config.samples_per_symbol,
        device=device,
        log_dir=str(log_dir),
    )
    model = get_model(config, device)
    model.ui_callback = ui_callback

    # 3. Execution
    if config.mode == "evaluate":
        loaded_checkpoint = None
        if model_is_trainable(config.model_name):
            model_path = config.model_path or default_checkpoint_path(config.model_name, model_dir)
            loaded_checkpoint = load_model_weights(model, model_path, device)
        results = evaluator.run_full_evaluation(model, train_hist=[], val_hist=[], model_name=config.model_name)

        # Grab a signal sample for the interactive UI plots
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            pred = model(sample_batch["x"].to(device)).detach().cpu().numpy()
            results["sample_signal"] = pred[0, 0, :] + 1j * pred[0, 1, :]
        results["model_name"] = config.model_name
        results["mode"] = "evaluate"
        results["device"] = device
        results["n_rx"] = config.n_rx
        results["snr_db"] = resolved_snr_db
        results["checkpoint_path"] = str(loaded_checkpoint) if loaded_checkpoint else None
        results["artifacts"] = collect_artifacts(config, config.model_name)
        return results

    elif config.mode == "train":
        if not model_is_trainable(config.model_name):
            raise ValueError(f"{config.model_name} is a baseline and does not have trainable weights.")

        model_dir.mkdir(parents=True, exist_ok=True)
        save_path = model_dir / f"{config.model_name.lower()}_model.pt"
        trained_model, t_hist, v_hist = train_model(
            model,
            train_loader,
            val_loader,
            plotter,
            epochs=config.epochs,
            device=device,
            lr=config.lr,
            save_path=str(save_path),
        )

        results = evaluator.run_full_evaluation(trained_model, t_hist, v_hist, config.model_name)
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            pred = trained_model(sample_batch["x"].to(device)).detach().cpu().numpy()
            results["sample_signal"] = pred[0, 0, :] + 1j * pred[0, 1, :]
        results["model_name"] = config.model_name
        results["mode"] = "train"
        results["device"] = device
        results["n_rx"] = config.n_rx
        results["snr_db"] = resolved_snr_db
        results["checkpoint_path"] = str(save_path)
        results["artifacts"] = collect_artifacts(config, config.model_name)
        return results

    raise ValueError(f"Unsupported experiment mode: {config.mode}")



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
            "FastICA": {"model": FastICABaseline(), "train": False},
            "Hybrid": {"model": HybridSeparator(in_ch=ExperimentConfig.n_rx * 2, out_ch=4).to(device), "train": True},
            "LSTM": {"model": LSTMSeparator(in_ch=ExperimentConfig.n_rx * 2, out_ch=4).to(device), "train": True},
            "IQ_CNN": {"model": IQCNNSeparator(in_ch=ExperimentConfig.n_rx * 2, out_ch=4).to(device), "train": True},
            "HTDemucs": {"model": RFHTDemucsWrapper(in_ch=ExperimentConfig.n_rx * 2, out_ch=4).to(device), "train": True},
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
    parser.add_argument("--source_a_type", type=str, default=None)
    parser.add_argument("--source_b_type", type=str, default=None)
    parser.add_argument("--source_family_mode", action="store_true")
    parser.add_argument("--source_a_family", type=str, default=None)
    parser.add_argument("--source_b_family", type=str, default=None)
    parser.add_argument("--mit_aligned", action="store_true")
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
    parser.add_argument("--mixing_mode", type=str, default="phase_only")
    parser.add_argument("--timing_offset", type=int, default=0)
    parser.add_argument("--carrier_offset", type=float, default=0.0)
    parser.add_argument("--phase_mismatch_deg", type=float, default=0.0)
    parser.add_argument("--amplitude_imbalance_db", type=float, default=0.0)
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
            source_a_type=args.source_a_type or args.modulation,
            source_b_type=args.source_b_type or args.modulation,
            source_family_mode=args.source_family_mode,
            source_a_family=tuple(args.source_a_family.split(",")) if args.source_a_family else (args.source_a_type or args.modulation,),
            source_b_family=tuple(args.source_b_family.split(",")) if args.source_b_family else (args.source_b_type or args.modulation,),
            mit_aligned=args.mit_aligned,
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
            mixing_mode=args.mixing_mode,
            timing_offset=args.timing_offset,
            carrier_offset=args.carrier_offset,
            phase_mismatch_deg=args.phase_mismatch_deg,
            amplitude_imbalance_db=args.amplitude_imbalance_db,
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

import os
from pathlib import Path

import torch

from evaluate import ModelEvaluator
from networks.hybrid_separator import HybridSeparator
from networks.iq_cnn_separator import IQCNNSeparator
from networks.linear_separator import LinearSeparator
from networks.lstm_separator import LSTMSeparator
from train import train_model
from utils.data_utils.dataset import make_loader
from utils.model_utils.symbol_utils import rrc_taps
from utils.data_utils.generator import QPSKConfig
from utils.plot_utils.plotting_utils import BeautifulRFPlotter


def build_model_registry():
    return {
        "Hybrid": lambda: HybridSeparator(in_ch=8, out_ch=4),
        "LSTM": lambda: LSTMSeparator(in_ch=8, out_ch=4),
        "Linear": lambda: LinearSeparator(in_ch=8, out_ch=4),
        "IQ_CNN": lambda: IQCNNSeparator(
            in_ch=8,
            out_ch=4,
            base_channels=int(os.getenv("DLAIR_IQCNN_BASE", "32")),
            dropout=float(os.getenv("DLAIR_IQCNN_DROPOUT", "0.0")),
        ),
    }


def main():
    project_root = Path(__file__).resolve().parent
    data_root = project_root / "data"
    logs_dir = project_root / "logs"
    viz_dir = project_root / "visualizations"
    batch_size = int(os.getenv("DLAIR_BATCH_SIZE", "16"))
    epochs = int(os.getenv("DLAIR_EPOCHS", "200"))
    lr = float(os.getenv("DLAIR_LR", "1e-3"))
    weight_decay = float(os.getenv("DLAIR_WEIGHT_DECAY", "0.0"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, _ = make_loader(str(data_root / "train"), batch_size=batch_size, shuffle=True)
    val_loader, _ = make_loader(str(data_root / "val"), batch_size=batch_size)

    plotter = BeautifulRFPlotter(save_dir=str(viz_dir))
    qpsk_cfg_soi = QPSKConfig(n_symbols=400, samples_per_symbol=2, rolloff=0.25, rrc_span_symbols=12)
    rrc = rrc_taps(
        sps=qpsk_cfg_soi.samples_per_symbol,
        beta=qpsk_cfg_soi.rolloff,
        span_symbols=qpsk_cfg_soi.rrc_span_symbols,
    )
    evaluator = ModelEvaluator(
        val_loader,
        plotter,
        rrc_taps=rrc,
        sps=qpsk_cfg_soi.samples_per_symbol,
        device=device,
        log_dir=str(logs_dir),
    )

    all_results = {}
    model_registry = build_model_registry()

    print(f"Device: {device}")
    print(
        f"Comparing {len(model_registry)} models with shared training settings "
        f"(epochs={epochs}, batch_size={batch_size}, lr={lr}, weight_decay={weight_decay})."
    )

    for name, build_model in model_registry.items():
        try:
            model = build_model().to(device)
        except Exception as exc:
            print(f"Skipping {name}: incompatible with current pipeline ({exc})")
            continue

        print(f"--- Training {name} ---")
        trained_model, train_hist, val_hist, training_metadata = train_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
        )
        all_results[name] = evaluator.run_full_evaluation(
            trained_model,
            train_hist,
            val_hist,
            name,
            training_metadata=training_metadata,
        )

    if not all_results:
        raise RuntimeError("No compatible models were available for comparison.")

    evaluator.plot_comparison(all_results)
    evaluator.print_rankings(all_results)
    evaluator.print_latex_table(all_results)


if __name__ == "__main__":
    main()

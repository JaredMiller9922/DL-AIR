import torch

from train import train_model
from evaluate import ModelEvaluator
from networks.hybrid_separator import HybridSeparator
from networks.lstm_separator import LSTMSeparator
from networks.linear_separator import LinearSeparator
from networks.iq_cnn_separator import IQCNNSeparator
from utils.data_utils.dataset import make_loader
from utils.plot_utils.plotting_utils import BeautifulRFPlotter
from cross_validator import GridSearchManager
from utils.model_utils.symbol_utils import rrc_taps
from utils.data_utils.generator import QPSKConfig

DO_CROSS_VAL = False #SET TO FALSE IF YOU DO NOT WANT CROSS VALIDATION TO OCCUR

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, _ = make_loader("../data/train", batch_size=16, shuffle=True)
    val_loader, _ = make_loader("../data/val", batch_size=16)
    
    plotter = BeautifulRFPlotter(save_dir="../visualizations")

    qpsk_cfg_soi = QPSKConfig(
        n_symbols=400,
        samples_per_symbol=2,
        rolloff=0.25,
        rrc_span_symbols=12,
    )

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
    )

    
    all_results = {}
    print("WE MADE IT BEFORE THE Separator")

    models_to_test = {
        "Hybrid": HybridSeparator(in_ch=2, out_ch=4).to(device),
        "LSTM": LSTMSeparator(in_ch=2, out_ch=4).to(device),
        "Linear": LinearSeparator(in_ch=2, out_ch=4).to(device),
        "IQ_CNN": IQCNNSeparator(in_ch=2, out_ch=4).to(device)
    }

    print("MOdels were tested")

    if DO_CROSS_VAL:
        manager = GridSearchManager(HybridSeparator, train_loader, val_loader, evaluator)
        grid = {
            'lr': [1e-3, 5e-4],
            'dropout': [0, 0.2],
            'hidden': [64, 128],
            'epochs': [20]
        }
        manager.run_grid_search(grid)
    else:
        for name, model in models_to_test.items():
            print(f"--- Training {name} ---")
            print(f"Model: {name} and parameters: {model.parameters()}")
            trained_model, t_hist, v_hist = train_model(model, train_loader, val_loader, plotter, epochs=20, device=device)
            
            # This saves the JSON, plots the waves, and logs SDR
            all_results[name] = evaluator.run_full_evaluation(trained_model, t_hist, v_hist, name)

    # Final visual duties
    evaluator.plot_comparison(all_results)
    evaluator.print_latex_table(all_results)

if __name__ == "__main__":
    main()
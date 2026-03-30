import torch

from train import train_model
from evaluate import ModelEvaluator
from networks.hybrid_separator import HybridSeparator
from networks.lstm_separator import LSTMSeparator
from networks.linear_separator import LinearSeparator
from utils.data_utils.dataset import make_loader
from utils.plot_utils.plotting_utils import BeautifulRFPlotter
from cross_validator import GridSearchManager





DO_CROSS_VAL = True #SET TO FALSE IF YOU DO NOT WANT CROSS VALIDATION TO OCCUR




def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, _ = make_loader("../data/train", batch_size=16, shuffle=True)
    val_loader, _ = make_loader("../data/val", batch_size=16)
    
    plotter = BeautifulRFPlotter(save_dir="../visualizations")
    evaluator = ModelEvaluator(val_loader, plotter, device=device)
    
    all_results = {}
    print("WE MADE IT BEFORE THE Separator")

    models_to_test = {
        "Hybrid": HybridSeparator(in_ch=8, out_ch=4).to(device),
        "LSTM": LSTMSeparator(in_ch=8, out_ch=4).to(device),
        "Linear": LinearSeparator(in_ch=8, out_ch=4).to(device)
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
            trained_model, t_hist, v_hist = train_model(model, train_loader, val_loader, epochs=10, device=device)
            
            # This saves the JSON, plots the waves, and logs SDR
            all_results[name] = evaluator.run_full_evaluation(trained_model, t_hist, v_hist, name)

    # Final visual duties
    evaluator.plot_comparison(all_results)
    evaluator.print_latex_table(all_results)

if __name__ == "__main__":
    main()
import json
from pathlib import Path

import torch

from evaluate import ModelEvaluator
from networks.iq_cnn_separator import IQCNNSeparator
from networks.lstm_separator import LSTMSeparator
from train import train_model
from utils.data_utils.dataset import make_loader
from utils.plot_utils.plotting_utils import BeautifulRFPlotter


def build_models(device):
    return {
        "LSTM": lambda: LSTMSeparator(in_ch=8, out_ch=4, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.15).to(device),
        "IQ_CNN": lambda: IQCNNSeparator(in_ch=8, out_ch=4, base_channels=48, dropout=0.08).to(device),
    }


def main():
    project_root = Path(__file__).resolve().parent
    data_root = project_root / "data"
    logs_root = project_root / "logs"
    viz_root = project_root / "visualizations"
    ckpt_root = project_root / "checkpoints" / "lr_search"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    epochs = 400
    learning_rates = [0.0012, 0.0014, 0.0017, 0.0020]
    weight_decay = 1e-4
    grad_clip = 1.0

    train_loader, _ = make_loader(str(data_root / "train"), batch_size=batch_size, shuffle=True)
    val_loader, _ = make_loader(str(data_root / "val"), batch_size=batch_size)

    plotter = BeautifulRFPlotter(save_dir=str(viz_root))
    evaluator = ModelEvaluator(val_loader, plotter, device=device, log_dir=str(logs_root))

    model_builders = build_models(device)
    all_results = {}
    lr_search_results = {}

    estimated_seconds = len(model_builders) * len(learning_rates) * epochs * 10
    print(f"Device: {device}")
    print(f"Running narrow LR search for {list(model_builders.keys())} with batch_size={batch_size}, epochs={epochs}")
    print(f"Learning rates: {learning_rates}")
    print(f"Estimated runtime is roughly {estimated_seconds / 3600:.1f}+ hours based on prior epoch timings.")

    for model_name, build_model in model_builders.items():
        best_run = None
        lr_search_results[model_name] = []

        for lr in learning_rates:
            print(f"--- Training {model_name} with lr={lr:.4g} ---")
            model = build_model()
            checkpoint_path = ckpt_root / model_name / f"lr_{str(lr).replace('.', 'p')}.pt"
            trained_model, t_hist, v_hist, metadata = train_model(
                model,
                train_loader,
                val_loader,
                epochs=epochs,
                device=device,
                lr=lr,
                weight_decay=weight_decay,
                grad_clip=grad_clip,
                checkpoint_path=str(checkpoint_path),
            )

            run_result = {
                "lr": lr,
                "model": trained_model,
                "train_history": t_hist,
                "val_history": v_hist,
                "metadata": metadata,
                "checkpoint_path": str(checkpoint_path),
            }
            lr_search_results[model_name].append(
                {
                    "lr": lr,
                    "best_val_pit_mse": metadata["best_val_pit_mse"],
                    "best_epoch": metadata["best_epoch"],
                    "checkpoint_path": str(checkpoint_path),
                }
            )

            if best_run is None or metadata["best_val_pit_mse"] < best_run["metadata"]["best_val_pit_mse"]:
                best_run = run_result

        if best_run is None:
            continue

        print(
            f"Best {model_name} learning rate: {best_run['lr']:.4g} "
            f"(best val PIT-MSE {best_run['metadata']['best_val_pit_mse']:.5f} at epoch {best_run['metadata']['best_epoch']})"
        )
        all_results[model_name] = evaluator.run_full_evaluation(
            best_run["model"],
            best_run["train_history"],
            best_run["val_history"],
            model_name,
            training_metadata=best_run["metadata"],
        )

    logs_root.mkdir(parents=True, exist_ok=True)
    with open(logs_root / "lr_search_results.json", "w") as handle:
        json.dump(lr_search_results, handle, indent=4)

    evaluator.plot_comparison(all_results)
    evaluator.print_latex_table(all_results)

    best_overall = min(all_results.items(), key=lambda item: item[1]["best_val_pit_mse"])
    print(
        f"\nBest overall model: {best_overall[0]} "
        f"with best validation PIT-MSE {best_overall[1]['best_val_pit_mse']:.5f}"
    )


if __name__ == "__main__":
    main()

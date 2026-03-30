import torch
import json
import os
import matplotlib.pyplot as plt
from utils.model_utils.losses import align_to_pit_target, pit_mse_loss, pit_sdr


def normalize_batch(x, y, y_alt=None, eps=1e-8):
    scale = torch.sqrt(torch.mean(x ** 2, dim=(1, 2), keepdim=True) + eps)
    x = x / scale
    y = y / scale
    if y_alt is not None:
        y_alt = y_alt / scale
    return x, y, y_alt

class ModelEvaluator:
    def __init__(self, val_loader, plotter, device = "cuda" if torch.cuda.is_available() else "cpu", log_dir="../logs"):
        self.val_loader = val_loader
        self.plotter = plotter
        self.device = device
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def run_full_evaluation(self, model, train_hist, val_hist, model_name, training_metadata=None):
        model.eval()
        total_mse, total_sdr = 0, 0
        normalize_inputs = bool(training_metadata and training_metadata.get("normalize_inputs"))
        
        # 1. Generate Wave Plots (Squiggly Lines)
        # We take the very first batch to show the "Success" visual
        with torch.no_grad():
            batch = next(iter(self.val_loader))
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            y_alt = batch.get("y_alt")
            if y_alt is not None:
                y_alt = y_alt.to(self.device)
            if normalize_inputs:
                x, y, y_alt = normalize_batch(x, y, y_alt)
            pred = model(x)
            aligned_y = align_to_pit_target(pred, y, y_alt)
            
            self.plotter.plot_data_pipeline(x, aligned_y) 
            self.plotter.plot_separation_performance(aligned_y, pred, model_name=model_name)

        # 2. Calculate Final Metrics (JSON Logs)
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                y_alt = batch.get("y_alt")
                if y_alt is not None:
                    y_alt = y_alt.to(self.device)
                if normalize_inputs:
                    x, y, y_alt = normalize_batch(x, y, y_alt)
                pred = model(x)
                total_mse += pit_mse_loss(pred, y, y_alt).item()
                total_sdr += pit_sdr(pred, y, y_alt).item()
        
        metrics = {
            "final_mse": total_mse / len(self.val_loader),
            "final_sdr_db": total_sdr / len(self.val_loader),
            "train_history": train_hist,
            "val_history": val_hist
        }

        if training_metadata:
            metrics.update(training_metadata)

        with open(os.path.join(self.log_dir, f"{model_name}_results.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        
        return metrics

    def plot_comparison(self, all_metrics):
        """Plots all model validation curves on one graph."""
        plt.figure(figsize=(10, 6))
        for name, m in all_metrics.items():
            plt.plot(m["val_history"], label=f"{name} (Val MSE)")
        plt.title("Model Performance Comparison")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.savefig("../visualizations/model_comparison.png")
        plt.close()

    def print_latex_table(self, all_metrics):
        print("\n% Copy this into your LaTeX report:")
        print(r"\begin{table}[h]\centering")
        print(r"\begin{tabular}{lcc}")
        print(r"\hline \textbf{Model} & \textbf{Final MSE} & \textbf{SDR (dB)} \\ \hline")
        for name, m in all_metrics.items():
            print(f"{name} & {m['final_mse']:.5f} & {m['final_sdr_db']:.2f} \\\\")
        print(r"\hline \end{tabular} \end{table}")

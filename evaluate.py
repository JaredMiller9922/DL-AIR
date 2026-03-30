import torch
import json
import os
import matplotlib.pyplot as plt
from utils.model_utils.losses import calculate_sdr, mse_loss

class ModelEvaluator:
    def __init__(self, val_loader, plotter, device = "cuda" if torch.cuda.is_available() else "cpu", log_dir="../logs"):
        self.val_loader = val_loader
        self.plotter = plotter
        self.device = device
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def run_full_evaluation(self, model, train_hist, val_hist, model_name):
        model.eval()
        total_mse, total_sdr = 0, 0
        
        # 1. Generate Wave Plots (Squiggly Lines)
        # We take the very first batch to show the "Success" visual
        with torch.no_grad():
            batch = next(iter(self.val_loader))
            x, y = batch["x"].to(self.device), batch["y"].to(self.device)
            pred = model(x)
            
            self.plotter.plot_data_pipeline(x, y) 
            self.plotter.plot_separation_performance(y, pred, model_name=model_name)
            if "meta" in batch and "srcA_symbols" in batch["meta"]:
                # We need to convert from torch/tensor back to numpy for the plotter
                syms = batch["meta"]["srcA_symbols"][0].cpu().numpy() 
                wave = y[0, :2, :].cpu().numpy() # Taking first 2 channels (I/Q) of Source A
                # Convert I/Q back to complex for the function
                complex_wave = wave[0] + 1j*wave[1]
                self.plotter.plot_modulation_process(syms, complex_wave, model_name=model_name)

        # 2. Calculate Final Metrics (JSON Logs)
        with torch.no_grad():
            for batch in self.val_loader:
                x, y = batch["x"].to(self.device), batch["y"].to(self.device)
                pred = model(x)
                total_mse += mse_loss(pred, y).item()
                total_sdr += calculate_sdr(pred, y).item()
        
        metrics = {
            "final_mse": total_mse / len(self.val_loader),
            "final_sdr_db": total_sdr / len(self.val_loader),
            "train_history": train_hist,
            "val_history": val_hist
        }

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
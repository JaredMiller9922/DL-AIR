import json
import os
import time

import matplotlib.pyplot as plt
import torch

from utils.model_utils.losses import align_to_pit_target, pit_mse_loss, pit_sdr


class ModelEvaluator:
    def __init__(self, val_loader, plotter, device="cuda" if torch.cuda.is_available() else "cpu", log_dir="../logs"):
        self.val_loader = val_loader
        self.plotter = plotter
        self.device = device
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def run_full_evaluation(self, model, train_hist, val_hist, model_name, training_metadata=None):
        model.eval()
        total_pit_mse = 0.0
        total_pit_sdr = 0.0
        total_batches = 0
        total_examples = 0
        inference_time = 0.0

        with torch.no_grad():
            batch = next(iter(self.val_loader))
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            y_alt = batch.get("y_alt")
            if y_alt is not None:
                y_alt = y_alt.to(self.device)

            pred = model(x)
            aligned_y = align_to_pit_target(pred, y, y_alt)
            self.plotter.plot_data_pipeline(x, aligned_y)
            self.plotter.plot_separation_performance(aligned_y, pred, model_name=model_name)

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                y_alt = batch.get("y_alt")
                if y_alt is not None:
                    y_alt = y_alt.to(self.device)

                start = time.perf_counter()
                pred = model(x)
                inference_time += time.perf_counter() - start

                total_pit_mse += pit_mse_loss(pred, y, y_alt).item()
                total_pit_sdr += pit_sdr(pred, y, y_alt).item()
                total_batches += 1
                total_examples += x.shape[0]

        metrics = {
            "primary_metric": "val_pit_mse",
            "val_pit_mse": total_pit_mse / total_batches,
            "val_pit_sdr_db": total_pit_sdr / total_batches,
            "final_mse": total_pit_mse / total_batches,
            "final_sdr_db": total_pit_sdr / total_batches,
            "train_history": train_hist,
            "val_history": val_hist,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "avg_inference_ms_per_batch": (inference_time / total_batches) * 1000.0,
            "examples_evaluated": total_examples,
        }

        if training_metadata:
            metrics.update(training_metadata)

        with open(os.path.join(self.log_dir, f"{model_name}_results.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        return metrics

    def plot_comparison(self, all_metrics):
        plt.figure(figsize=(10, 6))
        for name, metrics in all_metrics.items():
            plt.plot(metrics["val_history"], label=f"{name} (Val PIT-MSE)")
        plt.title("Model Performance Comparison")
        plt.xlabel("Epoch")
        plt.ylabel("PIT-MSE")
        plt.legend()
        plt.savefig(os.path.join(self.plotter.save_dir, "model_comparison.png"))
        plt.close()

    def print_rankings(self, all_metrics):
        ranked = sorted(
            all_metrics.items(),
            key=lambda item: (item[1]["val_pit_mse"], -item[1]["val_pit_sdr_db"]),
        )

        print("\nModel ranking (best first):")
        for idx, (name, metrics) in enumerate(ranked, start=1):
            print(
                f"{idx}. {name}: PIT-MSE={metrics['val_pit_mse']:.5f}, "
                f"PIT-SDR={metrics['val_pit_sdr_db']:.2f} dB, "
                f"params={metrics['parameter_count']:,}"
            )

        best_name, best_metrics = ranked[0]
        print(
            f"\nBest model: {best_name} "
            f"(PIT-MSE={best_metrics['val_pit_mse']:.5f}, "
            f"PIT-SDR={best_metrics['val_pit_sdr_db']:.2f} dB)"
        )

    def print_latex_table(self, all_metrics):
        print("\n% Copy this into your LaTeX report:")
        print(r"\begin{table}[h]\centering")
        print(r"\begin{tabular}{lcccc}")
        print(r"\hline \textbf{Model} & \textbf{Val PIT-MSE} & \textbf{PIT-SDR (dB)} & \textbf{Params} & \textbf{Best Epoch} \\ \hline")
        for name, metrics in sorted(all_metrics.items(), key=lambda item: item[1]["val_pit_mse"]):
            print(
                f"{name} & {metrics['val_pit_mse']:.5f} & {metrics['val_pit_sdr_db']:.2f} "
                f"& {metrics['parameter_count']} & {metrics.get('best_epoch', '-') } \\\\"
            )
        print(r"\hline \end{tabular} \end{table}")

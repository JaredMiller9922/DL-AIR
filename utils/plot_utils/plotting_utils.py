import os
import torch
import numpy as np
import matplotlib.pyplot as plt

class BeautifulRFPlotter:
    def __init__(self, save_dir="visualizations"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        # Use a clean, professional aesthetic
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    def plot_data_pipeline(self, x, y, batch_idx=0, num_samples=200, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        Plots the raw mixture against the true sources for a subset of time.
        x: (B, 2*n_rx, T) - Received mixtures
        y: (B, 4, T) - True sources [srcA_I, srcA_Q, srcB_I, srcB_Q]
        """

        # Extract the first sequence in the batch and limit the time dimension for visibility
        x_np = x[batch_idx, :, :num_samples].cpu().numpy()
        y_np = y[batch_idx, :, :num_samples].cpu().numpy()

        time = np.arange(num_samples)

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("Synthetic RF Pipeline: Mixture vs. Sources", fontsize=16, fontweight='bold')

        # 1. Plot Mixture (Antenna 0 only for clarity)
        axes[0].plot(time, x_np[0], label="Antenna 0 Mixture (I)", color='purple', alpha=0.8)
        axes[0].plot(time, x_np[1], label="Antenna 0 Mixture (Q)", color='purple', linestyle='--', alpha=0.6)
        axes[0].set_title("Received Mixture (Antenna 0)")
        axes[0].legend(loc="upper right")

        # 2. Plot Source A
        axes[1].plot(time, y_np[0], label="Source A (I)", color=self.colors[0], linewidth=2)
        axes[1].plot(time, y_np[1], label="Source A (Q)", color=self.colors[0], linestyle='--', linewidth=2)
        axes[1].set_title("Ground Truth: Source A (QPSK)")
        axes[1].legend(loc="upper right")

        # 3. Plot Source B
        axes[2].plot(time, y_np[2], label="Source B (I)", color=self.colors[1], linewidth=2)
        axes[2].plot(time, y_np[3], label="Source B (Q)", color=self.colors[1], linestyle='--', linewidth=2)
        axes[2].set_title("Ground Truth: Source B (Interferer)")
        axes[2].legend(loc="upper right")

        plt.xlabel("Time Samples")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "data_pipeline_waves.png"), dpi=300)
        plt.close()

    def plot_separation_performance(self, y_true, y_pred, model_name, batch_idx=0, num_samples=200):
        """
        Overlays the model's predictions on top of the ground truth sources.
        """
        y_true_np = y_true[batch_idx, :, :num_samples].cpu().detach().numpy()
        y_pred_np = y_pred[batch_idx, :, :num_samples].cpu().detach().numpy()
        time = np.arange(num_samples)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Separation Performance: {model_name}", fontsize=16, fontweight='bold')

        # Plot Source A Comparison (I channel only for cleaner visualization)
        axes[0].plot(time, y_true_np[0], label="True Source A (I)", color='black', alpha=0.6, linewidth=3)
        axes[0].plot(time, y_pred_np[0], label="Predicted Source A (I)", color=self.colors[0], linestyle='--', linewidth=2)
        axes[0].set_title("Source A Recovery (In-Phase)")
        axes[0].legend(loc="upper right")

        # Plot Source B Comparison (I channel only)
        axes[1].plot(time, y_true_np[2], label="True Source B (I)", color='black', alpha=0.6, linewidth=3)
        axes[1].plot(time, y_pred_np[2], label="Predicted Source B (I)", color=self.colors[1], linestyle='--', linewidth=2)
        axes[1].set_title("Source B Recovery (In-Phase)")
        axes[1].legend(loc="upper right")

        plt.xlabel("Time Samples")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{model_name}_separation.png"), dpi=300)
        plt.close()
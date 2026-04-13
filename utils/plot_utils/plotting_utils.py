import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
        axes[0].set_title("Received Mixture")
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

    def plot_modulation_process(self, symbols, wave, model_name="QPSK"):
        """
        Plots the mapping from digital symbols to the analog I/Q wave.
        symbols: complex ndarray (the raw QPSK points)
        wave: complex ndarray (the pulse-shaped wave)
        """
        num_symbols = 20
        sps = len(wave) // len(symbols)
        time_wave = np.arange(num_symbols * sps)
        time_syms = np.arange(0, num_symbols * sps, sps)

        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        
        # 1. Digital Symbols (In-Phase)
        ax[0].step(time_syms, symbols[:num_symbols].real, where='post', 
                   label="Digital Symbols (I)", color='red', alpha=0.5, linewidth=2)
        ax[0].plot(time_wave, wave[:num_symbols*sps].real, 
                   label="Modulated Wave (I)", color=self.colors[0], linewidth=2)
        ax[0].set_title(f"{model_name}: Digital Symbols to Pulse-Shaped Wave")
        ax[0].legend()

        # 2. Constellation Diagram (The "Proof" of QPSK)
        ax[1] = fig.add_subplot(2, 2, 3) # Re-arranging for a constellation spot
        ax[1].scatter(symbols.real, symbols.imag, c='red', label="Symbols")
        ax[1].scatter(wave.real, wave.imag, c=self.colors[0], alpha=0.1, s=1, label="Wave Samples")
        ax[1].set_title("Constellation Mapping")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{model_name}_modulation.png"))
        plt.close()


    def plot_symbol_recovery(self, true_symbols, recovered_symbols, model_name="model", num_symbols=40):
        """
        true_symbols: complex ndarray, shape (N,)
        recovered_symbols: complex ndarray, shape (N,)
        """
        n = min(num_symbols, len(true_symbols), len(recovered_symbols))
        t = np.arange(n)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # 1. Constellation plot
        axes[0].scatter(true_symbols.real, true_symbols.imag, label="True Symbols", alpha=0.7)
        axes[0].scatter(recovered_symbols.real, recovered_symbols.imag, label="Recovered Symbols", alpha=0.7, marker="x")
        axes[0].set_title(f"{model_name}: Symbol Recovery Constellation")
        axes[0].set_xlabel("In-Phase")
        axes[0].set_ylabel("Quadrature")
        axes[0].legend()
        axes[0].grid(True)

        # 2. Symbol sequence plot (I channel)
        axes[1].step(t, true_symbols[:n].real, where="post", label="True Symbol I", linewidth=2)
        axes[1].step(t, recovered_symbols[:n].real, where="post", label="Recovered Symbol I", linewidth=2, linestyle="--")
        axes[1].set_title(f"{model_name}: First {n} Symbol Decisions")
        axes[1].set_xlabel("Symbol Index")
        axes[1].set_ylabel("I Value")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{model_name}_symbol_recovery.png"), dpi=300)
        plt.close()

# --- UI Specific Plotting Helpers ---

def plot_3d_wave(signal, title="Wave"):
    """
    signal: complex ndarray
    """
    t = np.arange(len(signal))

    fig = go.Figure(data=[go.Scatter3d(
        x=t,
        y=signal.real,
        z=signal.imag,
        mode='lines',
        line=dict(color=t, colorscale='Viridis', width=4)
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Time',
            yaxis_title='In-Phase (I)',
            zaxis_title='Quadrature (Q)'
        )
    )
    return fig


def plot_spectrogram(signal, fs=1.0):
    """
    signal: complex ndarray
    """
    fig, ax = plt.subplots()

    ax.specgram(signal.real, NFFT=256, Fs=fs, cmap='magma')
    ax.set_title("Signal Spectrogram (Real Component)")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Time")

    return fig
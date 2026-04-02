import torch
import json
import os
import matplotlib.pyplot as plt
from utils.model_utils.losses import calculate_sdr, mse_loss
from utils.model_utils.symbol_utils import symbol_accuracy, recover_symbols_from_waveform

class ModelEvaluator:
    def __init__(
        self,
        val_loader,
        plotter,
        rrc_taps,
        sps,
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_dir="../logs",
    ):
        self.val_loader = val_loader
        self.plotter = plotter
        self.rrc_taps = rrc_taps
        self.sps = sps
        self.device = device
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def run_full_evaluation(self, model, train_hist, val_hist, model_name):
        model.eval()
        total_mse, total_sdr, total_sym_acc = 0, 0, 0
        total_examples = 0
        
        # 1. Generate Wave Plots (Squiggly Lines)
        # We take the very first batch to show the "Success" visual
        with torch.no_grad():
            batch = next(iter(self.val_loader))
            x, y = batch["x"].to(self.device), batch["y"].to(self.device)
            pred = model(x)
            
            self.plotter.plot_data_pipeline(x, y) 
            self.plotter.plot_separation_performance(y, pred, model_name=model_name)

            # --- Symbol Recovery Plot (FIRST SAMPLE ONLY) ---
            pred_np = pred.cpu().numpy()

            true_sym_a = batch["symbols_a"].cpu().numpy()
            true_sym_b = batch["symbols_b"].cpu().numpy()

            pred_a = pred_np[:, 0, :] + 1j * pred_np[:, 1, :]
            pred_b = pred_np[:, 2, :] + 1j * pred_np[:, 3, :]

            true_sym_a_c = true_sym_a[:, 0, :] + 1j * true_sym_a[:, 1, :]
            true_sym_b_c = true_sym_b[:, 0, :] + 1j * true_sym_b[:, 1, :]

            # Just take first example in batch
            i = 0

            n_symbols_a = len(true_sym_a_c[i])
            n_symbols_b = len(true_sym_b_c[i])

            rec_a = recover_symbols_from_waveform(
                pred_a[i],
                self.rrc_taps,
                self.sps,
                n_symbols_a,
            )

            rec_b = recover_symbols_from_waveform(
                pred_b[i],
                self.rrc_taps,
                self.sps,
                n_symbols_b,
            )

            # Plot Source A recovery
            self.plotter.plot_symbol_recovery(
                true_symbols=true_sym_a_c[i],
                recovered_symbols=rec_a,
                model_name=f"{model_name}_SourceA"
            )

            # Plot Source B recovery
            self.plotter.plot_symbol_recovery(
                true_symbols=true_sym_b_c[i],
                recovered_symbols=rec_b,
                model_name=f"{model_name}_SourceB"
            )

        # 2. Calculate Final Metrics (JSON Logs)
        with torch.no_grad():
            for batch in self.val_loader:
                x, y = batch["x"].to(self.device), batch["y"].to(self.device)
                pred = model(x)

                pred_np = pred.cpu().numpy()

                # True transmitted symbols
                true_sym_a = batch["symbols_a"].cpu().numpy()   # (B, 2, Nsym)
                true_sym_b = batch["symbols_b"].cpu().numpy()   # (B, 2, Nsym)

                # Predicted separated waveforms -> complex
                pred_a = pred_np[:, 0, :] + 1j * pred_np[:, 1, :]
                pred_b = pred_np[:, 2, :] + 1j * pred_np[:, 3, :]

                # True symbols -> complex
                true_sym_a_c = true_sym_a[:, 0, :] + 1j * true_sym_a[:, 1, :]
                true_sym_b_c = true_sym_b[:, 0, :] + 1j * true_sym_b[:, 1, :]

                # Recover symbols and score
                for i in range(pred_a.shape[0]):
                    n_symbols_a = len(true_sym_a_c[i])
                    n_symbols_b = len(true_sym_b_c[i])

                    rec_a = recover_symbols_from_waveform(
                        pred_a[i],
                        self.rrc_taps,
                        self.sps,
                        n_symbols_a,
                    )
                    rec_b = recover_symbols_from_waveform(
                        pred_b[i],
                        self.rrc_taps,
                        self.sps,
                        n_symbols_b,
                    )

                    # trim in case recovery returns slightly different length
                    n_a = min(len(rec_a), len(true_sym_a_c[i]))
                    n_b = min(len(rec_b), len(true_sym_b_c[i]))

                    acc_a = symbol_accuracy(rec_a[:n_a], true_sym_a_c[i][:n_a])
                    acc_b = symbol_accuracy(rec_b[:n_b], true_sym_b_c[i][:n_b])

                    total_sym_acc += 0.5 * (acc_a + acc_b)
                    total_examples += 1

                total_mse += mse_loss(pred, y).item()
                total_sdr += calculate_sdr(pred, y).item()

        metrics = {
            "final_mse": total_mse / len(self.val_loader),
            "final_sdr_db": total_sdr / len(self.val_loader),
            "final_symbol_accuracy": total_sym_acc / total_examples if total_examples > 0 else 0.0,
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
        print(r"\begin{tabular}{lccc}")
        print(r"\hline \textbf{Model} & \textbf{Final MSE} & \textbf{SDR (dB)} & \textbf{Sym Acc} \\ \hline")
        for name, m in all_metrics.items():
            print(f"{name} & {m['final_mse']:.5f} & {m['final_sdr_db']:.2f} & {m['final_symbol_accuracy']:.4f} \\\\")
        print(r"\hline")
        print(r"\end{tabular}")
        print(r"\end{table}")
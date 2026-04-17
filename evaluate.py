import torch
import json
import os
import matplotlib.pyplot as plt
from utils.model_utils.losses import calculate_sdr, mse_loss
from utils.model_utils.symbol_utils import symbol_accuracy, recover_symbols_from_waveform


class ModelEvaluator:
    def __init__(
        self,
        train_loader,
        val_loader,
        plotter,
        rrc_taps,
        sps,
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_dir="logs",
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.plotter = plotter
        self.rrc_taps = rrc_taps
        self.sps = sps
        self.device = device
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def _compute_metrics_on_loader(self, model, loader):
        model.eval()
        total_mse = 0.0
        total_sdr = 0.0
        total_sym_acc = 0.0
        total_examples = 0

        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)

                pred = model(x)
                pred_np = pred.detach().cpu().numpy()

                true_sym_a = batch["symbols_a"].cpu().numpy()   # (B, 2, Nsym)
                true_sym_b = batch["symbols_b"].cpu().numpy()   # (B, 2, Nsym)

                pred_a = pred_np[:, 0, :] + 1j * pred_np[:, 1, :]
                pred_b = pred_np[:, 2, :] + 1j * pred_np[:, 3, :]

                true_sym_a_c = true_sym_a[:, 0, :] + 1j * true_sym_a[:, 1, :]
                true_sym_b_c = true_sym_b[:, 0, :] + 1j * true_sym_b[:, 1, :]

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

                    # direct assignment
                    n_aa = min(len(rec_a), len(true_sym_a_c[i]))
                    n_bb = min(len(rec_b), len(true_sym_b_c[i]))
                    acc_direct_a = symbol_accuracy(rec_a[:n_aa], true_sym_a_c[i][:n_aa])
                    acc_direct_b = symbol_accuracy(rec_b[:n_bb], true_sym_b_c[i][:n_bb])
                    direct_score = 0.5 * (acc_direct_a + acc_direct_b)

                    # swapped assignment removes symbol ambiguity
                    n_ab = min(len(rec_a), len(true_sym_b_c[i]))
                    n_ba = min(len(rec_b), len(true_sym_a_c[i]))
                    acc_swap_a = symbol_accuracy(rec_a[:n_ab], true_sym_b_c[i][:n_ab])
                    acc_swap_b = symbol_accuracy(rec_b[:n_ba], true_sym_a_c[i][:n_ba])
                    swap_score = 0.5 * (acc_swap_a + acc_swap_b)

                    total_sym_acc += max(direct_score, swap_score)
                    total_examples += 1

                total_mse += mse_loss(pred, y).item()
                total_sdr += calculate_sdr(pred, y).item()

        num_batches = len(loader)
        return {
            "mse": total_mse / num_batches if num_batches > 0 else 0.0,
            "sdr_db": total_sdr / num_batches if num_batches > 0 else 0.0,
            "symbol_accuracy": total_sym_acc / total_examples if total_examples > 0 else 0.0,
        }

    def run_full_evaluation(self, model, train_hist, val_hist, model_name):
        model.eval()

        # -----------------------------
        # 1. Validation plots only
        # -----------------------------
        with torch.no_grad():
            batch = next(iter(self.val_loader))
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            pred = model(x)

            self.plotter.plot_data_pipeline(x, y)
            self.plotter.plot_separation_performance(y, pred, model_name=model_name)

            pred_np = pred.detach().cpu().numpy()

            true_sym_a = batch["symbols_a"].cpu().numpy()
            true_sym_b = batch["symbols_b"].cpu().numpy()

            pred_a = pred_np[:, 0, :] + 1j * pred_np[:, 1, :]
            pred_b = pred_np[:, 2, :] + 1j * pred_np[:, 3, :]

            true_sym_a_c = true_sym_a[:, 0, :] + 1j * true_sym_a[:, 1, :]
            true_sym_b_c = true_sym_b[:, 0, :] + 1j * true_sym_b[:, 1, :]

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

            # score both assignments for first example
            n_aa = min(len(rec_a), len(true_sym_a_c[i]))
            n_bb = min(len(rec_b), len(true_sym_b_c[i]))
            direct_score = 0.5 * (
                symbol_accuracy(rec_a[:n_aa], true_sym_a_c[i][:n_aa]) +
                symbol_accuracy(rec_b[:n_bb], true_sym_b_c[i][:n_bb])
            )

            n_ab = min(len(rec_a), len(true_sym_b_c[i]))
            n_ba = min(len(rec_b), len(true_sym_a_c[i]))
            swap_score = 0.5 * (
                symbol_accuracy(rec_a[:n_ab], true_sym_b_c[i][:n_ab]) +
                symbol_accuracy(rec_b[:n_ba], true_sym_a_c[i][:n_ba])
            )

            if swap_score > direct_score:
                plot_true_a = true_sym_b_c[i]
                plot_true_b = true_sym_a_c[i]
            else:
                plot_true_a = true_sym_a_c[i]
                plot_true_b = true_sym_b_c[i]

            self.plotter.plot_symbol_recovery(
                true_symbols=plot_true_a,
                recovered_symbols=rec_a,
                model_name=f"{model_name}_SourceA"
            )

            self.plotter.plot_symbol_recovery(
                true_symbols=plot_true_b,
                recovered_symbols=rec_b,
                model_name=f"{model_name}_SourceB"
            )

        # -----------------------------
        # 2. Compute train + val metrics
        # -----------------------------
        train_metrics = self._compute_metrics_on_loader(model, self.train_loader)
        val_metrics = self._compute_metrics_on_loader(model, self.val_loader)

        metrics = {
            "train_mse": train_metrics["mse"],
            "train_sdr_db": train_metrics["sdr_db"],
            "train_symbol_accuracy": train_metrics["symbol_accuracy"],
            "final_mse": val_metrics["mse"],
            "final_sdr_db": val_metrics["sdr_db"],
            "final_symbol_accuracy": val_metrics["symbol_accuracy"],
            "val_symbol_accuracy": val_metrics["symbol_accuracy"],
            "train_history": train_hist,
            "val_history": val_hist,
        }

        with open(os.path.join(self.log_dir, f"{model_name}_results.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        print(
            f"{model_name} | "
            f"Train Sym Acc: {metrics['train_symbol_accuracy']:.3f} | "
            f"Val Sym Acc: {metrics['val_symbol_accuracy']:.3f}"
        )

        return metrics

    def plot_comparison(self, all_metrics):
        plt.figure(figsize=(10, 6))
        for name, m in all_metrics.items():
            if len(m["val_history"]) > 0:
                plt.plot(m["val_history"], label=f"{name} (Val MSE)")
        plt.title("Model Performance Comparison")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.savefig("visualizations/model_comparison.png")
        plt.close()

    def print_latex_table(self, all_metrics):
        print("\n% Copy this into your LaTeX report:")
        print(r"\begin{table}[h]\centering")
        print(r"\begin{tabular}{lcccc}")
        print(r"\hline \textbf{Model} & \textbf{Final MSE} & \textbf{SDR (dB)} & \textbf{Train Sym Acc} & \textbf{Val Sym Acc} \\ \hline")
        for name, m in all_metrics.items():
            print(
                f"{name} & "
                f"{m['final_mse']:.5f} & "
                f"{m['final_sdr_db']:.2f} & "
                f"{m['train_symbol_accuracy']:.4f} & "
                f"{m['val_symbol_accuracy']:.4f} \\\\"
            )
        print(r"\hline")
        print(r"\end{tabular}")
        print(r"\end{table}")
import json
import os
import time

import matplotlib.pyplot as plt
import torch

from utils.model_utils.losses import align_to_pit_target, mse_loss, pit_mse_loss, pit_sdr

try:
    from utils.model_utils.symbol_utils import recover_symbols_from_waveform, symbol_accuracy
except ImportError:
    recover_symbols_from_waveform = None
    symbol_accuracy = None


class ModelEvaluator:
    def __init__(
        self,
        val_loader,
        plotter,
        rrc_taps=None,
        sps=None,
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

    def _can_score_symbols(self, batch):
        return (
            recover_symbols_from_waveform is not None
            and symbol_accuracy is not None
            and self.rrc_taps is not None
            and self.sps is not None
            and "symbols_a" in batch
            and "symbols_b" in batch
        )

    def _symbol_metrics(self, pred, batch, model_name):
        if not self._can_score_symbols(batch):
            return None

        assert recover_symbols_from_waveform is not None
        assert symbol_accuracy is not None
        assert self.rrc_taps is not None
        assert self.sps is not None

        pred_np = pred.detach().cpu().numpy()
        true_sym_a = batch["symbols_a"].cpu().numpy()
        true_sym_b = batch["symbols_b"].cpu().numpy()

        pred_a = pred_np[:, 0, :] + 1j * pred_np[:, 1, :]
        pred_b = pred_np[:, 2, :] + 1j * pred_np[:, 3, :]
        true_sym_a_c = true_sym_a[:, 0, :] + 1j * true_sym_a[:, 1, :]
        true_sym_b_c = true_sym_b[:, 0, :] + 1j * true_sym_b[:, 1, :]

        acc_total = 0.0
        total_examples = 0
        for i in range(pred_a.shape[0]):
            n_symbols_a = len(true_sym_a_c[i])
            n_symbols_b = len(true_sym_b_c[i])

            rec_a = recover_symbols_from_waveform(pred_a[i], self.rrc_taps, self.sps, n_symbols_a)
            rec_b = recover_symbols_from_waveform(pred_b[i], self.rrc_taps, self.sps, n_symbols_b)

            n_a = min(len(rec_a), len(true_sym_a_c[i]))
            n_b = min(len(rec_b), len(true_sym_b_c[i]))
            acc_a = symbol_accuracy(rec_a[:n_a], true_sym_a_c[i][:n_a])
            acc_b = symbol_accuracy(rec_b[:n_b], true_sym_b_c[i][:n_b])
            acc_total += 0.5 * (acc_a + acc_b)
            total_examples += 1

            if i == 0:
                self.plotter.plot_symbol_recovery(true_sym_a_c[i], rec_a, model_name=f"{model_name}_SourceA")
                self.plotter.plot_symbol_recovery(true_sym_b_c[i], rec_b, model_name=f"{model_name}_SourceB")

        return acc_total / max(1, total_examples)

    def run_full_evaluation(self, model, train_hist, val_hist, model_name, training_metadata=None):
        model.eval()
        total_pit_mse = 0.0
        total_pit_sdr = 0.0
        total_plain_mse = 0.0
        total_batches = 0
        total_examples = 0
        inference_time = 0.0
        symbol_acc_values = []

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
            symbol_acc = self._symbol_metrics(pred, batch, model_name)
            if symbol_acc is not None:
                symbol_acc_values.append(symbol_acc)

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

                aligned_y = align_to_pit_target(pred, y, y_alt)
                total_pit_mse += pit_mse_loss(pred, y, y_alt).item()
                total_pit_sdr += pit_sdr(pred, y, y_alt).item()
                total_plain_mse += mse_loss(pred, aligned_y).item()
                total_batches += 1
                total_examples += x.shape[0]

                symbol_acc = self._symbol_metrics(pred, batch, model_name)
                if symbol_acc is not None:
                    symbol_acc_values.append(symbol_acc)

        metrics = {
            "primary_metric": "val_pit_mse",
            "val_pit_mse": total_pit_mse / total_batches,
            "val_pit_sdr_db": total_pit_sdr / total_batches,
            "final_mse": total_plain_mse / total_batches,
            "final_sdr_db": total_pit_sdr / total_batches,
            "train_history": train_hist,
            "val_history": val_hist,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "avg_inference_ms_per_batch": (inference_time / total_batches) * 1000.0,
            "examples_evaluated": total_examples,
        }

        if symbol_acc_values:
            metrics["final_symbol_accuracy"] = sum(symbol_acc_values) / len(symbol_acc_values)

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
            suffix = ""
            if "final_symbol_accuracy" in metrics:
                suffix = f", sym_acc={metrics['final_symbol_accuracy']:.4f}"
            print(
                f"{idx}. {name}: PIT-MSE={metrics['val_pit_mse']:.5f}, "
                f"PIT-SDR={metrics['val_pit_sdr_db']:.2f} dB, "
                f"params={metrics['parameter_count']:,}{suffix}"
            )

        best_name, best_metrics = ranked[0]
        print(
            f"\nBest model: {best_name} "
            f"(PIT-MSE={best_metrics['val_pit_mse']:.5f}, "
            f"PIT-SDR={best_metrics['val_pit_sdr_db']:.2f} dB)"
        )

    def print_latex_table(self, all_metrics):
        include_symbol_acc = any("final_symbol_accuracy" in metrics for metrics in all_metrics.values())
        print("\n% Copy this into your LaTeX report:")
        if include_symbol_acc:
            print(r"\begin{table}[h]\centering")
            print(r"\begin{tabular}{lccccc}")
            print(r"\hline \textbf{Model} & \textbf{Val PIT-MSE} & \textbf{PIT-SDR (dB)} & \textbf{Sym Acc} & \textbf{Params} & \textbf{Best Epoch} \\ \hline")
            for name, metrics in sorted(all_metrics.items(), key=lambda item: item[1]["val_pit_mse"]):
                print(
                    f"{name} & {metrics['val_pit_mse']:.5f} & {metrics['val_pit_sdr_db']:.2f} & "
                    f"{metrics.get('final_symbol_accuracy', 0.0):.4f} & {metrics['parameter_count']} & {metrics.get('best_epoch', '-')} \\\\"
                )
            print(r"\hline \end{tabular} \end{table}")
        else:
            print(r"\begin{table}[h]\centering")
            print(r"\begin{tabular}{lcccc}")
            print(r"\hline \textbf{Model} & \textbf{Val PIT-MSE} & \textbf{PIT-SDR (dB)} & \textbf{Params} & \textbf{Best Epoch} \\ \hline")
            for name, metrics in sorted(all_metrics.items(), key=lambda item: item[1]["val_pit_mse"]):
                print(
                    f"{name} & {metrics['val_pit_mse']:.5f} & {metrics['val_pit_sdr_db']:.2f} "
                    f"& {metrics['parameter_count']} & {metrics.get('best_epoch', '-')} \\\\"
                )
            print(r"\hline \end{tabular} \end{table}")

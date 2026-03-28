# utils/report_utils.py
import os
import json

def log_metrics(model_name, metrics_dict, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{model_name}_results.json")
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

def generate_latex_table(results_dict):
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lcc}")
    print(r"\hline")
    print(r"\textbf{Model Architecture} & \textbf{Final Val MSE} & \textbf{Parameters} \\")
    print(r"\hline")
    for model_name, mse in results_dict.items():
        print(rf"{model_name} & {mse:.4f} & - \\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Comparison of Separation Models on Synthetic RF Data}")
    print(r"\label{tab:results}")
    print(r"\end{table}")
# cross_validation.py
import torch
import os
import json
from itertools import product

from train import train_model

class GridSearchManager:
    def __init__(self, model_class, train_loader, val_loader, evaluator):
        self.model_class = model_class
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator
        self.results = []

# In cross_validation.py
    def run_grid_search(self, param_grid):
        keys, values = zip(*param_grid.items())
        for v in product(*values):
            params = dict(zip(keys, v))
            
            # This allows you to pass 'dropout', 'num_blocks', etc., dynamically
            # Filter params that belong to the model vs. those that belong to the trainer
            model_params = {k: v for k, v in params.items() if k not in ['lr', 'epochs']}
            train_params = {k: v for k, v in params.items() if k in ['lr', 'epochs']}
            
            model = self.model_class(in_ch=8, out_ch=4, **model_params).to(self.device)
            
            # Pass the specific LR and Epochs from the grid
            trained_model, t_hist, v_hist = train_model(
                model, self.train_loader, self.val_loader, self.evaluator.plotter,
                **train_params, device=self.device
            )
                            
            # 3. Evaluate and Plot
            config_name = "_".join([f"{k}-{val}" for k, val in params.items()])
            metrics = self.evaluator.run_full_evaluation(trained_model, t_hist, v_hist, config_name)
            
            self.results.append({
                "params": params,
                "final_mse": metrics["final_mse"],
                "final_sdr": metrics["final_sdr_db"]
            })

        # Save the master leaderboard
        with open(os.path.join(self.evaluator.log_dir, "grid_search_results.json"), "w") as f:
            json.dump(self.results, f, indent=4)
        
        return self.results
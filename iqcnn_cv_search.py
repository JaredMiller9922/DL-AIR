import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from networks.iq_cnn_separator import IQCNNSeparator
from train import evaluate_model, seed_everything, train_model
from utils.data_utils.dataset import SavedRFDataset


def make_loader(dataset, indices, batch_size, shuffle, num_workers):
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def make_kfold_splits(num_examples, num_folds, seed):
    rng = random.Random(seed)
    indices = list(range(num_examples))
    rng.shuffle(indices)

    fold_sizes = [num_examples // num_folds] * num_folds
    for idx in range(num_examples % num_folds):
        fold_sizes[idx] += 1

    splits = []
    cursor = 0
    for fold_size in fold_sizes:
        val_indices = indices[cursor:cursor + fold_size]
        train_indices = indices[:cursor] + indices[cursor + fold_size:]
        splits.append((train_indices, val_indices))
        cursor += fold_size
    return splits


def make_holdout_split(num_examples, holdout_fraction, seed):
    rng = random.Random(seed)
    indices = list(range(num_examples))
    rng.shuffle(indices)
    holdout_size = max(1, int(num_examples * holdout_fraction))
    val_indices = indices[:holdout_size]
    train_indices = indices[holdout_size:]
    return train_indices, val_indices


def build_model(config):
    return IQCNNSeparator(
        in_ch=8,
        out_ch=4,
        base_channels=config["base_channels"],
        dropout=config["dropout"],
    )


def curated_configs():
    return [
        {
            "base_channels": 32,
            "dropout": 0.05,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "input_noise_std": 0.005,
            "receiver_drop_prob": 0.05,
            "grad_clip": 1.0,
            "scheduler_name": "cosine",
            "batch_size": 16,
            "optimizer_name": "adamw",
            "normalize_batches": False,
        },
        {
            "base_channels": 48,
            "dropout": 0.1,
            "lr": 5e-4,
            "weight_decay": 2e-4,
            "input_noise_std": 0.01,
            "receiver_drop_prob": 0.05,
            "grad_clip": 1.0,
            "scheduler_name": "cosine",
            "batch_size": 16,
            "optimizer_name": "adamw",
            "normalize_batches": True,
        },
        {
            "base_channels": 64,
            "dropout": 0.15,
            "lr": 2e-4,
            "weight_decay": 3e-4,
            "input_noise_std": 0.005,
            "receiver_drop_prob": 0.1,
            "grad_clip": 0.5,
            "scheduler_name": "plateau",
            "batch_size": 8,
            "optimizer_name": "adamw",
            "normalize_batches": False,
        },
        {
            "base_channels": 40,
            "dropout": 0.08,
            "lr": 7e-4,
            "weight_decay": 8e-5,
            "input_noise_std": 0.0025,
            "receiver_drop_prob": 0.025,
            "grad_clip": 1.0,
            "scheduler_name": "cosine",
            "batch_size": 16,
            "optimizer_name": "adamw",
            "normalize_batches": True,
        },
    ]


def sample_config(rng):
    return {
        "base_channels": rng.choice([24, 32, 40, 48, 64]),
        "dropout": rng.choice([0.0, 0.05, 0.08, 0.1, 0.15, 0.2]),
        "lr": 10 ** rng.uniform(math.log10(2e-4), math.log10(2e-3)),
        "weight_decay": 10 ** rng.uniform(math.log10(1e-6), math.log10(5e-4)),
        "input_noise_std": rng.choice([0.0, 0.0025, 0.005, 0.01, 0.02]),
        "receiver_drop_prob": rng.choice([0.0, 0.025, 0.05, 0.1]),
        "grad_clip": rng.choice([0.5, 1.0, 2.0]),
        "scheduler_name": rng.choice(["cosine", "plateau"]),
        "batch_size": rng.choice([8, 16]),
        "optimizer_name": "adamw",
        "normalize_batches": rng.choice([False, True]),
    }


def config_stream(seed):
    rng = random.Random(seed)
    for config in curated_configs():
        yield config
    while True:
        yield sample_config(rng)


def save_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=4)


def append_jsonl(path, payload):
    with open(path, "a") as handle:
        handle.write(json.dumps(payload) + "\n")


def run_trial(trial_idx, config, dataset, device, run_dir, num_folds, epochs, patience, num_workers, seed, use_amp):
    trial_dir = run_dir / f"trial_{trial_idx:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    fold_summaries = []
    best_fold = None

    for fold_idx, (train_indices, val_indices) in enumerate(make_kfold_splits(len(dataset), num_folds, seed + trial_idx * 101 + 7), start=1):
        print(f"Trial {trial_idx} | Fold {fold_idx}/{num_folds} | config={config}")

        train_loader = make_loader(dataset, train_indices, config["batch_size"], True, num_workers)
        val_loader = make_loader(dataset, val_indices, config["batch_size"], False, num_workers)

        checkpoint_path = trial_dir / f"fold_{fold_idx:02d}_best.pt"
        model = build_model(config).to(device)
        model, train_hist, val_hist, metadata = train_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            device=device,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            optimizer_name=config["optimizer_name"],
            scheduler_name=config["scheduler_name"],
            scheduler_patience=max(3, patience // 2),
            early_stopping_patience=patience,
            grad_clip=config["grad_clip"],
            normalize_batches=config["normalize_batches"],
            input_noise_std=config["input_noise_std"],
            receiver_drop_prob=config["receiver_drop_prob"],
            checkpoint_path=str(checkpoint_path),
            checkpoint_extra={"trial_idx": trial_idx, "fold_idx": fold_idx, "config": config},
            use_amp=use_amp,
            seed=seed + fold_idx,
        )

        fold_metrics = evaluate_model(model, val_loader, device, normalize_batches=config["normalize_batches"])
        fold_summary = {
            "fold_idx": fold_idx,
            "checkpoint_path": str(checkpoint_path),
            "train_history": train_hist,
            "val_history": val_hist,
            **metadata,
            **fold_metrics,
        }
        fold_summaries.append(fold_summary)

        if best_fold is None or fold_summary["val_pit_mse"] < best_fold["val_pit_mse"]:
            best_fold = fold_summary

    mean_pit_mse = sum(item["val_pit_mse"] for item in fold_summaries) / len(fold_summaries)
    mean_pit_sdr = sum(item["val_pit_sdr_db"] for item in fold_summaries) / len(fold_summaries)
    if best_fold is None:
        raise RuntimeError("No folds completed for this trial.")
    summary = {
        "trial_idx": trial_idx,
        "config": config,
        "num_folds": num_folds,
        "mean_val_pit_mse": mean_pit_mse,
        "mean_val_pit_sdr_db": mean_pit_sdr,
        "fold_summaries": fold_summaries,
        "best_fold_checkpoint": best_fold["checkpoint_path"],
        "best_fold_epoch": best_fold["best_epoch"],
    }
    save_json(trial_dir / "summary.json", summary)
    return summary


def retrain_best_model(best_config, combined_dataset, test_dataset, device, run_dir, epochs, patience, num_workers, seed, use_amp):
    final_dir = run_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    train_indices, val_indices = make_holdout_split(len(combined_dataset), holdout_fraction=0.1, seed=seed)
    train_loader = make_loader(combined_dataset, train_indices, best_config["batch_size"], True, num_workers)
    val_loader = make_loader(combined_dataset, val_indices, best_config["batch_size"], False, num_workers)
    test_loader = DataLoader(
        test_dataset,
        batch_size=best_config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    checkpoint_path = final_dir / "best_iqcnn_final.pt"
    model = build_model(best_config).to(device)
    model, train_hist, val_hist, metadata = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        device=device,
        lr=best_config["lr"],
        weight_decay=best_config["weight_decay"],
        optimizer_name=best_config["optimizer_name"],
        scheduler_name=best_config["scheduler_name"],
        scheduler_patience=max(3, patience // 2),
        early_stopping_patience=patience,
        grad_clip=best_config["grad_clip"],
        normalize_batches=best_config["normalize_batches"],
        input_noise_std=best_config["input_noise_std"],
        receiver_drop_prob=best_config["receiver_drop_prob"],
        checkpoint_path=str(checkpoint_path),
        checkpoint_extra={"stage": "final_retrain", "config": best_config},
        use_amp=use_amp,
        seed=seed,
    )

    holdout_metrics = evaluate_model(model, val_loader, device, normalize_batches=best_config["normalize_batches"])
    test_metrics = evaluate_model(model, test_loader, device, normalize_batches=best_config["normalize_batches"])
    final_summary = {
        "config": best_config,
        "checkpoint_path": str(checkpoint_path),
        "train_history": train_hist,
        "val_history": val_hist,
        "holdout_metrics": holdout_metrics,
        "test_metrics": test_metrics,
        **metadata,
    }
    save_json(final_dir / "final_summary.json", final_summary)
    return final_summary


def main(args):
    seed_everything(args.seed)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    project_root = Path(__file__).resolve().parent
    train_dataset = SavedRFDataset(str(project_root / args.train_dir))
    val_dataset = SavedRFDataset(str(project_root / args.val_dir))
    test_dataset = SavedRFDataset(str(project_root / args.test_dir))
    combined_dataset = ConcatDataset([train_dataset, val_dataset])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root / args.output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    budget_seconds = args.time_budget_hours * 3600.0
    start = time.time()
    best_summary = None
    trial_summaries = []
    stream = config_stream(args.seed)
    jsonl_path = run_dir / "trial_results.jsonl"

    print(f"Device: {device}")
    print(f"Combined CV dataset size: {len(combined_dataset)} | test size: {len(test_dataset)}")
    print(f"Time budget: {args.time_budget_hours:.2f} hours | folds: {args.num_folds} | epochs per fold: {args.epochs}")

    trial_idx = 1
    while True:
        elapsed = time.time() - start
        if elapsed >= budget_seconds and trial_idx > 1:
            break
        if args.max_trials is not None and trial_idx > args.max_trials:
            break

        config = next(stream)
        summary = run_trial(
            trial_idx,
            config,
            combined_dataset,
            device,
            run_dir,
            args.num_folds,
            args.epochs,
            args.patience,
            args.num_workers,
            args.seed,
            args.use_amp,
        )
        summary["elapsed_hours"] = (time.time() - start) / 3600.0
        trial_summaries.append(summary)
        append_jsonl(jsonl_path, summary)

        if best_summary is None or summary["mean_val_pit_mse"] < best_summary["mean_val_pit_mse"]:
            best_summary = summary
            save_json(run_dir / "best_trial_summary.json", best_summary)
            save_json(run_dir / "best_config.json", best_summary["config"])

        print(
            f"Completed trial {trial_idx}: mean PIT-MSE={summary['mean_val_pit_mse']:.5f}, "
            f"mean PIT-SDR={summary['mean_val_pit_sdr_db']:.2f} dB | elapsed={(time.time() - start) / 3600.0:.2f}h"
        )
        trial_idx += 1

    if best_summary is None:
        raise RuntimeError("No hyperparameter trials completed.")

    ranked_trials = sorted(trial_summaries, key=lambda item: item["mean_val_pit_mse"])
    save_json(run_dir / "leaderboard.json", ranked_trials)

    final_summary = retrain_best_model(
        best_summary["config"],
        combined_dataset,
        test_dataset,
        device,
        run_dir,
        args.final_epochs,
        args.final_patience,
        args.num_workers,
        args.seed + 999,
        args.use_amp,
    )

    experiment_summary = {
        "device": device,
        "time_budget_hours": args.time_budget_hours,
        "num_trials_completed": len(trial_summaries),
        "best_trial": best_summary,
        "final_retrain": final_summary,
    }
    save_json(run_dir / "experiment_summary.json", experiment_summary)

    print("\nBest hyperparameters:")
    print(json.dumps(best_summary["config"], indent=4))
    print(
        f"Best CV PIT-MSE={best_summary['mean_val_pit_mse']:.5f}, "
        f"Best CV PIT-SDR={best_summary['mean_val_pit_sdr_db']:.2f} dB"
    )
    print(
        f"Final test PIT-MSE={final_summary['test_metrics']['val_pit_mse']:.5f}, "
        f"Final test PIT-SDR={final_summary['test_metrics']['val_pit_sdr_db']:.2f} dB"
    )
    print(f"Saved search artifacts to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="data/train")
    parser.add_argument("--val-dir", default="data/val")
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--output-dir", default="search_runs/iqcnn_cv")
    parser.add_argument("--time-budget-hours", type=float, default=12.0)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--final-epochs", type=int, default=120)
    parser.add_argument("--final-patience", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-amp", action="store_true")
    main(parser.parse_args())

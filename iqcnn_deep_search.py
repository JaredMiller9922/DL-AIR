import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from networks.iq_cnn_separator import IQCNNSeparator
from train import evaluate_model, seed_everything, train_model
from utils.data_utils.dataset import SavedRFDataset


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def save_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=4)


def build_model(config):
    return IQCNNSeparator(
        in_ch=8,
        out_ch=4,
        base_channels=config["base_channels"],
        dropout=config["dropout"],
    )


def focused_configs():
    return [
        {
            "name": "deep_cfg_01",
            "base_channels": 48,
            "dropout": 0.05,
            "lr": 0.0012,
            "weight_decay": 1.0e-4,
            "input_noise_std": 0.005,
            "receiver_drop_prob": 0.05,
            "grad_clip": 2.0,
            "scheduler_name": "plateau",
            "batch_size": 8,
            "optimizer_name": "adamw",
            "normalize_batches": False,
        },
        {
            "name": "deep_cfg_02",
            "base_channels": 48,
            "dropout": 0.08,
            "lr": 0.0014,
            "weight_decay": 1.0e-4,
            "input_noise_std": 0.005,
            "receiver_drop_prob": 0.10,
            "grad_clip": 2.0,
            "scheduler_name": "plateau",
            "batch_size": 8,
            "optimizer_name": "adamw",
            "normalize_batches": False,
        },
        {
            "name": "deep_cfg_03",
            "base_channels": 48,
            "dropout": 0.08,
            "lr": 0.0016886827584225638,
            "weight_decay": 0.00013153477417331963,
            "input_noise_std": 0.005,
            "receiver_drop_prob": 0.10,
            "grad_clip": 2.0,
            "scheduler_name": "plateau",
            "batch_size": 8,
            "optimizer_name": "adamw",
            "normalize_batches": False,
        },
        {
            "name": "deep_cfg_04",
            "base_channels": 48,
            "dropout": 0.10,
            "lr": 0.0017,
            "weight_decay": 1.5e-4,
            "input_noise_std": 0.005,
            "receiver_drop_prob": 0.10,
            "grad_clip": 2.0,
            "scheduler_name": "plateau",
            "batch_size": 8,
            "optimizer_name": "adamw",
            "normalize_batches": False,
        },
        {
            "name": "deep_cfg_05",
            "base_channels": 48,
            "dropout": 0.08,
            "lr": 0.0017,
            "weight_decay": 2.0e-4,
            "input_noise_std": 0.0025,
            "receiver_drop_prob": 0.10,
            "grad_clip": 2.0,
            "scheduler_name": "plateau",
            "batch_size": 8,
            "optimizer_name": "adamw",
            "normalize_batches": False,
        },
        {
            "name": "deep_cfg_06",
            "base_channels": 48,
            "dropout": 0.08,
            "lr": 0.0017,
            "weight_decay": 1.0e-4,
            "input_noise_std": 0.010,
            "receiver_drop_prob": 0.10,
            "grad_clip": 2.0,
            "scheduler_name": "plateau",
            "batch_size": 8,
            "optimizer_name": "adamw",
            "normalize_batches": False,
        },
        {
            "name": "deep_cfg_07",
            "base_channels": 48,
            "dropout": 0.08,
            "lr": 0.0017,
            "weight_decay": 1.5e-4,
            "input_noise_std": 0.005,
            "receiver_drop_prob": 0.05,
            "grad_clip": 1.0,
            "scheduler_name": "plateau",
            "batch_size": 8,
            "optimizer_name": "adamw",
            "normalize_batches": False,
        },
        {
            "name": "deep_cfg_08",
            "base_channels": 48,
            "dropout": 0.12,
            "lr": 0.0015,
            "weight_decay": 2.0e-4,
            "input_noise_std": 0.005,
            "receiver_drop_prob": 0.10,
            "grad_clip": 2.0,
            "scheduler_name": "plateau",
            "batch_size": 8,
            "optimizer_name": "adamw",
            "normalize_batches": False,
        },
    ]


def rank_key(summary):
    return (summary["best_val_pit_mse"], -summary["best_val_pit_sdr_db"], summary["best_epoch"])


def evaluate_checkpoint(model, checkpoint_path, data_loader, device, normalize_batches):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return evaluate_model(model, data_loader, device, normalize_batches=normalize_batches)


def main(args):
    seed_everything(args.seed)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    project_root = Path(__file__).resolve().parent
    train_dataset = SavedRFDataset(str(project_root / args.train_dir))
    val_dataset = SavedRFDataset(str(project_root / args.val_dir))
    test_dataset = SavedRFDataset(str(project_root / args.test_dir))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root / args.output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    trial_summaries = []
    best_summary = None
    best_checkpoint_path = None

    print(f"Device: {device}")
    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)} | Test size: {len(test_dataset)}")
    configs = focused_configs()
    if args.max_trials is not None:
        configs = configs[:args.max_trials]
    print(f"Focused deep search configs: {len(configs)} | epochs per config: {args.epochs}")

    for trial_idx, config in enumerate(configs, start=1):
        trial_name = config["name"]
        trial_dir = run_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nTrial {trial_idx}/{len(configs)}: {trial_name} | config={config}")

        train_loader = make_loader(train_dataset, config["batch_size"], True, args.num_workers)
        val_loader = make_loader(val_dataset, config["batch_size"], False, args.num_workers)
        test_loader = make_loader(test_dataset, config["batch_size"], False, args.num_workers)

        checkpoint_path = trial_dir / "best.pt"
        model = build_model(config).to(device)
        model, train_hist, val_hist, metadata = train_model(
            model,
            train_loader,
            val_loader,
            epochs=args.epochs,
            device=device,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            optimizer_name=config["optimizer_name"],
            scheduler_name=config["scheduler_name"],
            scheduler_patience=args.scheduler_patience,
            early_stopping_patience=args.patience,
            grad_clip=config["grad_clip"],
            normalize_batches=config["normalize_batches"],
            input_noise_std=config["input_noise_std"],
            receiver_drop_prob=config["receiver_drop_prob"],
            checkpoint_path=str(checkpoint_path),
            checkpoint_extra={"trial_name": trial_name, "config": config},
            use_amp=args.use_amp,
            seed=args.seed + trial_idx,
        )

        val_metrics = evaluate_model(model, val_loader, device, normalize_batches=config["normalize_batches"])
        test_metrics = evaluate_model(model, test_loader, device, normalize_batches=config["normalize_batches"])

        trial_summary = {
            "trial_idx": trial_idx,
            "trial_name": trial_name,
            "config": config,
            "checkpoint_path": str(checkpoint_path),
            "train_history": train_hist,
            "val_history": val_hist,
            **metadata,
            "best_val_pit_sdr_db": val_metrics["val_pit_sdr_db"],
            "selected_val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        trial_summaries.append(trial_summary)
        save_json(trial_dir / "summary.json", trial_summary)

        if best_summary is None or rank_key(trial_summary) < rank_key(best_summary):
            best_summary = trial_summary
            best_checkpoint_path = checkpoint_path
            save_json(run_dir / "best_config.json", config)
            save_json(run_dir / "best_trial_summary.json", best_summary)
            shutil.copy2(checkpoint_path, run_dir / "best_overall.pt")

        print(
            f"Selected score: best Val PIT-MSE={trial_summary['best_val_pit_mse']:.5f}, "
            f"Val PIT-SDR={trial_summary['best_val_pit_sdr_db']:.2f} dB, best epoch={trial_summary['best_epoch']}"
        )

    if best_summary is None or best_checkpoint_path is None:
        raise RuntimeError("No deep-search trials completed.")

    ranked_trials = sorted(trial_summaries, key=rank_key)
    save_json(run_dir / "leaderboard.json", ranked_trials)

    best_model = build_model(best_summary["config"]).to(device)
    val_loader = make_loader(val_dataset, best_summary["config"]["batch_size"], False, args.num_workers)
    test_loader = make_loader(test_dataset, best_summary["config"]["batch_size"], False, args.num_workers)
    best_val_metrics = evaluate_checkpoint(
        best_model,
        best_checkpoint_path,
        val_loader,
        device,
        best_summary["config"]["normalize_batches"],
    )
    best_test_metrics = evaluate_checkpoint(
        best_model,
        best_checkpoint_path,
        test_loader,
        device,
        best_summary["config"]["normalize_batches"],
    )

    experiment_summary = {
        "device": device,
        "search_type": "focused_deep_search",
        "selection_metric": "best_val_pit_mse",
        "epochs_per_config": args.epochs,
        "num_trials_completed": len(trial_summaries),
        "best_trial": best_summary,
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_checkpoint_val_metrics": best_val_metrics,
        "best_checkpoint_test_metrics": best_test_metrics,
    }
    save_json(run_dir / "experiment_summary.json", experiment_summary)

    print("\nBest config:")
    print(json.dumps(best_summary["config"], indent=4))
    print(
        f"Best selected validation PIT-MSE={best_summary['best_val_pit_mse']:.5f} "
        f"at epoch {best_summary['best_epoch']}"
    )
    print(
        f"Best checkpoint test PIT-MSE={best_test_metrics['val_pit_mse']:.5f}, "
        f"test PIT-SDR={best_test_metrics['val_pit_sdr_db']:.2f} dB"
    )
    print(f"Saved focused deep-search artifacts to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="data/train")
    parser.add_argument("--val-dir", default="data/val")
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--output-dir", default="search_runs/iqcnn_deep")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-amp", action="store_true")
    main(parser.parse_args())

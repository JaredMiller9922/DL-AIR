from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import benchmark_config
from main import (
    build_family_config,
    build_signal_configs,
    default_checkpoint_path,
    get_model,
    load_model_weights,
    model_is_trainable,
    normalize_model_name,
    run_experiment,
)
from utils.analysis.rf_features import complex_mixture_features
from utils.data_utils.dataset import SyntheticRFDataset
from utils.data_utils.generator import RFMixtureGenerator, SourceConfig


DEFAULT_MODELS = ["FastICA", "Hybrid", "LSTM", "Linear", "IQ_CNN", "HTDemucs"]


def split_csv(value: str | None, default: Iterable[str]) -> List[str]:
    if value is None or not value.strip():
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def swapped_sources(y: torch.Tensor) -> torch.Tensor:
    return y[:, [2, 3, 0, 1], :]


def align_prediction_and_target(pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = min(pred.shape[-1], y.shape[-1])
    pred = pred[..., :length]
    y = y[..., :length]
    y_swap = swapped_sources(y)

    dims = tuple(range(1, pred.ndim))
    direct = ((pred - y) ** 2).mean(dim=dims)
    swapped = ((pred - y_swap) ** 2).mean(dim=dims)
    use_swap = swapped < direct
    view_shape = [pred.shape[0]] + [1] * (pred.ndim - 1)
    y_best = torch.where(use_swap.view(*view_shape), y_swap, y)
    return pred, y_best


def batch_metrics(pred: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    pred, y_best = align_prediction_and_target(pred, y)
    mse = torch.mean((pred - y_best) ** 2)
    noise = y_best - pred
    dims = tuple(range(1, y_best.ndim))
    sdr = 10 * torch.log10(torch.sum(y_best**2, dim=dims) / (torch.sum(noise**2, dim=dims) + 1e-8))
    return {
        "mse": float(mse.detach().cpu()),
        "sdr_db": float(sdr.mean().detach().cpu()),
        "n_examples": int(pred.shape[0]),
    }


def make_dataset(cfg, num_examples: int, seed: int) -> SyntheticRFDataset:
    generator = RFMixtureGenerator(seed=seed)
    qpsk_cfg, noise_cfg, mix_cfg, _ = build_signal_configs(cfg)
    source_a_cfg = SourceConfig(
        source_type=cfg.source_a_type,
        n_symbols=cfg.num_symbols,
        samples_per_symbol=cfg.samples_per_symbol,
        rolloff=cfg.rolloff,
        rrc_span_symbols=cfg.rrc_span_symbols,
        normalize_power=cfg.normalize_power,
        num_channels=cfg.n_rx,
    )
    source_b_cfg = SourceConfig(
        source_type=cfg.source_b_type,
        n_symbols=cfg.num_symbols,
        samples_per_symbol=cfg.samples_per_symbol,
        rolloff=cfg.rolloff,
        rrc_span_symbols=cfg.rrc_span_symbols,
        normalize_power=cfg.normalize_power,
        num_channels=cfg.n_rx,
    )
    return SyntheticRFDataset(
        num_examples=num_examples,
        generator=generator,
        qpsk_cfg_soi=qpsk_cfg,
        qpsk_cfg_int=qpsk_cfg,
        noise_cfg=noise_cfg,
        mix_cfg=mix_cfg,
        source_a_cfg=source_a_cfg,
        source_b_cfg=source_b_cfg,
        family_cfg=build_family_config(cfg),
    )


def evaluate_model(cfg, model_name: str, loader: DataLoader, device: str, allow_random_init: bool = False) -> Dict:
    row = {
        "benchmark": cfg.benchmark_name,
        "model": model_name,
        "model_type": "neural" if model_name != "FastICA" else "classical_separator",
        "mode": "evaluate",
        "n_rx": cfg.n_rx,
        "num_examples": len(loader.dataset),
        "mse": np.nan,
        "sdr_db": np.nan,
        "symbol_accuracy": np.nan,
        "checkpoint_path": None,
        "status": "ok",
    }

    try:
        cfg.model_name = normalize_model_name(model_name)
        model = get_model(cfg, device)
        if model_is_trainable(cfg.model_name):
            checkpoint = default_checkpoint_path(cfg.model_name, cfg.model_dir)
            if checkpoint is None or not checkpoint.exists():
                if not allow_random_init:
                    row["status"] = "missing_checkpoint"
                    return row
                row["status"] = "random_init_no_checkpoint"
            else:
                loaded = load_model_weights(model, checkpoint, device)
                row["checkpoint_path"] = str(loaded)

        model.eval()
        totals = {"mse": 0.0, "sdr_db": 0.0, "n_examples": 0}
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device).float()
                y = batch["y"].to(device).float()
                pred = model(x)
                metrics = batch_metrics(pred, y)
                totals["mse"] += metrics["mse"] * metrics["n_examples"]
                totals["sdr_db"] += metrics["sdr_db"] * metrics["n_examples"]
                totals["n_examples"] += metrics["n_examples"]

        if totals["n_examples"] > 0:
            row["mse"] = totals["mse"] / totals["n_examples"]
            row["sdr_db"] = totals["sdr_db"] / totals["n_examples"]
    except Exception as exc:
        row["status"] = f"error: {type(exc).__name__}: {exc}"

    return row


def train_model_for_benchmark(cfg, model_name: str) -> Dict:
    cfg.model_name = normalize_model_name(model_name)
    cfg.mode = "train"
    try:
        results = run_experiment(cfg)
        return {
            "benchmark": cfg.benchmark_name,
            "model": cfg.model_name,
            "model_type": "neural",
            "mode": "train",
            "n_rx": cfg.n_rx,
            "num_examples": cfg.val_examples,
            "mse": results.get("final_mse"),
            "sdr_db": results.get("final_sdr_db"),
            "symbol_accuracy": results.get("final_symbol_accuracy"),
            "checkpoint_path": results.get("checkpoint_path"),
            "status": "ok",
        }
    except Exception as exc:
        return {
            "benchmark": cfg.benchmark_name,
            "model": model_name,
            "model_type": "neural",
            "mode": "train",
            "n_rx": cfg.n_rx,
            "num_examples": cfg.val_examples,
            "mse": np.nan,
            "sdr_db": np.nan,
            "symbol_accuracy": np.nan,
            "checkpoint_path": None,
            "status": f"error: {type(exc).__name__}: {exc}",
        }


def feature_table_for_benchmark(cfg, num_examples: int, seed: int) -> pd.DataFrame:
    generator = RFMixtureGenerator(seed=seed)
    qpsk_cfg, noise_cfg, mix_cfg, _ = build_signal_configs(cfg)
    family_cfg = build_family_config(cfg)
    source_a_cfg = SourceConfig(
        source_type=cfg.source_a_type,
        n_symbols=cfg.num_symbols,
        samples_per_symbol=cfg.samples_per_symbol,
        rolloff=cfg.rolloff,
        rrc_span_symbols=cfg.rrc_span_symbols,
        normalize_power=cfg.normalize_power,
        num_channels=cfg.n_rx,
    )
    source_b_cfg = SourceConfig(
        source_type=cfg.source_b_type,
        n_symbols=cfg.num_symbols,
        samples_per_symbol=cfg.samples_per_symbol,
        rolloff=cfg.rolloff,
        rrc_span_symbols=cfg.rrc_span_symbols,
        normalize_power=cfg.normalize_power,
        num_channels=cfg.n_rx,
    )

    rows = []
    for idx in range(num_examples):
        ex = generator.generate_mixture(
            qpsk_cfg,
            qpsk_cfg,
            noise_cfg,
            mix_cfg,
            source_a_cfg=source_a_cfg,
            source_b_cfg=source_b_cfg,
            family_cfg=family_cfg,
        )
        row = {
            "benchmark": cfg.benchmark_name,
            "example_index": idx,
            "source_a_type": ex["meta"]["source_a"]["source_type"],
            "source_b_type": ex["meta"]["source_b"]["source_type"],
            "alpha": ex["meta"]["alpha"],
            "snr_db": ex["meta"]["snr_db"],
            "mixing_mode": ex["meta"]["mixing_mode"],
        }
        row.update(complex_mixture_features(ex["mixture"]))
        rows.append(row)
    return pd.DataFrame(rows)


def run_classical_feature_baselines(feature_df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    if feature_df.empty or feature_df["source_b_type"].nunique() < 2:
        return pd.DataFrame(
            [
                {
                    "benchmark": benchmark,
                    "model": "Classical feature baselines",
                    "model_type": "classical_feature",
                    "task": "source_b_type",
                    "mean_cv_accuracy": np.nan,
                    "std_cv_accuracy": np.nan,
                    "status": "not_enough_classes",
                }
            ]
        )

    feature_cols = [
        col
        for col in feature_df.select_dtypes(include=[np.number]).columns
        if col not in {"example_index"}
    ]
    X = feature_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = feature_df["source_b_type"].astype(str)
    min_class = int(y.value_counts().min())
    if min_class < 2:
        return pd.DataFrame(
            [
                {
                    "benchmark": benchmark,
                    "model": "Classical feature baselines",
                    "model_type": "classical_feature",
                    "task": "source_b_type",
                    "mean_cv_accuracy": np.nan,
                    "std_cv_accuracy": np.nan,
                    "status": "not_enough_examples_per_class",
                }
            ]
        )

    cv = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=42)
    models = {
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.05),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    }
    rows = []
    for name, model in models.items():
        try:
            pipe = make_pipeline(StandardScaler(), model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
            rows.append(
                {
                    "benchmark": benchmark,
                    "model": name,
                    "model_type": "classical_feature",
                    "task": "source_b_type",
                    "mean_cv_accuracy": float(scores.mean()),
                    "std_cv_accuracy": float(scores.std()),
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "benchmark": benchmark,
                    "model": name,
                    "model_type": "classical_feature",
                    "task": "source_b_type",
                    "mean_cv_accuracy": np.nan,
                    "std_cv_accuracy": np.nan,
                    "status": f"error: {type(exc).__name__}: {exc}",
                }
            )
    return pd.DataFrame(rows)


def write_outputs(output_dir: Path, cfg, model_rows: List[Dict], feature_df: pd.DataFrame, classical_df: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_df = pd.DataFrame(model_rows)
    model_df.to_csv(output_dir / "synthetic_model_results.csv", index=False)
    feature_df.to_csv(output_dir / "synthetic_feature_examples.csv", index=False)
    classical_df.to_csv(output_dir / "classical_feature_results.csv", index=False)

    payload = {
        "config": asdict(cfg),
        "model_results": model_df.to_dict(orient="records"),
        "classical_feature_results": classical_df.to_dict(orient="records"),
    }
    with open(output_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run synthetic RF benchmark suites.")
    parser.add_argument("--benchmark", default="benchmark_modulation_diverse")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--num_examples", type=int, default=64)
    parser.add_argument("--feature_examples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=["evaluate", "train"], default="evaluate")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--allow_random_init", action="store_true")
    parser.add_argument("--output_root", default="reporting_pipeline/outputs/benchmarks")
    args = parser.parse_args()

    cfg = benchmark_config(args.benchmark)
    cfg.batch_size = args.batch_size
    cfg.val_examples = args.num_examples
    cfg.test_examples = args.num_examples
    cfg.train_examples = max(args.num_examples, args.batch_size)
    cfg.epochs = args.epochs
    cfg.mode = args.mode
    cfg.__post_init__()

    output_dir = Path(args.output_root) / cfg.benchmark_name
    model_names = split_csv(args.models, DEFAULT_MODELS)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    if args.mode == "evaluate":
        dataset = make_dataset(cfg, args.num_examples, args.seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        model_rows = [
            evaluate_model(cfg, model_name, loader, device, allow_random_init=args.allow_random_init)
            for model_name in model_names
        ]
    else:
        model_rows = []
        for model_name in model_names:
            normalized = normalize_model_name(model_name)
            if normalized == "FastICA":
                eval_cfg = benchmark_config(args.benchmark)
                eval_cfg.batch_size = args.batch_size
                eval_cfg.val_examples = args.num_examples
                eval_cfg.mode = "evaluate"
                dataset = make_dataset(eval_cfg, args.num_examples, args.seed)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
                model_rows.append(evaluate_model(eval_cfg, normalized, loader, device))
            else:
                train_cfg = benchmark_config(args.benchmark)
                train_cfg.batch_size = args.batch_size
                train_cfg.train_examples = max(args.num_examples, args.batch_size)
                train_cfg.val_examples = args.num_examples
                train_cfg.epochs = args.epochs
                model_rows.append(train_model_for_benchmark(train_cfg, normalized))

    feature_df = feature_table_for_benchmark(cfg, args.feature_examples, args.seed + 1000)
    classical_df = run_classical_feature_baselines(feature_df, cfg.benchmark_name)
    write_outputs(output_dir, cfg, model_rows, feature_df, classical_df)

    print(f"Wrote benchmark outputs to {output_dir}")
    print(pd.DataFrame(model_rows).to_string(index=False))
    print(classical_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

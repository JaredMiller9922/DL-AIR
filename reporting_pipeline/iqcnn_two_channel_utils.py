import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ExperimentConfig
from networks.iq_cnn_separator import IQCNNSeparator
from reporting_pipeline.repo_data import generate_fixed_splits, make_loader
from utils.model_utils.losses import align_to_pit_target, calculate_sdr, mse_loss, pit_mse_loss
from utils.model_utils.symbol_utils import recover_symbols_from_waveform, rrc_taps, symbol_accuracy


def make_iqcnn_2ch_config(dataset_path: str, epochs: int, train_examples: int, val_examples: int, test_examples: int) -> ExperimentConfig:
    cfg = ExperimentConfig(
        mode="iqcnn_2ch_study",
        model_name="IQ_CNN",
        dataset_path=dataset_path,
        generate_new_data=True,
        use_on_the_fly_data=False,
        batch_size=32,
        epochs=epochs,
        lr=1e-3,
        n_symbols=400,
        samples_per_symbol=2,
        rolloff=0.25,
        rrc_span_symbols=12,
        normalize_power=True,
        noise_enabled=True,
        alpha=1.0,
        snr_db=15.0,
        n_rx=2,
        random_phase=True,
    )
    cfg._train_examples = train_examples
    cfg._val_examples = val_examples
    cfg._test_examples = test_examples
    return cfg


def ensure_dataset(cfg: ExperimentConfig, seed: int) -> Path:
    dataset_root = Path(cfg.dataset_path)
    generate_fixed_splits(
        cfg,
        dataset_root,
        train_size=cfg._train_examples,
        val_size=cfg._val_examples,
        test_size=cfg._test_examples,
        seed=seed,
    )
    return dataset_root


def build_loaders(dataset_root: Path, batch_size: int):
    train_loader = make_loader(dataset_root / "train", batch_size=batch_size, shuffle=True)
    val_loader = make_loader(dataset_root / "val", batch_size=batch_size, shuffle=False)
    test_loader = make_loader(dataset_root / "test", batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_iqcnn_with_loss(cfg: ExperimentConfig, dataset_root: Path, output_root: Path, loss_name: str, base_channels: int = 48, dropout: float = 0.08):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = build_loaders(dataset_root, cfg.batch_size)
    model = IQCNNSeparator(in_ch=2 * cfg.n_rx, out_ch=4, base_channels=base_channels, dropout=dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    train_hist = []
    val_hist = []
    best_val = float("inf")
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    checkpoint_path = output_root / f"iqcnn_{loss_name}.pt"
    output_root.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_train = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_alt = batch["y_alt"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                pred = model(x)
                if loss_name == "pit_mse":
                    loss = pit_mse_loss(pred, y, y_alt)
                elif loss_name == "mse":
                    loss = mse_loss(pred, y)
                else:
                    raise ValueError(f"Unsupported loss {loss_name}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_train += float(loss.item())

        avg_train = total_train / max(1, len(train_loader))
        avg_val, _ = evaluate_iqcnn(model, val_loader, cfg)
        train_hist.append(avg_train)
        val_hist.append(avg_val["pit_mse"])  # compare on PIT metric for fairness
        scheduler.step(avg_val["pit_mse"])

        if avg_val["pit_mse"] < best_val:
            best_val = avg_val["pit_mse"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model": best_state,
                    "best_epoch": best_epoch,
                    "best_val_pit_mse": best_val,
                    "loss_name": loss_name,
                    "config": asdict(cfg),
                },
                checkpoint_path,
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(
                f"[{loss_name}] Epoch {epoch}: train_loss={avg_train:.6f} "
                f"val_pit_mse={avg_val['pit_mse']:.6f} val_soi_acc={avg_val['soi_symbol_accuracy']:.4f} "
                f"time={time.perf_counter() - epoch_start:.1f}s"
            )

    model.load_state_dict(best_state)
    test_metrics, example = evaluate_iqcnn(model, test_loader, cfg)
    summary = {
        "loss_name": loss_name,
        "best_epoch": best_epoch,
        "best_val_pit_mse": best_val,
        "train_history": train_hist,
        "val_history": val_hist,
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_root),
    }
    with open(output_root / f"iqcnn_{loss_name}_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    plot_training_curves(train_hist, val_hist, output_root / f"iqcnn_{loss_name}_curves.png", title=f"IQ_CNN ({loss_name})")
    plot_example(example, output_root / f"iqcnn_{loss_name}_example.png", title=f"IQ_CNN ({loss_name})")
    return summary


def evaluate_iqcnn(model, loader, cfg: ExperimentConfig):
    device = next(model.parameters()).device
    taps = rrc_taps(cfg.samples_per_symbol, cfg.rolloff, cfg.rrc_span_symbols)
    model.eval()
    total_pit_mse = 0.0
    total_mse = 0.0
    total_sdr = 0.0
    total_soi_acc = 0.0
    total_int_acc = 0.0
    count = 0
    example = None

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_alt = batch["y_alt"].to(device)
            pred = model(x)
            aligned = align_to_pit_target(pred, y, y_alt)
            total_pit_mse += pit_mse_loss(pred, y, y_alt).item()
            total_mse += mse_loss(pred, aligned).item()
            total_sdr += calculate_sdr(pred, aligned).item()

            pred_np = aligned.detach().cpu().numpy()
            syms_a = batch["symbols_a"].numpy()
            syms_b = batch["symbols_b"].numpy()
            for i in range(pred_np.shape[0]):
                pred_a = pred_np[i, 0] + 1j * pred_np[i, 1]
                pred_b = pred_np[i, 2] + 1j * pred_np[i, 3]
                true_a = syms_a[i][0] + 1j * syms_a[i][1]
                true_b = syms_b[i][0] + 1j * syms_b[i][1]
                rec_a = recover_symbols_from_waveform(pred_a, taps, cfg.samples_per_symbol, len(true_a))
                rec_b = recover_symbols_from_waveform(pred_b, taps, cfg.samples_per_symbol, len(true_b))
                total_soi_acc += float(symbol_accuracy(rec_a[: len(true_a)], true_a))
                total_int_acc += float(symbol_accuracy(rec_b[: len(true_b)], true_b))
                count += 1
                if example is None:
                    example = {"x": batch["x"][i].numpy(), "y": batch["y"][i].numpy(), "pred": pred_np[i]}

    metrics = {
        "pit_mse": total_pit_mse / max(1, len(loader)),
        "wave_mse": total_mse / max(1, len(loader)),
        "sdr_db": total_sdr / max(1, len(loader)),
        "soi_symbol_accuracy": total_soi_acc / max(1, count),
        "int_symbol_accuracy": total_int_acc / max(1, count),
        "avg_symbol_accuracy": 0.5 * ((total_soi_acc / max(1, count)) + (total_int_acc / max(1, count))),
    }
    return metrics, example


def plot_training_curves(train_hist, val_hist, out_path: Path, title: str):
    plt.figure(figsize=(8, 4.5))
    plt.plot(train_hist, label="Train Loss")
    plt.plot(val_hist, label="Val PIT-MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_example(example: dict, out_path: Path, title: str, num_samples: int = 200):
    x = example["x"][:, :num_samples]
    y = example["y"][:, :num_samples]
    pred = example["pred"][:, :num_samples]
    t = range(num_samples)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, y[0], label="SOI I")
    axes[0].plot(t, y[2], label="INT I")
    axes[0].set_title(f"{title}: Clean Sources")
    axes[0].legend()

    axes[1].plot(t, x[0], label="RX0 I")
    axes[1].plot(t, x[2], label="RX1 I")
    axes[1].set_title(f"{title}: Two-Channel Mixture")
    axes[1].legend()

    axes[2].plot(t, y[0], color="black", alpha=0.6, label="True SOI I")
    axes[2].plot(t, pred[0], linestyle="--", label="Pred SOI I")
    axes[2].plot(t, y[2], color="tab:orange", alpha=0.6, label="True INT I")
    axes[2].plot(t, pred[2], linestyle="--", color="tab:red", label="Pred INT I")
    axes[2].set_title(f"{title}: Separated Outputs")
    axes[2].legend(ncol=2, fontsize=8)
    axes[2].set_xlabel("Sample")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()

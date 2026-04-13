import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reporting_pipeline.iqcnn_two_channel_utils import ensure_dataset, make_iqcnn_2ch_config, train_iqcnn_with_loss


def main():
    output_root = PROJECT_ROOT / "reporting_pipeline" / "outputs" / "iqcnn_2ch_loss_study"
    dataset_root = output_root / "data"

    cfg = make_iqcnn_2ch_config(
        dataset_path=str(dataset_root),
        epochs=60,
        train_examples=20000,
        val_examples=2000,
        test_examples=2000,
    )
    dataset_root = ensure_dataset(cfg, seed=41)

    pit_summary = train_iqcnn_with_loss(cfg, dataset_root, output_root, "pit_mse", base_channels=48, dropout=0.08)
    mse_summary = train_iqcnn_with_loss(cfg, dataset_root, output_root, "mse", base_channels=48, dropout=0.08)

    comparison = {
        "config": {
            "dataset_path": str(dataset_root),
            "epochs": cfg.epochs,
            "train_examples": cfg._train_examples,
            "val_examples": cfg._val_examples,
            "test_examples": cfg._test_examples,
            "n_rx": cfg.n_rx,
            "alpha": cfg.alpha,
            "snr_db": cfg.snr_db,
        },
        "pit_mse": pit_summary,
        "mse": mse_summary,
        "recommended_loss": "pit_mse" if pit_summary["test_metrics"]["avg_symbol_accuracy"] >= mse_summary["test_metrics"]["avg_symbol_accuracy"] else "mse",
    }
    with open(output_root / "comparison_summary.json", "w") as handle:
        json.dump(comparison, handle, indent=2)

    print(json.dumps(comparison, indent=2))
    print(f"Saved loss-study outputs to {output_root}")


if __name__ == "__main__":
    main()

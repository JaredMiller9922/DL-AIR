import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reporting_pipeline.iqcnn_two_channel_utils import ensure_dataset, make_iqcnn_2ch_config, train_iqcnn_with_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", choices=["pit_mse", "mse"], default="mse")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--train-examples", type=int, default=40000)
    parser.add_argument("--val-examples", type=int, default=4000)
    parser.add_argument("--test-examples", type=int, default=4000)
    args = parser.parse_args()

    output_root = PROJECT_ROOT / "reporting_pipeline" / "outputs" / "iqcnn_2ch_full_train"
    dataset_root = output_root / "data"

    cfg = make_iqcnn_2ch_config(
        dataset_path=str(dataset_root),
        epochs=args.epochs,
        train_examples=args.train_examples,
        val_examples=args.val_examples,
        test_examples=args.test_examples,
    )
    dataset_root = ensure_dataset(cfg, seed=51)
    summary = train_iqcnn_with_loss(cfg, dataset_root, output_root, args.loss, base_channels=48, dropout=0.08)

    with open(output_root / "run_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved full-train outputs to {output_root}")


if __name__ == "__main__":
    main()

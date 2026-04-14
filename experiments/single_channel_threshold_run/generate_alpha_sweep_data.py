import argparse
import json
from pathlib import Path

from sweep_common import PROJECT_ROOT, alpha_tag, generate_dataset_for_alpha, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate saved datasets for alpha sweep")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.80, 0.85, 0.90, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.07, 1.10])
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default="experiments/single_channel_threshold_run/alpha_sweep/data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    data_root = PROJECT_ROOT / args.data_root
    data_root.mkdir(parents=True, exist_ok=True)

    generated = []
    for idx, alpha in enumerate(args.alphas):
        print(f"Generating alpha={alpha:.2f}")
        target = generate_dataset_for_alpha(data_root, alpha, train_size=args.train_size, val_size=args.val_size, seed=args.seed + idx)
        generated.append({"alpha": alpha, "path": str(target)})

    with open(data_root / "index.json", "w") as handle:
        json.dump(
            {
                "alphas": args.alphas,
                "train_size": args.train_size,
                "val_size": args.val_size,
                "generated": generated,
            },
            handle,
            indent=2,
        )
    print(f"Generated {len(generated)} alpha datasets under {data_root}")
    print("Alpha tags:", ", ".join(alpha_tag(alpha) for alpha in args.alphas))


if __name__ == "__main__":
    main()

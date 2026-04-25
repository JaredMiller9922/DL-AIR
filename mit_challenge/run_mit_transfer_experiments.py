from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = MIT_ROOT / "sourceCode"
DEFAULT_OCTAVE = Path(r"D:\Octave-11.1.0\mingw64\bin\octave-cli.exe")


CHECKPOINTS = {
    "Hybrid": PROJECT_ROOT / "pytorch_models" / "hybrid_model.pt",
    "LSTM": PROJECT_ROOT / "pytorch_models" / "lstm_model.pt",
    "IQ_CNN": PROJECT_ROOT / "pytorch_models" / "iq_cnn_model.pt",
    "HTDemucs": PROJECT_ROOT / "pytorch_models" / "htdemucs_model.pt",
}


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_model_name(model_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in model_name)


def octave_quote(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def run_model(octave_exe: Path, model_name: str, n_alpha: int, n_frames: int) -> dict:
    checkpoint = CHECKPOINTS.get(model_name)
    row = {
        "model": model_name,
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "n_alpha": n_alpha,
        "n_frames": n_frames,
        "status": "ok",
    }
    if checkpoint is None or not checkpoint.exists():
        row["status"] = "missing_checkpoint"
        return row

    eval_cmd = (
        f"cd('{octave_quote(SOURCE_DIR)}'); "
        f"evalLearnedDebugMain({n_alpha}, {n_frames}, '{model_name}', '{octave_quote(checkpoint)}')"
    )
    completed = subprocess.run(
        [str(octave_exe), "--eval", eval_cmd],
        cwd=str(SOURCE_DIR),
        text=True,
        capture_output=True,
    )
    row["returncode"] = completed.returncode
    row["stdout_tail"] = completed.stdout[-2000:]
    row["stderr_tail"] = completed.stderr[-2000:]
    if completed.returncode != 0:
        row["status"] = "octave_failed"
    return row


def collect_summaries(models: list[str]) -> pd.DataFrame:
    tables = []
    for model_name in models:
        summary_path = MIT_ROOT / f"debugEval_learned_{safe_model_name(model_name)}" / "debug_run_summary.csv"
        if not summary_path.exists():
            continue
        table = pd.read_csv(summary_path)
        table["model"] = model_name
        table["summary_path"] = str(summary_path)
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MIT transfer scoring for learned separator checkpoints.")
    parser.add_argument("--models", default="Hybrid,LSTM,IQ_CNN,HTDemucs")
    parser.add_argument("--n_alpha", type=int, default=1)
    parser.add_argument("--n_frames", type=int, default=5)
    parser.add_argument("--octave_exe", default=str(DEFAULT_OCTAVE))
    parser.add_argument("--output_csv", default=str(MIT_ROOT / "mit_transfer_model_summary.csv"))
    args = parser.parse_args()

    models = split_csv(args.models)
    octave_exe = Path(args.octave_exe)
    run_rows = []
    if not octave_exe.exists():
        run_rows = [
            {
                "model": model,
                "checkpoint_path": str(CHECKPOINTS.get(model)) if CHECKPOINTS.get(model) else None,
                "n_alpha": args.n_alpha,
                "n_frames": args.n_frames,
                "status": f"missing_octave: {octave_exe}",
            }
            for model in models
        ]
    else:
        for model in models:
            run_rows.append(run_model(octave_exe, model, args.n_alpha, args.n_frames))

    run_df = pd.DataFrame(run_rows)
    summary_df = collect_summaries(models)
    if not summary_df.empty:
        metric_summary = (
            summary_df.groupby("model", dropna=False)
            .agg(
                frame_success_rate=("frameSuccessRate", "mean"),
                mean_ber=("ber", "mean"),
                n_scored_alpha=("alphaIndex", "count"),
            )
            .reset_index()
        )
        run_df = run_df.merge(metric_summary, on="model", how="left")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    run_df.to_csv(output_csv, index=False)
    print(f"Wrote MIT transfer model summary to {output_csv}")
    print(run_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

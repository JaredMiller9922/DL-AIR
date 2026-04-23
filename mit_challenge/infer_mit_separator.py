import argparse
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ExperimentConfig
from main import get_model, load_model_weights, normalize_model_name
from utils.model_utils.conversion_helpers import complex_matrix_to_iq_channels


def read_mit_iqdata(path: Path, n_rx: int) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f4")
    if raw.size % 2 != 0:
        raise ValueError(f"Odd number of float32 values in {path}")

    complex_samples = raw[0::2] + 1j * raw[1::2]
    if complex_samples.size % n_rx != 0:
        raise ValueError(
            f"Complex sample count {complex_samples.size} is not divisible by n_rx={n_rx} for {path}"
        )

    samples_per_rx = complex_samples.size // n_rx
    return complex_samples.reshape(samples_per_rx, n_rx).T.astype(np.complex64)


def write_mit_iqdata(path_without_ext: Path, signal: np.ndarray) -> None:
    signal = np.asarray(signal, dtype=np.complex64).reshape(-1)
    interleaved = np.empty(signal.size * 2, dtype="<f4")
    interleaved[0::2] = signal.real.astype("<f4")
    interleaved[1::2] = signal.imag.astype("<f4")
    path_without_ext.parent.mkdir(parents=True, exist_ok=True)
    interleaved.tofile(path_without_ext.with_suffix(".iqdata"))


def checkpoint_state_dict(checkpoint_path: Path):
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if state_dict is None and isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        state_dict = checkpoint
    if any(str(key).startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def infer_expected_input_channels(state_dict) -> int:
    candidate_keys = [
        "in_proj.weight",
        "unet.input_proj.weight",
        "net.time_encoders.0.conv.weight",
        "input_proj.weight",
    ]
    for key in candidate_keys:
        if key in state_dict:
            weight = state_dict[key]
            if weight.ndim == 2:
                return int(weight.shape[1])
            if weight.ndim >= 3:
                return int(weight.shape[1])
    raise ValueError("Could not infer checkpoint input channel count from known model keys.")


def output_iq_to_complex_sources(output_iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if output_iq.ndim != 2 or output_iq.shape[0] < 4:
        raise ValueError(f"Expected model output shape (>=4, T), got {output_iq.shape}")
    return (
        output_iq[0].astype(np.float32) + 1j * output_iq[1].astype(np.float32),
        output_iq[2].astype(np.float32) + 1j * output_iq[3].astype(np.float32),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a learned separator on MIT backend .iqdata frames.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--alphaIndex", type=int, required=True)
    parser.add_argument("--frameLen", type=int, required=True)
    parser.add_argument("--setIndex", type=int, required=True)
    parser.add_argument("--nFrames", type=int, required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--n_rx", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--receiver_mode",
        default="auto",
        choices=["auto", "first"],
        help="How to adapt MIT n_rx receivers to the checkpoint input channel count.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    state_dict = checkpoint_state_dict(checkpoint_path)
    expected_real_channels = infer_expected_input_channels(state_dict)
    if expected_real_channels % 2 != 0:
        raise ValueError(f"Checkpoint expects odd real channel count: {expected_real_channels}")
    expected_rx = expected_real_channels // 2
    if expected_rx > args.n_rx:
        raise ValueError(
            f"Checkpoint expects {expected_rx} receivers but MIT input only provides n_rx={args.n_rx}"
        )

    model_name = normalize_model_name(args.model_name)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = ExperimentConfig(model_name=model_name, n_rx=expected_rx, dropout=0.0)
    model = get_model(config, device)
    load_model_weights(model, checkpoint_path, device)
    model.eval()

    print(
        f"MIT learned inference: model={model_name}, checkpoint={checkpoint_path}, "
        f"MIT n_rx={args.n_rx}, model receivers={expected_rx}"
    )
    if expected_rx < args.n_rx:
        print(f"Using first {expected_rx} of {args.n_rx} MIT receiver streams for this checkpoint.")

    written = 0
    with torch.no_grad():
        for frame in range(1, args.nFrames + 1):
            input_path = (
                input_dir
                / f"input_frameLen_{args.frameLen}_setIndex_{args.setIndex}_alphaIndex_{args.alphaIndex}_frame{frame}.iqdata"
            )
            mixture = read_mit_iqdata(input_path, args.n_rx)
            model_mixture = mixture[:expected_rx]
            x = complex_matrix_to_iq_channels(model_mixture)
            x_tensor = torch.from_numpy(x).unsqueeze(0).float().to(device)
            y_hat = model(x_tensor).detach().cpu().numpy()[0]
            source_a, source_b = output_iq_to_complex_sources(y_hat)

            base = (
                f"_frameLen_{args.frameLen}_setIndex_{args.setIndex}_"
                f"alphaIndex_{args.alphaIndex}_frame{frame}"
            )
            write_mit_iqdata(output_dir / f"outputA{base}", source_a)
            write_mit_iqdata(output_dir / f"outputB{base}", source_b)
            written += 2

    print(f"Wrote {written} MIT-compatible separated .iqdata files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

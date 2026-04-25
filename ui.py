import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import BENCHMARK_DESCRIPTIONS, BENCHMARK_PRESETS, ExperimentConfig, benchmark_config
from main import (
    build_family_config,
    build_signal_configs,
    build_source_configs,
    checkpoint_candidates,
    default_checkpoint_path,
    model_is_trainable,
    run_experiment,
    supported_model_names,
)
from utils.data_utils.dataset import SyntheticRFDataset
from utils.data_utils.generator import (
    RFMixtureGenerator,
    SUPPORTED_SOURCE_TYPES,
)
from utils.plot_utils.plotting_utils import plot_3d_wave, plot_spectrogram


st.set_page_config(page_title="RF Separation Dashboard", layout="wide")

SOURCE_TYPE_ORDER = [
    "QPSK",
    "BPSK",
    "8PSK",
    "16QAM",
    "PAM",
    "ASK",
    "MULTITONE",
    "COLORED_NOISE",
    "CHIRP",
    "BURSTY",
    "RECORDED",
]
SOURCE_TYPE_OPTIONS = [name for name in SOURCE_TYPE_ORDER if name in SUPPORTED_SOURCE_TYPES]
MIXING_MODE_OPTIONS = ["phase_only", "random_complex", "real", "identity"]
PRESET_OPTIONS = ["manual"] + sorted(BENCHMARK_PRESETS)
DEFAULT_PRESET = "benchmark_qpsk_baseline"


if "results_by_model" not in st.session_state:
    st.session_state["results_by_model"] = {}
if "last_run_order" not in st.session_state:
    st.session_state["last_run_order"] = []


def relative_label(path):
    path = Path(path)
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def default_models_for_nrx(n_rx):
    defaults = ["FastICA", "Hybrid", "IQ_CNN"] if n_rx >= 2 else ["Hybrid", "IQ_CNN"]
    return [model for model in defaults if model in supported_model_names(n_rx)]


def metric_rows(results_by_model):
    rows = []
    for model_name, result in results_by_model.items():
        rows.append(
            {
                "Model": model_name,
                "Mode": result.get("mode", ""),
                "SDR (dB)": round(result.get("final_sdr_db", 0.0), 2),
                "Symbol Acc (%)": round(result.get("final_symbol_accuracy", 0.0) * 100, 2),
                "MSE": round(result.get("final_mse", 0.0), 6),
                "Checkpoint": relative_label(result["checkpoint_path"])
                if result.get("checkpoint_path")
                else "baseline",
            }
        )
    return rows


def log_rows():
    rows = []
    for path in sorted((ROOT_DIR / "logs").glob("*_results.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        rows.append(
            {
                "Model": path.name.replace("_results.json", ""),
                "SDR (dB)": round(data.get("final_sdr_db", 0.0), 2),
                "Symbol Acc (%)": round(data.get("final_symbol_accuracy", 0.0) * 100, 2),
                "MSE": round(data.get("final_mse", 0.0), 6),
                "Source": relative_label(path),
            }
        )
    return rows


def checkpoint_rows(model_names):
    rows = []
    for model_name in model_names:
        if not model_is_trainable(model_name):
            rows.append({"Model": model_name, "Status": "baseline", "Path": ""})
            continue
        found = next((path for path in checkpoint_candidates(model_name) if path.exists()), None)
        rows.append(
            {
                "Model": model_name,
                "Status": "ready" if found else "missing",
                "Path": relative_label(found) if found else relative_label(default_checkpoint_path(model_name)),
            }
        )
    return rows


def show_image(path, caption):
    if not path:
        return
    path = Path(path)
    if path.is_file():
        st.image(str(path), caption=caption, use_container_width=True)


def load_markdown(path):
    path = Path(path)
    if not path.exists():
        return f"Missing documentation file: `{relative_label(path)}`"
    return path.read_text(encoding="utf-8")


def preset_rows():
    rows = []
    for name in sorted(BENCHMARK_PRESETS):
        cfg = BENCHMARK_PRESETS[name]
        rows.append(
            {
                "Preset": name,
                "n_rx": cfg.n_rx,
                "SOI": cfg.source_a_type if not cfg.source_family_mode else ",".join(cfg.source_a_family),
                "Interferer": cfg.source_b_type if not cfg.source_family_mode else ",".join(cfg.source_b_family),
                "Family mode": cfg.source_family_mode,
                "MIT aligned": cfg.mit_aligned,
                "Mixing": cfg.mixing_mode,
                "Description": BENCHMARK_DESCRIPTIONS.get(name, ""),
            }
        )
    return rows


def config_to_state(cfg):
    snr_value = float(cfg.snr_db) if cfg.snr_db is not None else 25.0
    return {
        "n_rx": int(cfg.n_rx),
        "epochs": int(cfg.epochs),
        "train_examples": int(cfg.train_examples),
        "val_examples": int(cfg.val_examples),
        "test_examples": int(cfg.test_examples),
        "batch_size": int(cfg.batch_size),
        "lr": float(cfg.lr),
        "dropout": float(cfg.dropout),
        "source_family_mode": bool(cfg.source_family_mode),
        "mit_aligned": bool(cfg.mit_aligned),
        "source_a_type": cfg.source_a_type,
        "source_b_type": cfg.source_b_type,
        "source_a_family": list(cfg.source_a_family),
        "source_b_family": list(cfg.source_b_family),
        "modulation_family_sweep": bool(cfg.modulation_family_sweep),
        "use_cross_val": bool(cfg.use_cross_val),
        "num_symbols": int(cfg.num_symbols),
        "n_symbols_range": tuple(int(v) for v in (cfg.n_symbols_range or (cfg.num_symbols, cfg.num_symbols))),
        "samples_per_symbol": int(cfg.samples_per_symbol),
        "samples_per_symbol_choices": list(cfg.samples_per_symbol_choices or (cfg.samples_per_symbol,)),
        "rolloff": float(cfg.rolloff),
        "rolloff_range": tuple(float(v) for v in (cfg.rolloff_range or (cfg.rolloff, cfg.rolloff))),
        "rrc_span_symbols": int(cfg.rrc_span_symbols),
        "normalize_power": bool(cfg.normalize_power),
        "alpha": float(cfg.alpha),
        "alpha_range": tuple(float(v) for v in (cfg.alpha_range or (cfg.alpha, cfg.alpha))),
        "random_phase": bool(cfg.random_phase),
        "phase_shift_deg": int(cfg.phase_shift_deg),
        "interference_phase_shift": int(cfg.interference_phase_shift),
        "mixing_mode": cfg.mixing_mode,
        "random_complex_mixing": bool(cfg.random_complex_mixing),
        "timing_offset": int(cfg.timing_offset),
        "timing_offset_range": tuple(int(v) for v in (cfg.timing_offset_range or (cfg.timing_offset, cfg.timing_offset))),
        "carrier_offset": float(cfg.carrier_offset),
        "carrier_offset_range": tuple(float(v) for v in (cfg.carrier_offset_range or (cfg.carrier_offset, cfg.carrier_offset))),
        "phase_mismatch_deg": float(cfg.phase_mismatch_deg),
        "phase_mismatch_range_deg": tuple(
            float(v) for v in (cfg.phase_mismatch_range_deg or (cfg.phase_mismatch_deg, cfg.phase_mismatch_deg))
        ),
        "amplitude_imbalance_db": float(cfg.amplitude_imbalance_db),
        "amplitude_imbalance_db_range": tuple(
            float(v)
            for v in (cfg.amplitude_imbalance_db_range or (cfg.amplitude_imbalance_db, cfg.amplitude_imbalance_db))
        ),
        "noise_enabled": bool(cfg.noise_enabled),
        "noise_mode": "Variance" if cfg.noise_variance is not None and cfg.snr_db is None else "SNR",
        "snr_db": snr_value,
        "snr_db_range": tuple(float(v) for v in (cfg.snr_db_range or (snr_value, snr_value))),
        "noise_variance": float(cfg.noise_variance if cfg.noise_variance is not None else 0.01),
        "dataset_path": cfg.dataset_path,
        "save_dir": cfg.save_dir,
        "log_dir": cfg.log_dir,
        "model_dir": cfg.model_dir,
        "transfer_eval_target": cfg.transfer_eval_target or "",
        "fs": float(cfg.fs),
        "symbol_rate": float(cfg.symbol_rate) if cfg.symbol_rate is not None else 1.0,
        "T1": float(cfg.T1),
        "T2": float(cfg.T2) if cfg.T2 is not None else 0.0,
    }


def apply_preset_state(preset_name):
    if preset_name == "manual":
        return
    cfg = benchmark_config(preset_name)
    for key, value in config_to_state(cfg).items():
        st.session_state[key] = value
    st.session_state["active_benchmark_preset"] = preset_name
    st.session_state["selected_models"] = default_models_for_nrx(cfg.n_rx)


def initialize_state():
    if st.session_state.get("_ui_initialized"):
        return
    st.session_state["benchmark_preset_name"] = DEFAULT_PRESET
    apply_preset_state(DEFAULT_PRESET)
    st.session_state.setdefault("active_benchmark_preset", DEFAULT_PRESET)
    st.session_state.setdefault("selected_models", default_models_for_nrx(st.session_state["n_rx"]))
    st.session_state.setdefault("save_train", 2000)
    st.session_state.setdefault("save_val", 200)
    st.session_state.setdefault("save_test", 200)
    st.session_state.setdefault("_ui_initialized", True)


def consume_pending_preset():
    pending = st.session_state.pop("_pending_preset_name", None)
    if pending is None:
        return
    st.session_state["benchmark_preset_name"] = pending
    apply_preset_state(pending)


def range_or_none(values, enabled):
    if not enabled:
        return None
    return tuple(values)


def choices_or_none(values, enabled):
    if not enabled:
        return None
    cleaned = tuple(sorted(int(v) for v in values))
    return cleaned or None


def current_models_for_nrx(n_rx):
    models = supported_model_names(n_rx)
    selected = [model for model in st.session_state.get("selected_models", []) if model in models]
    if not selected:
        selected = default_models_for_nrx(n_rx)
        st.session_state["selected_models"] = selected
    return models, selected


def make_config(mode, model_name, model_path=None, custom_symbols=None):
    state = st.session_state
    mit_aligned = bool(state.get("mit_aligned", False))
    family_mode = bool(state.get("source_family_mode", False) or mit_aligned)
    n_rx = 4 if mit_aligned else int(state.get("n_rx", 2))

    noise_enabled = bool(state.get("noise_enabled", False))
    noise_mode = state.get("noise_mode", "SNR")
    active_snr_db = float(state.get("snr_db", 25.0)) if noise_enabled and noise_mode == "SNR" else None
    active_noise_variance = (
        float(state.get("noise_variance", 0.01)) if noise_enabled and noise_mode == "Variance" else None
    )

    source_a_type = state.get("source_a_type", "QPSK")
    source_b_type = state.get("source_b_type", "QPSK")
    source_a_family = tuple(state.get("source_a_family", [source_a_type])) or (source_a_type,)
    source_b_family = tuple(state.get("source_b_family", [source_b_type])) or (source_b_type,)

    benchmark_name = state.get("benchmark_preset_name", DEFAULT_PRESET)
    if benchmark_name == "manual":
        benchmark_name = "manual"

    return ExperimentConfig(
        mode=mode,
        model_name=model_name,
        model_path=str(model_path) if model_path else None,
        dropout=float(state.get("dropout", 0.1)),
        batch_size=int(state.get("batch_size", 16)),
        epochs=int(state.get("epochs", 10)),
        lr=float(state.get("lr", 1e-3)),
        modulation=source_a_type,
        source_a_type=source_a_type,
        source_b_type=source_b_type,
        source_family_mode=family_mode,
        source_a_family=source_a_family,
        source_b_family=source_b_family,
        modulation_family_sweep=bool(state.get("modulation_family_sweep", False)),
        use_cross_val=bool(state.get("use_cross_val", False)),
        num_symbols=int(state.get("num_symbols", 400)),
        n_rx=n_rx,
        samples_per_symbol=int(state.get("samples_per_symbol", 2)),
        rolloff=float(state.get("rolloff", 0.25)),
        rrc_span_symbols=int(state.get("rrc_span_symbols", 12)),
        normalize_power=bool(state.get("normalize_power", True)),
        n_symbols_range=range_or_none(state.get("n_symbols_range", (400, 400)), family_mode),
        samples_per_symbol_choices=choices_or_none(
            state.get("samples_per_symbol_choices", [state.get("samples_per_symbol", 2)]),
            family_mode,
        ),
        rolloff_range=range_or_none(state.get("rolloff_range", (0.25, 0.25)), family_mode),
        alpha=float(state.get("alpha", 1.0)),
        noise_alpha=float(state.get("alpha", 1.0)),
        random_phase=bool(state.get("random_phase", False)),
        phase_shift_deg=int(state.get("phase_shift_deg", 5)),
        interference_phase_shift=int(state.get("interference_phase_shift", 0)),
        mixing_mode=state.get("mixing_mode", "phase_only"),
        random_complex_mixing=bool(state.get("random_complex_mixing", False)),
        timing_offset=int(state.get("timing_offset", 0)),
        timing_offset_range=range_or_none(state.get("timing_offset_range", (0, 0)), family_mode),
        carrier_offset=float(state.get("carrier_offset", 0.0)),
        carrier_offset_range=range_or_none(state.get("carrier_offset_range", (0.0, 0.0)), family_mode),
        phase_mismatch_deg=float(state.get("phase_mismatch_deg", 0.0)),
        phase_mismatch_range_deg=range_or_none(
            state.get("phase_mismatch_range_deg", (0.0, 0.0)),
            family_mode,
        ),
        amplitude_imbalance_db=float(state.get("amplitude_imbalance_db", 0.0)),
        amplitude_imbalance_db_range=range_or_none(
            state.get("amplitude_imbalance_db_range", (0.0, 0.0)),
            family_mode,
        ),
        alpha_range=range_or_none(state.get("alpha_range", (1.0, 1.0)), family_mode),
        snr_db_range=range_or_none(state.get("snr_db_range", (25.0, 25.0)), family_mode and noise_mode == "SNR"),
        dataset_path=str(state.get("dataset_path", "data")),
        train_examples=int(state.get("train_examples", 512)),
        val_examples=int(state.get("val_examples", 128)),
        test_examples=int(state.get("test_examples", 200)),
        save_dir=str(state.get("save_dir", "visualizations")),
        log_dir=str(state.get("log_dir", "logs")),
        model_dir=str(state.get("model_dir", "pytorch_models")),
        benchmark_name=benchmark_name,
        transfer_eval_target=(state.get("transfer_eval_target", "") or None),
        mit_aligned=mit_aligned,
        noise_enabled=noise_enabled,
        snr_db=active_snr_db,
        noise_variance=active_noise_variance,
        custom_symbols=custom_symbols,
        fs=float(state.get("fs", 1.0)),
        symbol_rate=float(state.get("symbol_rate", 1.0)) or None,
        T1=float(state.get("T1", 0.0)),
        T2=float(state.get("T2", 0.0)) or None,
    )


def config_preview(config):
    return {
        "benchmark_name": config.benchmark_name,
        "mit_aligned": config.mit_aligned,
        "source_family_mode": config.source_family_mode,
        "source_a_type": config.source_a_type,
        "source_b_type": config.source_b_type,
        "source_a_family": list(config.source_a_family),
        "source_b_family": list(config.source_b_family),
        "n_rx": config.n_rx,
        "num_symbols": config.num_symbols,
        "n_symbols_range": config.n_symbols_range,
        "samples_per_symbol": config.samples_per_symbol,
        "samples_per_symbol_choices": config.samples_per_symbol_choices,
        "rolloff": config.rolloff,
        "rolloff_range": config.rolloff_range,
        "alpha": config.alpha,
        "alpha_range": config.alpha_range,
        "noise_enabled": config.noise_enabled,
        "snr_db": config.snr_db,
        "noise_variance": config.noise_variance,
        "snr_db_range": config.snr_db_range,
        "mixing_mode": config.mixing_mode,
        "timing_offset": config.timing_offset,
        "timing_offset_range": config.timing_offset_range,
        "carrier_offset": config.carrier_offset,
        "carrier_offset_range": config.carrier_offset_range,
        "phase_mismatch_deg": config.phase_mismatch_deg,
        "phase_mismatch_range_deg": config.phase_mismatch_range_deg,
        "amplitude_imbalance_db": config.amplitude_imbalance_db,
        "amplitude_imbalance_db_range": config.amplitude_imbalance_db_range,
    }


def render_result(model_name, result):
    st.subheader(model_name)
    c1, c2, c3 = st.columns(3)
    c1.metric("SDR", f"{result.get('final_sdr_db', 0.0):.2f} dB")
    c2.metric("Symbol accuracy", f"{result.get('final_symbol_accuracy', 0.0) * 100:.1f}%")
    c3.metric("MSE", f"{result.get('final_mse', 0.0):.6f}")

    signal = result.get("sample_signal")
    if signal is not None:
        p1, p2 = st.columns(2)
        with p1:
            st.plotly_chart(
                plot_3d_wave(signal, title=f"{model_name} recovered Source A IQ"),
                use_container_width=True,
            )
        with p2:
            st.pyplot(plot_spectrogram(signal), clear_figure=True)

    artifacts = result.get("artifacts", {})
    g1, g2 = st.columns(2)
    with g1:
        show_image(artifacts.get("source_a_symbols", ""), "Source A symbol recovery")
        show_image(artifacts.get("source_b_symbols", ""), "Source B symbol recovery")
    with g2:
        show_image(artifacts.get("separation", ""), "Waveform separation")
        show_image(artifacts.get("data_pipeline", ""), "Synthetic RF data pipeline")

    snapshots = [Path(path) for path in artifacts.get("training_snapshots", []) if Path(path).exists()]
    if snapshots:
        st.markdown("#### Training snapshots")
        snap_cols = st.columns(min(2, len(snapshots)))
        for idx, path in enumerate(snapshots):
            with snap_cols[idx % len(snap_cols)]:
                st.image(str(path), caption=path.stem, use_container_width=True)


initialize_state()
consume_pending_preset()

st.title("RF Signal Separation Dashboard")
st.caption(
    "Train, evaluate, and generate datasets for the original QPSK setup, the expanded waveform-family generator, and MIT-aligned synthetic regimes."
)

with st.sidebar:
    st.header("Run Settings")

    preset_name = st.selectbox(
        "Benchmark preset",
        PRESET_OPTIONS,
        index=PRESET_OPTIONS.index(st.session_state.get("benchmark_preset_name", DEFAULT_PRESET)),
        key="benchmark_preset_name",
        help="Choose a named benchmark template, then optionally fine-tune the controls below.",
    )
    preset_cols = st.columns([1, 1])
    with preset_cols[0]:
        if st.button("Apply preset", use_container_width=True, disabled=preset_name == "manual"):
            apply_preset_state(preset_name)
    with preset_cols[1]:
        if st.button("Reset baseline", use_container_width=True):
            st.session_state["_pending_preset_name"] = DEFAULT_PRESET
            st.rerun()
    if preset_name != "manual":
        st.caption(BENCHMARK_DESCRIPTIONS.get(preset_name, ""))
    st.caption(f"Applied preset: `{st.session_state.get('active_benchmark_preset', 'manual')}`")

    with st.expander("Data Regime", expanded=True):
        st.checkbox("MIT-aligned synthetic mode", key="mit_aligned")
        st.checkbox("Sample waveform families per example", key="source_family_mode")
        st.checkbox("Modulation-family sweep metadata", key="modulation_family_sweep")
        st.checkbox("Cross-validation flag", key="use_cross_val")
        st.selectbox("Receive antennas", [1, 2, 4], key="n_rx")
        st.selectbox("Source A waveform", SOURCE_TYPE_OPTIONS, key="source_a_type")
        st.selectbox("Source B / interferer waveform", SOURCE_TYPE_OPTIONS, key="source_b_type")
        st.multiselect(
            "Source A family",
            SOURCE_TYPE_OPTIONS,
            key="source_a_family",
            help="Used when waveform-family mode is enabled.",
        )
        st.multiselect(
            "Source B family",
            SOURCE_TYPE_OPTIONS,
            key="source_b_family",
            help="Used when waveform-family mode is enabled.",
        )
        st.text_input("Transfer target tag", key="transfer_eval_target")
        if st.session_state.get("mit_aligned"):
            st.info("MIT-aligned mode keeps the existing workflow but forces MIT-style synthetic defaults when you run.")

    effective_n_rx = 4 if st.session_state.get("mit_aligned") else int(st.session_state.get("n_rx", 2))
    available_models, selected_models = current_models_for_nrx(effective_n_rx)
    st.multiselect("Models", available_models, key="selected_models")

    with st.expander("Training", expanded=False):
        st.slider("Epochs", 1, 300, key="epochs")
        st.number_input("Train examples", 64, 50000, step=64, key="train_examples")
        st.number_input("Validation examples", 32, 10000, step=32, key="val_examples")
        st.number_input("Test examples", 32, 20000, step=32, key="test_examples")
        st.select_slider("Batch size", options=[8, 16, 32, 64], key="batch_size")
        st.selectbox("Learning rate", [1e-2, 1e-3, 5e-4, 1e-4], key="lr")
        st.slider("Dropout", 0.0, 0.5, key="dropout")

    family_mode_active = bool(st.session_state.get("source_family_mode") or st.session_state.get("mit_aligned"))

    with st.expander("Signal", expanded=True):
        st.slider("Fixed symbols per source", 4, 2048, step=4, key="num_symbols")
        st.slider("Symbol range for family mode", 4, 2048, step=4, key="n_symbols_range")
        st.select_slider("Fixed samples per symbol", options=[1, 2, 4, 8, 16], key="samples_per_symbol")
        st.multiselect(
            "Samples per symbol choices",
            [1, 2, 4, 8, 16],
            key="samples_per_symbol_choices",
            help="Used when family mode is enabled.",
        )
        st.slider("Fixed RRC rolloff", 0.05, 1.0, step=0.05, key="rolloff")
        st.slider("Rolloff range for family mode", 0.05, 1.0, step=0.05, key="rolloff_range")
        st.slider("RRC span", 2, 20, key="rrc_span_symbols")
        st.checkbox("Normalize source power", key="normalize_power")

    with st.expander("Channel + Mixing", expanded=True):
        st.slider("Fixed interference gain alpha", 0.0, 4.0, step=0.05, key="alpha")
        st.slider("Alpha range for family mode", 0.0, 4.0, step=0.05, key="alpha_range")
        st.selectbox("Mixing mode", MIXING_MODE_OPTIONS, key="mixing_mode")
        st.checkbox("Force random complex mixing", key="random_complex_mixing")
        st.checkbox("Random receive phase", key="random_phase")
        st.slider("Antenna phase step (deg)", 0, 45, key="phase_shift_deg")
        st.slider("Interference phase (deg)", -180, 180, key="interference_phase_shift")
        st.slider("Fixed timing offset", -16, 16, key="timing_offset")
        st.slider("Timing-offset range", -16, 16, key="timing_offset_range")
        st.slider("Fixed carrier offset", -0.05, 0.05, step=0.001, key="carrier_offset")
        st.slider("Carrier-offset range", -0.05, 0.05, step=0.001, key="carrier_offset_range")
        st.slider("Fixed phase mismatch (deg)", -45.0, 45.0, step=0.5, key="phase_mismatch_deg")
        st.slider("Phase-mismatch range (deg)", -45.0, 45.0, step=0.5, key="phase_mismatch_range_deg")
        st.slider("Fixed amplitude imbalance (dB)", -6.0, 6.0, step=0.25, key="amplitude_imbalance_db")
        st.slider("Amplitude-imbalance range (dB)", -6.0, 6.0, step=0.25, key="amplitude_imbalance_db_range")

    with st.expander("Noise", expanded=False):
        st.checkbox("Enable noise", key="noise_enabled")
        st.radio("Noise control", ["SNR", "Variance"], horizontal=True, key="noise_mode")
        st.slider("Fixed SNR (dB)", -30.0, 50.0, key="snr_db")
        st.slider("SNR range for family mode", -30.0, 50.0, key="snr_db_range")
        st.slider("Fixed noise variance", 0.0, 1.0, step=0.005, key="noise_variance")

    with st.expander("Paths + Metadata", expanded=False):
        st.text_input("Dataset path", key="dataset_path")
        st.text_input("Visualization dir", key="save_dir")
        st.text_input("Log dir", key="log_dir")
        st.text_input("Model dir", key="model_dir")
        st.number_input("Sample rate metadata", 0.1, 100.0, step=0.1, key="fs")
        st.number_input("Symbol rate metadata", 0.0, 100.0, step=0.1, key="symbol_rate")
        st.number_input("T1 metadata", -100.0, 100.0, step=0.1, key="T1")
        st.number_input("T2 metadata", -100.0, 100.0, step=0.1, key="T2")

    with st.expander("Dataset Builder", expanded=False):
        st.number_input("Saved train examples", 64, 50000, step=64, key="save_train")
        st.number_input("Saved validation examples", 32, 10000, step=32, key="save_val")
        st.number_input("Saved test examples", 32, 10000, step=32, key="save_test")
        if st.button("Generate dataset", use_container_width=True):
            cfg = make_config("generate", selected_models[0] if selected_models else "Hybrid")
            qpsk_cfg, noise_cfg, mix_cfg, _ = build_signal_configs(cfg)
            source_a_cfg, source_b_cfg = build_source_configs(cfg)
            family_cfg = build_family_config(cfg)
            dataset = SyntheticRFDataset(
                num_examples=1,
                generator=RFMixtureGenerator(seed=42),
                qpsk_cfg_soi=qpsk_cfg,
                qpsk_cfg_int=qpsk_cfg,
                noise_cfg=noise_cfg,
                mix_cfg=mix_cfg,
                custom_symbols=None,
                source_a_cfg=source_a_cfg,
                source_b_cfg=source_b_cfg,
                family_cfg=family_cfg,
            )
            dataset_root = Path(cfg.dataset_path)
            if not dataset_root.is_absolute():
                dataset_root = ROOT_DIR / dataset_root
            with st.spinner("Saving synthetic dataset"):
                dataset.save_splits(
                    train_size=int(st.session_state["save_train"]),
                    val_size=int(st.session_state["save_val"]),
                    test_size=int(st.session_state["save_test"]),
                    root_dir=str(dataset_root),
                    overwrite=True,
                )
            st.success(f"Dataset generated at {relative_label(dataset_root)}")
            if family_mode_active:
                st.caption("Each saved example is sampled from the configured waveform family rather than one fixed regime.")

    st.dataframe(pd.DataFrame(checkpoint_rows(available_models)), use_container_width=True, hide_index=True)

tab_run, tab_plots, tab_board, tab_gallery, tab_docs = st.tabs(
    ["Run Models", "Plots", "Leaderboard", "Artifact Gallery", "Docs"]
)

with tab_run:
    run_mode = st.radio(
        "Action",
        ["Evaluate checkpoints", "Train selected models"],
        horizontal=True,
    )
    auto_train_missing = st.checkbox(
        "Train missing checkpoints before evaluation",
        value=False,
        disabled=run_mode != "Evaluate checkpoints",
    )
    custom_symbols = st.text_input(
        "Custom Source A bits or text",
        value="",
        help="Used by the legacy QPSK path. For non-QPSK source families this field is ignored.",
    )

    uploaded_path = None
    manual_path = None
    if len(selected_models) == 1 and model_is_trainable(selected_models[0]):
        up1, up2 = st.columns(2)
        with up1:
            uploaded = st.file_uploader("Checkpoint upload", type=["pt", "pth"])
            if uploaded is not None:
                upload_dir = ROOT_DIR / "weights"
                upload_dir.mkdir(exist_ok=True)
                uploaded_path = upload_dir / f"ui_{selected_models[0]}.pt"
                uploaded_path.write_bytes(uploaded.read())
        with up2:
            manual_value = st.text_input("Checkpoint path", value="")
            manual_path = Path(manual_value) if manual_value.strip() else None

    preview_model = selected_models[0] if selected_models else "Hybrid"
    preview_cfg = make_config(
        "train" if run_mode == "Train selected models" else "evaluate",
        preview_model,
        model_path=manual_path or uploaded_path,
        custom_symbols=custom_symbols.strip() or None,
    )
    with st.expander("Current configuration preview", expanded=False):
        st.json(config_preview(preview_cfg), expanded=False)

    if st.button("Run selected models", type="primary", use_container_width=True):
        if not selected_models:
            st.warning("Select at least one model.")
        else:
            run_results = {}
            run_order = []
            for model_name in selected_models:
                trainable = model_is_trainable(model_name)
                mode = "train" if run_mode == "Train selected models" and trainable else "evaluate"
                model_path = manual_path or uploaded_path

                if mode == "evaluate" and trainable and model_path is None:
                    model_path = default_checkpoint_path(model_name)
                    if model_path is not None and not Path(model_path).exists():
                        if auto_train_missing:
                            mode = "train"
                            model_path = None
                        else:
                            st.warning(f"Skipping {model_name}: no checkpoint found.")
                            continue

                with st.expander(f"{model_name} run", expanded=True):
                    progress = st.progress(0)
                    status = st.empty()
                    chart = None
                    if mode == "train":
                        chart = st.line_chart(pd.DataFrame(columns=["Train MSE", "Val MSE"]))

                    def callback(epoch, train_loss, val_loss):
                        progress.progress(min(1.0, epoch / max(1, st.session_state["epochs"])))
                        status.write(
                            f"Epoch {epoch}/{st.session_state['epochs']} | Train {train_loss:.6f} | Val {val_loss:.6f}"
                        )
                        chart.add_rows(
                            pd.DataFrame(
                                {"Train MSE": [train_loss], "Val MSE": [val_loss]},
                                index=[epoch],
                            )
                        )

                    try:
                        cfg = make_config(
                            mode,
                            model_name,
                            model_path=model_path,
                            custom_symbols=custom_symbols.strip() or None,
                        )
                        result = run_experiment(cfg, ui_callback=callback if mode == "train" else None)
                        progress.progress(1.0)
                        status.write("Complete")
                        run_results[model_name] = result
                        run_order.append(model_name)
                    except Exception as exc:
                        st.error(f"{model_name} failed: {exc}")

            st.session_state["results_by_model"].update(run_results)
            st.session_state["last_run_order"] = run_order

    rows = metric_rows(st.session_state["results_by_model"])
    if rows:
        st.dataframe(
            pd.DataFrame(rows).sort_values("SDR (dB)", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


with tab_plots:
    results = st.session_state["results_by_model"]
    if not results:
        st.info("Run or evaluate a model to populate plots.")
    else:
        ordered_names = st.session_state["last_run_order"] or list(results.keys())
        model_name = st.selectbox("Result", [name for name in ordered_names if name in results])
        render_result(model_name, results[model_name])


with tab_board:
    current_rows = metric_rows(st.session_state["results_by_model"])
    if current_rows:
        st.subheader("Current session")
        st.dataframe(
            pd.DataFrame(current_rows).sort_values("SDR (dB)", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    saved_rows = log_rows()
    if saved_rows:
        st.subheader("Saved logs")
        st.dataframe(
            pd.DataFrame(saved_rows).sort_values("SDR (dB)", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    report_csv = ROOT_DIR / "reporting_pipeline" / "outputs" / "tables" / "model_comparison_reference.csv"
    if report_csv.exists():
        st.subheader("Final report reference")
        st.dataframe(pd.read_csv(report_csv), use_container_width=True)


with tab_gallery:
    gallery_roots = [
        ROOT_DIR / "visualizations" / f"{effective_n_rx}-channel",
        ROOT_DIR / "visualizations",
        ROOT_DIR / "reporting_pipeline" / "outputs" / "figures",
        ROOT_DIR / "reporting_pipeline" / "outputs" / "phase2_two_channel_reference" / "figures",
        ROOT_DIR / "reporting_pipeline" / "outputs" / "iqcnn_2ch_loss_study",
        ROOT_DIR / "reporting_pipeline" / "outputs" / "iqcnn_2ch_full_train",
    ]
    images = []
    for root in gallery_roots:
        if root.exists():
            images.extend(sorted(root.glob("*.png")))

    if not images:
        st.info("No plot artifacts found.")
    else:
        selected_image = st.selectbox("Image", images, format_func=relative_label)
        st.image(str(selected_image), caption=relative_label(selected_image), use_container_width=True)


with tab_docs:
    st.subheader("Dashboard Guide")
    st.markdown(load_markdown(ROOT_DIR / "docs" / "STREAMLIT_GUIDE.md"))
    st.divider()
    st.subheader("Benchmark Presets")
    st.dataframe(pd.DataFrame(preset_rows()), use_container_width=True, hide_index=True)
    st.divider()
    st.subheader("Repository README")
    st.markdown(load_markdown(ROOT_DIR / "README.md"))

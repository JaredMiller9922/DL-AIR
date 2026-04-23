import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import ExperimentConfig
from main import (
    checkpoint_candidates,
    default_checkpoint_path,
    model_is_trainable,
    run_experiment,
    supported_model_names,
)
from utils.data_utils.dataset import SyntheticRFDataset
from utils.data_utils.generator import MixtureConfig, NoiseConfig, QPSKConfig, RFMixtureGenerator
from utils.plot_utils.plotting_utils import plot_3d_wave, plot_spectrogram


st.set_page_config(page_title="RF Separation Dashboard", layout="wide")

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


def make_config(mode, model_name, model_path=None, custom_symbols=None):
    active_noise_variance = noise_variance if noise_enabled and noise_mode == "Variance" else None
    active_snr_db = snr_db if noise_enabled and noise_mode == "SNR" else None
    return ExperimentConfig(
        mode=mode,
        model_name=model_name,
        model_path=str(model_path) if model_path else None,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        dropout=dropout,
        num_symbols=num_symbols,
        n_rx=n_rx,
        samples_per_symbol=samples_per_symbol,
        rolloff=rolloff,
        rrc_span_symbols=rrc_span_symbols,
        normalize_power=True,
        alpha=alpha,
        noise_enabled=noise_enabled,
        snr_db=active_snr_db,
        noise_variance=active_noise_variance,
        random_phase=random_phase,
        phase_shift_deg=phase_shift_deg,
        interference_phase_shift=interference_phase_shift,
        train_examples=train_examples,
        val_examples=val_examples,
        save_dir="visualizations",
        log_dir="logs",
        model_dir="pytorch_models",
        custom_symbols=custom_symbols,
    )


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


st.title("RF Signal Separation Dashboard")

with st.sidebar:
    st.header("Run Settings")
    n_rx = st.selectbox("Receive antennas", [1, 2, 4], index=1)
    available_models = supported_model_names(n_rx)
    default_selection = ["FastICA", "Hybrid", "IQ_CNN"] if n_rx >= 2 else ["Hybrid", "IQ_CNN"]
    selected_models = st.multiselect(
        "Models",
        available_models,
        default=[model for model in default_selection if model in available_models],
    )

    with st.expander("Training", expanded=True):
        epochs = st.slider("Epochs", 1, 300, 10)
        train_examples = st.number_input("Train examples", 64, 50000, 512, step=64)
        val_examples = st.number_input("Validation examples", 32, 10000, 128, step=32)
        batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64], value=16)
        lr = st.selectbox("Learning rate", [1e-2, 1e-3, 5e-4, 1e-4], index=1)
        dropout = st.slider("Dropout", 0.0, 0.5, 0.1)

    with st.expander("Signal", expanded=True):
        num_symbols = st.slider("QPSK symbols", 16, 1200, 400, step=16)
        samples_per_symbol = st.select_slider("Samples per symbol", options=[2, 4, 8], value=2)
        rolloff = st.slider("RRC rolloff", 0.05, 1.0, 0.25, step=0.05)
        rrc_span_symbols = st.slider("RRC span", 2, 20, 12)
        alpha = st.slider("Interference gain", 0.0, 4.0, 1.0, step=0.05)
        phase_shift_deg = st.slider("Antenna phase step", 0, 45, 5)
        interference_phase_shift = st.slider("Interference phase", -180, 180, 0)
        random_phase = st.checkbox("Random receive phase", value=False)
        noise_enabled = st.checkbox("AWGN", value=False)
        noise_mode = st.radio("Noise control", ["SNR", "Variance"], horizontal=True, disabled=not noise_enabled)
        snr_db = st.slider("SNR (dB)", -30.0, 50.0, 25.0, disabled=not noise_enabled or noise_mode != "SNR")
        noise_variance = st.slider(
            "Noise variance",
            0.0,
            1.0,
            0.01,
            disabled=not noise_enabled or noise_mode != "Variance",
        )

    with st.expander("Data Manager"):
        save_train = st.number_input("Saved train examples", 64, 50000, 2000, step=64)
        save_val = st.number_input("Saved validation examples", 32, 10000, 200, step=32)
        save_test = st.number_input("Saved test examples", 32, 10000, 200, step=32)
        if st.button("Generate dataset", use_container_width=True):
            cfg = make_config("generate", selected_models[0] if selected_models else "Hybrid")
            q_cfg = QPSKConfig(
                n_symbols=cfg.num_symbols,
                samples_per_symbol=cfg.samples_per_symbol,
                rolloff=cfg.rolloff,
                rrc_span_symbols=cfg.rrc_span_symbols,
                normalize_power=cfg.normalize_power,
                num_channels=cfg.n_rx,
            )
            mix_cfg = MixtureConfig(
                alpha=cfg.alpha,
                snr_db=cfg.snr_db,
                n_rx=cfg.n_rx,
                random_phase=cfg.random_phase,
                phase_shift_deg=cfg.phase_shift_deg,
                interference_phase_shift=cfg.interference_phase_shift,
            )
            noise_cfg = NoiseConfig(
                enabled=cfg.noise_enabled,
                snr_db=cfg.snr_db,
                sigma2=cfg.noise_variance,
            )
            dataset = SyntheticRFDataset(
                num_examples=1,
                generator=RFMixtureGenerator(seed=42),
                qpsk_cfg_soi=q_cfg,
                qpsk_cfg_int=q_cfg,
                noise_cfg=noise_cfg,
                mix_cfg=mix_cfg,
            )
            with st.spinner("Saving synthetic dataset"):
                dataset.save_splits(
                    train_size=int(save_train),
                    val_size=int(save_val),
                    test_size=int(save_test),
                    root_dir=str(ROOT_DIR / "data"),
                    overwrite=True,
                )
            st.success("Dataset generated")

    st.dataframe(pd.DataFrame(checkpoint_rows(available_models)), use_container_width=True, hide_index=True)


tab_run, tab_plots, tab_board, tab_gallery = st.tabs(
    ["Run Models", "Plots", "Leaderboard", "Artifact Gallery"]
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
    custom_symbols = st.text_input("Custom Source A bits or text", value="")

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
                        progress.progress(min(1.0, epoch / max(1, epochs)))
                        status.write(
                            f"Epoch {epoch}/{epochs} | Train {train_loss:.6f} | Val {val_loss:.6f}"
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
        ROOT_DIR / "visualizations" / f"{n_rx}-channel",
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

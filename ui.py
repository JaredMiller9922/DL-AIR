import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import glob
import json

# Ensure utils are findable
root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path:
    sys.path.append(root_path)

from config import ExperimentConfig
from main import run_experiment
# Import dataset tools specifically for the UI data generator button
from utils.data_utils.generator import RFMixtureGenerator, QPSKConfig, NoiseConfig, MixtureConfig
from utils.data_utils.dataset import SyntheticRFDataset
from utils.plot_utils.plotting_utils import plot_3d_wave, plot_spectrogram

st.set_page_config(page_title="RF Separation Dashboard", layout="wide")
st.title("📡 RF Signal Separation Dashboard")

if "results" not in st.session_state:
    st.session_state["results"] = None

# ==========================================
# SIDEBAR: SETTINGS & DATA GENERATOR
# ==========================================
st.sidebar.title("Architecture & Physics")
model_name = st.sidebar.selectbox("Model", ["Hybrid", "LSTM", "Linear", "IQ_CNN", "HTDemucs"])

with st.sidebar.expander("Hyperparameters", expanded=True):
    epochs = st.slider("Epochs", 1, 300, 50)
    batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64], value=16)
    lr = st.selectbox("Learning Rate", [1e-2, 1e-3, 5e-4, 1e-4], index=1)
    dropout = st.slider("Dropout", 0.0, 0.5, 0.1)

with st.sidebar.expander("Signal Physics", expanded=True):
    alpha = st.slider("Interference Gain (α)", 0.0, 2.0, 1.0)
    noise_enabled = st.checkbox("Enable AWGN", value=False)
    snr_db = st.slider("SNR (dB)", 0.0, 50.0, 25.0, disabled=not noise_enabled)

noise_variance = st.slider("Noise Variance (σ²)", 0.0, 1.0, 0.01, disabled=not noise_enabled)

st.sidebar.divider()
st.sidebar.subheader("Backend Data Manager")
if st.sidebar.button("Generate Synthetic Dataset", use_container_width=True):
    with st.spinner("Generating 24,000 files in ./data..."):
        gen = RFMixtureGenerator(seed=42)
        q_cfg = QPSKConfig()
        ds = SyntheticRFDataset(
            num_examples=1, generator=gen, qpsk_cfg_soi=q_cfg, qpsk_cfg_int=q_cfg,
            noise_cfg=NoiseConfig(enabled=noise_enabled), mix_cfg=MixtureConfig(alpha=alpha, snr_db=snr_db if noise_enabled else 100)
        )
        ds.save_splits(train_size=20000, val_size=2000, test_size=2000, root_dir="data", overwrite=True)
        st.sidebar.success("Data generated successfully!")

# ==========================================
# MAIN TABS
# ==========================================
tab_train, tab_eval, tab_logs = st.tabs(["Train Model", "Evaluate Symbols", "Metrics Leaderboard"])

# --- TAB 1: TRAIN ---
with tab_train:
    st.markdown("### Train on Backend Data")
    if st.button("Execute Training", use_container_width=True):
        config = ExperimentConfig(
            mode="train", model_name=model_name, batch_size=batch_size, epochs=epochs,
            lr=lr, dropout=dropout, alpha=alpha, snr_db=snr_db if noise_enabled else 100.0,
            noise_enabled=noise_enabled, noise_variance=noise_variance
        )

        st.markdown("#### Live Telemetry")
        col_prog, col_metrics = st.columns([3, 1])
        progress_bar = col_prog.progress(0)
        status_text = col_metrics.empty()
        
        # The Live Chart Placeholder
        chart_data = pd.DataFrame(columns=["Train MSE", "Val MSE"])
        loss_chart = st.line_chart(chart_data)

        def ui_callback(epoch, t_loss, v_loss):
            progress_bar.progress(epoch / config.epochs)
            status_text.markdown(f"**Epoch {epoch}/{config.epochs}**<br>Train: {t_loss:.6f}<br>Val: {v_loss:.6f}", unsafe_allow_html=True)
            # Add new data point to the live chart
            new_row = pd.DataFrame({"Train MSE": [t_loss], "Val MSE": [v_loss]}, index=[epoch])
            loss_chart.add_rows(new_row)

        try:
            st.session_state["results"] = run_experiment(config, ui_callback=ui_callback)
            st.success("Training Complete! View analysis in the Evaluate tab.")
        except Exception as e:
            st.error(f"Execution Error: {e}")

# --- TAB 2: EVALUATE ---
with tab_eval:
    st.markdown("### Recover Custom Symbols")
    c1, c2 = st.columns(2)
    with c1:
        custom_bits = st.text_input("Source A Bits (Binary):", placeholder="e.g. 10110010")
        st.caption("Must be an even length string (2 bits per QPSK symbol). If blank, uses random data.")
    with c2:
        weights_file = st.file_uploader("Upload .pt Weights (Optional)", type=["pt"])
        if weights_file:
            os.makedirs("weights", exist_ok=True)
            w_path = f"weights/temp_{model_name}.pt"
            with open(w_path, "wb") as f: f.write(weights_file.read())
            model_path = w_path
        else:
            model_path = f"pytorch_models/{model_name.lower()}_model.pt"

    if st.button("🔬 Run Inference", use_container_width=True):
        config = ExperimentConfig(
            mode="evaluate", model_name=model_name, model_path=model_path,
            alpha=alpha, snr_db=snr_db if noise_enabled else 100.0, noise_enabled=noise_enabled,
            custom_symbols=custom_bits.strip() if custom_bits else None
        )
        with st.spinner("Demodulating and Recovering Symbols..."):
            try:
                st.session_state["results"] = run_experiment(config)
            except Exception as e:
                st.error(f"Evaluation Error: {e}")

    # Results Rendering
    if st.session_state["results"]:
        res = st.session_state["results"]
        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Signal-to-Distortion Ratio (SDR)", f"{res.get('final_sdr_db', 0):.2f} dB")
        m2.metric("Symbol Recovery Accuracy", f"{res.get('final_symbol_accuracy', 0)*100:.1f} %")

        st.markdown("### Visual Recovery Analysis")
        img_c1, img_c2 = st.columns(2)
        sep_img = f"visualizations/{model_name}_separation.png"
        sym_img = f"visualizations/{model_name}_SourceA_symbol_recovery.png"
        
        if "sample_signal" in res:
            st.markdown("### Advanced Signal Visualization")

            fig3d = plot_3d_wave(res["sample_signal"], title="Recovered Signal (3D IQ)")
            st.plotly_chart(fig3d, use_container_width=True)

            fig_spec = plot_spectrogram(res["sample_signal"])
            st.pyplot(fig_spec)

        if os.path.exists(sym_img): img_c1.image(sym_img, caption="Constellation & Sequence Decisions")
        if os.path.exists(sep_img): img_c2.image(sep_img, caption="Waveform Separation")

# --- TAB 3: METRICS ---
with tab_logs:
    st.header("Model Leaderboard")
    log_files = glob.glob("logs/*_results.json")
    if log_files:
        data = []
        for f in log_files:
            with open(f, 'r') as j:
                d = json.load(j)
                data.append({
                    "Model": os.path.basename(f).replace("_results.json", ""),
                    "SDR (dB)": round(d.get("final_sdr_db", 0), 2),
                    "Sym Acc (%)": round(d.get("final_symbol_accuracy", 0)*100, 2),
                    "MSE": round(d.get("final_mse", 0), 6)
                })
        st.dataframe(pd.DataFrame(data).sort_values("SDR (dB)", ascending=False), use_container_width=True)
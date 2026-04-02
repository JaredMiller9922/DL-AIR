import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sys
import os
import glob
import json
import torch

# This ensures that even if run from different folders, 'utils' is findable
root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path:
    sys.path.append(root_path)

from config import ExperimentConfig
from main import run_experiment
# Import the specific functions from your plotting_utils
from utils.plot_utils.plotting_utils import (
    BeautifulRFPlotter, 
    plot_3d_wave, 
    plot_spectrogram, 
    animate_wave
)

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="RF Signal Separation Lab", layout="wide")
st.title("RF Signal Separation UI")

if "results" not in st.session_state:
    st.session_state["results"] = None






# -------------------------
# SIDEBAR CONTROLS
# -------------------------
st.sidebar.title("Experiment Settings")
mode = st.sidebar.selectbox("Mode", ["train", "inference"])
model_name = st.sidebar.selectbox("Model Architecture", ["Hybrid", "LSTM", "Linear", "IQ_CNN"])

with st.sidebar.expander("Hyperparameters", expanded=(mode == "train")):
    epochs = st.slider("Epochs", 1, 300, 50)
    batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64], value=16)
    lr = st.selectbox("Learning Rate", [1e-2, 1e-3, 5e-4, 1e-4], index=1)
    dropout = st.slider("Dropout", 0.0, 0.5, 0.1)

with st.sidebar.expander("Signal Parameters"):
    noise_alpha = st.slider("Interference (α)", 0.0, 2.0, 0.8)
    snr_db = st.slider("SNR (dB)", 0, 50, 25)
    gen_new = st.checkbox("Regenerate Dataset", value=False)






# -------------------------
# TABS
# -------------------------
tab_exec, tab_viz, tab_logs = st.tabs(["Execution", "Analysis", "Metrics"])

# =========================
# TAB 1: EXECUTION & LIVE TRAINING
# =========================
with tab_exec:
    col_ctrl, col_live = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("Pipeline Control")
        if st.button("🚀 Start Experiment", use_container_width=True):
            config = ExperimentConfig(
                mode=mode, 
                model_name=model_name, 
                batch_size=batch_size, 
                epochs=epochs,
                lr=lr, 
                dropout=dropout,
                noise_alpha=noise_alpha, 
                noise_sigma=snr_db, 
                generate_new_data=gen_new
            )

            # UI Live Update Elements
            progress_bar = st.progress(0)
            status = st.empty()
            chart_placeholder = st.empty()
            
            # Helper for chart updates
            history = {"Train MSE": [], "Val MSE": []}

            def ui_callback(epoch, t_loss, v_loss):
                progress_bar.progress(epoch / config.epochs)
                status.text(f"Epoch {epoch}/{config.epochs} | Train: {t_loss:.5f} | Val: {v_loss:.5f}")
                history["Train MSE"].append(t_loss)
                history["Val MSE"].append(v_loss)
                chart_placeholder.line_chart(pd.DataFrame(history))

            with st.spinner(f"Running {model_name}..."):
                try:
                    results = run_experiment(config, ui_callback=ui_callback if mode == "train" else None)
                    st.session_state["results"] = results
                    st.success("Target reached. Metrics updated.")
                except Exception as e:
                    st.error(f"Execution Error: {e}")

# =========================
# TAB 2: VISUALIZATIONS
# =========================
with tab_viz:
    if st.session_state["results"]:
        res = st.session_state["results"]
        
        # Metric Overview
        m1, m2, m3 = st.columns(3)
        m1.metric("Final MSE", f"{res.get('final_mse', 0):.6f}")
        m2.metric("SDR", f"{res.get('final_sdr_db', 0):.2f} dB")
        m3.metric("Status", "Complete" if mode == "train" else "Inference")

        st.divider()

        # Row 1: Interactive Plotly Visualization
        st.subheader("Interactive Signal Geometry")
        # We assume 'last_signal' is returned in the results dict from run_experiment
        if "sample_signal" in res:
            signal_to_plot = res["sample_signal"] # Expected shape (T,)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**3D Waveform Projection** (Spatial + Temporal)")
                fig_3d = plot_3d_wave(signal_to_plot, title=f"{model_name} Output")
                st.plotly_chart(fig_3d, use_container_width=True)
            
            with c2:
                st.markdown("**Frequency Domain Analysis** (Spectrogram)")
                fig_spec = plot_spectrogram(signal_to_plot)
                st.pyplot(fig_spec)
        else:
            st.warning("No signal data found in results. Ensure main.py returns 'sample_signal'.")

        # Row 2: Static PNG Outputs from BeautifulRFPlotter
        st.divider()
        st.subheader("Separation Performance")
        
        sep_img = f"./visualizations/{model_name}_separation.png"
        pipe_img = "./visualizations/data_pipeline_waves.png"
        
        col_img1, col_img2 = st.columns(2)
        if os.path.exists(sep_img):
            col_img1.image(sep_img, caption="Predicted vs Ground Truth")
        if os.path.exists(pipe_img):
            col_img2.image(pipe_img, caption="Input Mixture Analysis")
            
    else:
        st.info("Please execute the pipeline to generate visual data.")

# =========================
# TAB 3: METRICS 
# =========================
with tab_logs:
    st.header("Comparative Performance")
    log_files = glob.glob("./logs/*_results.json")
    if log_files:
        data = []
        for f in log_files:
            with open(f, 'r') as j:
                d = json.load(j)
                data.append({
                    "Model": os.path.basename(f).replace("_results.json", ""),
                    "MSE": d.get("final_mse"),
                    "SDR (dB)": d.get("final_sdr_db")
                })
        df = pd.DataFrame(data).sort_values("SDR (dB)", ascending=False)
        st.table(df)
    else:
        st.write("No historical logs found.")
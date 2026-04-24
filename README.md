# DL-AIR RF Separation Project

This repository is an RF source-separation and analysis workspace with three main layers:

1. synthetic-data generation and model training,
2. MIT RF Challenge backend evaluation and transfer analysis,
3. notebook- and dashboard-based comparison workflows.

The project now supports both the original QPSK-only workflow and a broader waveform-family generator with multichannel and MIT-aligned settings.

## What is in here

- `ui.py`
  Streamlit dashboard for training, evaluation, dataset generation, artifact review, and preset-driven experiments.
- `main.py`
  Shared experiment entry point used by the UI and CLI workflows.
- `config.py`
  Backward-compatible `ExperimentConfig` plus named benchmark presets.
- `utils/data_utils/`
  Synthetic generator, dataset wrappers, and smoke checks.
- `mit_challenge/`
  MIT RF Challenge integration, debug evaluation outputs, feature tables, and analysis notebook.
- `run_rf_benchmarks.py`
  Synthetic benchmark runner for model sweeps and classical feature baselines.
- `reporting_pipeline/`
  Existing reporting scripts, figures, and reference tables.

## Supported synthetic waveform families

The expanded generator keeps the original QPSK path intact and adds opt-in support for:

- `BPSK`
- `QPSK`
- `8PSK`
- `16QAM`
- `PAM` / `ASK`
- `MULTITONE` / OFDM-like interferers
- `COLORED_NOISE`
- `CHIRP`
- `BURSTY`
- `RECORDED` placeholder hooks

It also supports broader channel and mixing controls such as:

- variable `n_rx`,
- fixed or family-sampled symbol counts,
- fixed or family-sampled `samples_per_symbol`,
- rolloff ranges,
- phase-only or random-complex mixing,
- timing offset,
- carrier offset,
- phase mismatch,
- amplitude imbalance,
- alpha and SNR sweeps,
- MIT-aligned synthetic defaults.

## Benchmark presets

The project ships with named benchmark presets in `config.py`:

- `benchmark_qpsk_baseline`
- `benchmark_modulation_diverse`
- `benchmark_multichannel_phase_only`
- `benchmark_mit_aligned`
- `benchmark_interferer_diverse`

These are exposed in the dashboard and can also be used from code through `benchmark_config(...)`.

## Quick start

### 1. Launch the dashboard

```bash
streamlit run ui.py
```

Use the sidebar to pick a benchmark preset, tune waveform/mixing settings, and train or evaluate models.

### 2. Generate synthetic datasets

Use either:

- the **Dataset Builder** in the Streamlit app, or
- the existing generator scripts under `utils/data_utils/`.

The UI dataset builder now respects the expanded waveform-family settings and MIT-aligned mode.

### 3. Run synthetic benchmark sweeps

```bash
python run_rf_benchmarks.py --benchmark benchmark_modulation_diverse --models FastICA,Hybrid,LSTM,IQ_CNN,HTDemucs
```

Outputs are written under `reporting_pipeline/outputs/benchmarks/`.

### 4. Run MIT transfer experiments

Use the files under `mit_challenge/`:

- `infer_mit_separator.py`
- `run_mit_transfer_experiments.py`
- `sourceCode/evalLearnedDebugMain.m`

These keep the MIT backend evaluation chain intact while adding learned-separator support alongside FastICA.

### 5. Open the notebook

`mit_challenge/DataAnalysis.ipynb` is the main project notebook. It now covers:

- demod-side single-channel exploratory analysis,
- recovery-side MIT analysis,
- learned-separator MIT evaluation,
- synthetic benchmark summaries,
- multichannel unsupervised MIT mixture analysis,
- separator-comparison and prediction analysis.

## Important backward-compatibility note

Nothing in the expanded generator replaces the original QPSK workflow. If you stay on the baseline preset or leave waveform-family mode off, the older behavior remains the default path.

## Recommended workflow

1. Start with `benchmark_qpsk_baseline` to verify the original path.
2. Move to `benchmark_modulation_diverse` or `benchmark_interferer_diverse` for broader synthetic benchmarking.
3. Use `benchmark_mit_aligned` when preparing for MIT transfer studies.
4. Review outputs in the dashboard, saved CSV/JSON summaries, and `mit_challenge/DataAnalysis.ipynb`.

## Where outputs go

- `logs/`
  Model result JSON logs.
- `visualizations/`
  Separation plots and training snapshots.
- `reporting_pipeline/outputs/`
  Benchmark tables, figures, and reports.
- `mit_challenge/debugEval*`
  MIT backend scoring outputs.
- `mit_challenge/separation_frame_features*.csv`
  MIT frame-level feature tables.

## Documentation in the app

The dashboard includes a **Docs** tab that mirrors the project guide and explains how to use the expanded synthetic-generator controls.

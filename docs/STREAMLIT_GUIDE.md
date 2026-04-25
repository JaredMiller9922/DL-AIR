# Streamlit Dashboard Guide

The dashboard in `ui.py` is meant to be the quickest way to move between the original QPSK setup and the expanded benchmark family without rewriting scripts.

## Sidebar workflow

### 1. Benchmark preset

Start with a preset, then fine-tune if you need to:

- `benchmark_qpsk_baseline`
  Original fixed QPSK/QPSK regime.
- `benchmark_modulation_diverse`
  Broader digital-modulation families with random complex mixing.
- `benchmark_multichannel_phase_only`
  Four-receiver phase-structured benchmark.
- `benchmark_mit_aligned`
  Short-frame four-receiver synthetic mode designed to be closer to MIT evaluation conditions.
- `benchmark_interferer_diverse`
  QPSK SOI with more structured interferer types.

Use **Apply preset** to load those defaults into the controls.

### 2. Data Regime

This section controls the synthetic source families:

- **MIT-aligned synthetic mode**
  Forces MIT-style synthetic defaults when you run or build data.
- **Sample waveform families per example**
  Draws Source A / Source B types from the selected families instead of using one fixed pair.
- **Source A waveform / Source B waveform**
  Fixed waveform types for the classic path.
- **Source A family / Source B family**
  Family choices for diverse benchmarks.

### 3. Signal

These controls set waveform length and pulse-shaping assumptions:

- fixed symbol count,
- family-mode symbol range,
- fixed or family-sampled samples-per-symbol,
- fixed or family-sampled rolloff,
- RRC span,
- power normalization.

### 4. Channel + Mixing

These controls shape the separation problem itself:

- fixed or sampled `alpha`,
- mixing mode (`phase_only`, `random_complex`, `real`, `identity`),
- random receive phase,
- phase step and interference phase,
- timing offset,
- carrier offset,
- phase mismatch,
- amplitude imbalance.

### 5. Noise

You can work either in:

- **SNR mode**, or
- **fixed variance mode**.

For family benchmarks, SNR can also be sampled across a range.

### 6. Dataset Builder

This section saves synthetic train/val/test splits to disk. It now respects:

- waveform-family mode,
- MIT-aligned mode,
- expanded mixing and noise controls,
- the configured dataset path.

## Tabs

### Run Models

Train or evaluate selected models using the current sidebar configuration.

### Plots

Inspect signal-space plots, spectrograms, symbol recovery, and training snapshots from the latest run.

### Leaderboard

Compare current-session results against saved logs and reference tables.

### Artifact Gallery

Browse generated plot artifacts already saved in the repo.

### Docs

This tab shows:

- the dashboard guide,
- preset summary table,
- the repo-level README.

## Suggested usage patterns

### Original project path

1. Load `benchmark_qpsk_baseline`.
2. Keep waveform-family mode off.
3. Train or evaluate `Hybrid`, `IQ_CNN`, or `FastICA`.

### Broader synthetic benchmark path

1. Load `benchmark_modulation_diverse`.
2. Keep waveform-family mode on.
3. Run a model sweep or generate a saved dataset for controlled experiments.

### MIT-transfer preparation path

1. Load `benchmark_mit_aligned`.
2. Keep the four-receiver setup.
3. Use the generated synthetic data and compare later against MIT results in `mit_challenge/DataAnalysis.ipynb`.

## A practical note

The app keeps the older defaults alive on purpose. If you do nothing beyond the baseline preset, the original QPSK-centric workflow still works the way it did before.

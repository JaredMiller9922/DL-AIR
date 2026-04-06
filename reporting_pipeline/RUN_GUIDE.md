# Reporting Pipeline Run Guide

This guide explains the final-report evaluation flow in the simplest possible way.

## What gets evaluated

- `FastICA` baseline
- `Tiny`
- `Linear`
- `Hybrid`
- `LSTM`
- `IQ_CNN`

If a learned model checkpoint already exists, the pipeline loads it.

If a learned model checkpoint does **not** exist, the pipeline trains the model first, saves the checkpoint, and then runs the reporting sweeps.

That means the reporting run is designed to use **trained** learned models.

## What the run does

1. Creates or reuses a fixed training/validation dataset
2. Creates evaluation datasets for:
   - `alpha` sweep at fixed SNR
   - `SNR (dB)` sweep at fixed alpha
3. Loads or trains each learned model
4. Runs each model and `FastICA` on the exact same sweep datasets
5. Saves:
   - JSON summaries
   - CSV tables
   - PNG figures
   - QPSK alphabet artifacts

## Quick smoke test

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python reporting_pipeline/run_final_report_eval.py --quick
```

This uses fewer examples, fewer sweep points, and fewer active models.

## Full reporting run

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python reporting_pipeline/run_final_report_eval.py
```

## Force retraining

If you want all learned models retrained instead of reusing checkpoints:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python reporting_pipeline/run_final_report_eval.py --force-retrain
```

## Where outputs go

- `reporting_pipeline/outputs/json/`
- `reporting_pipeline/outputs/tables/`
- `reporting_pipeline/outputs/figures/`

## Most useful report files

- `reporting_pipeline/outputs/tables/model_comparison_reference.csv`
- `reporting_pipeline/outputs/tables/alpha_robustness.csv`
- `reporting_pipeline/outputs/tables/snr_robustness.csv`
- `reporting_pipeline/outputs/json/failure_thresholds.json`
- `reporting_pipeline/outputs/figures/qpsk_symbol_alphabet.png`
- `reporting_pipeline/outputs/figures/alpha_sweep_soi_symbol_accuracy.png`
- `reporting_pipeline/outputs/figures/noise_sweep_soi_symbol_accuracy.png`
- `reporting_pipeline/outputs/figures/alpha_sweep_wave_mse.png`
- `reporting_pipeline/outputs/figures/noise_sweep_wave_mse.png`

## Main idea for the report

The final-report pipeline is set up to answer this question clearly:

> How much waveform degradation can occur before symbol recovery begins to fail?

The sweeps over `alpha` and `SNR (dB)` are intended to make that visible for both the signal of interest and the interference signal.

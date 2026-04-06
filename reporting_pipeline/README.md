# Final Report Evaluation Plan

This folder is intentionally separate from the rest of the repo so the final reporting workflow can evolve without disrupting teammates.

## Goals

- Evaluate learned models and a FastICA baseline on the same exact synthetic RF-like data.
- Use trained models for reporting. Learned models load saved checkpoints when available and train automatically if a checkpoint is missing.
- Measure both wave recovery and symbol recovery.
- Run robustness sweeps over:
  - `alpha` (interference strength) at a fixed moderate SNR
  - `SNR in dB` at a fixed moderate `alpha`
- Save Overleaf-ready plots and tables.

## Metrics to report

- Wave-level:
  - `pit_mse`
  - `wave_mse`
  - `sdr_db`
- Symbol-level:
  - `soi_symbol_accuracy`
  - `int_symbol_accuracy`
  - `avg_symbol_accuracy`
  - `soi_symbol_error_rate`
  - `int_symbol_error_rate`

## QPSK symbol alphabet

- `0 -> (1 + 1j)/sqrt(2)`
- `1 -> (-1 + 1j)/sqrt(2)`
- `2 -> (-1 - 1j)/sqrt(2)`
- `3 -> (1 - 1j)/sqrt(2)`

## One-command run

From the repo root:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python reporting_pipeline/run_final_report_eval.py
```

Quick smoke test:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python reporting_pipeline/run_final_report_eval.py --quick
```

## Main outputs

- `reporting_pipeline/outputs/json/`
- `reporting_pipeline/outputs/tables/`
- `reporting_pipeline/outputs/figures/`

## Design notes

- FastICA is a baseline, not a trained model.
- Learned models are evaluated using trained checkpoints.
- The data generator uses the reporting mixture equation:
  - `s_mix = s_soi + alpha * s_int + noise`
- The same sweep datasets are used for every model and baseline.

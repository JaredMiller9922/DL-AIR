# Quick Reference

## Four big accomplishment zones

1. Model comparison and training core
   - `main.py`
   - `train.py`
   - `evaluate.py`
   - `networks/`

2. IQ-CNN and experiment ladder
   - `IQ_CNN.py`
   - `networks/iq_cnn_separator.py`
   - `experiments/simple_wave_separation/`
   - `experiments/structured_rf_bridge/`
   - `experiments/qpsk_rf_step/`

3. Cross-validation and tuning
   - `iqcnn_cv_search.py`
   - `iqcnn_deep_search.py`
   - `search_runs/`
   - `logs/lr_search_results.json`

4. Final reporting pipeline
   - `reporting_pipeline/`
   - `Final Report/`
   - `reports/overleaf_demo/`

## One-line story for each zone

- Training core: compare and evaluate separators fairly.
- Experiment ladder: prove the system can learn from easy waves up to RF-like QPSK packets.
- Cross-validation: search for better IQ-CNN settings and document robustness.
- Reporting pipeline: make final figures, tables, and an Overleaf-ready report skeleton.

## If asked ``where is FastICA?``

- Baseline code: `reporting_pipeline/baselines.py`

## If asked ``where are symbol metrics?``

- Symbol utilities: `utils/model_utils/symbol_utils.py`
- Final report metrics: `reporting_pipeline/metrics.py`
- Evaluation integration: `evaluate.py`

## If asked ``what data equation are we using?``

- Final reporting plan uses:
  - `s_mix = s_soi + alpha * s_int + noise`

## If asked ``what files matter most?``

- `main.py`
- `train.py`
- `evaluate.py`
- `networks/iq_cnn_separator.py`
- `iqcnn_cv_search.py`
- `iqcnn_deep_search.py`
- `reporting_pipeline/run_final_report_eval.py`
- `Final Report/main.tex`

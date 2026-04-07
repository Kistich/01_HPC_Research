#!/usr/bin/env bash
set -euo pipefail

cd "/Volumes/EXTERNAL_US/backup2/01_HPC_Research/Stage03_simulator/4_Simulator"
python3 run_trace_v2_experiments.py \
  --mode full \
  --trace-root "/Volumes/EXTERNAL_US/backup2/01_HPC_Research/Stage03_simulator/3_Trace_Generation/decisive_traces_v2_n1000" \
  --output-root "/Volumes/EXTERNAL_US/backup2/01_HPC_Research/Stage03_simulator/4_Simulator/experiment_packages/trace_v2_runs" \
  --run-id "20260407r1_expand_t235_n1000" \
  --families "T2,T3,T5" \
  --schedulers "SRTF,CPGJS" \
  --seed-filter "0,1" \
  --load-filter "0.5,0.7,0.9,1.1,1.3" \
  --cpu-nodes 4 \
  --gpu-nodes 2 \
  --cpgjs-variant engineering \
  --solver-timeout-seconds 0.0 \
  --solve-trigger-mode event \
  --event-refresh-interval-seconds 60.0 \
  --primary-lambda-energy 0.001 \
  --primary-gamma-balance 0.01 \
  --primary-phi-preemption 0.1 \
  --primary-phi-grid "0,0.05,0.1,0.2" \
  --runtime-estimation-mode estimated \
  --runtime-estimator-variant adaptive_history_v2 \
  --evaluation-regime learned_est \
  --runtime-error-distribution lognormal \
  --runtime-error-cpu-sigma 0.2 \
  --runtime-error-gpu-sigma 0.3 \
  --runtime-error-hybrid-sigma 0.4 \
  --runtime-error-seed 0 \
  --runtime-history-window 64 \
  --runtime-min-noise-scale 0.45 \
  --runtime-bias-correction-mix 0.35 \
  --runtime-cold-start-boost 0.45 \
  --runtime-refinement-strength 0.65 \
  --runtime-heteroskedasticity-strength 0.12 \
  --runtime-heavy-tail-mix-prob 0.03 \
  --runtime-heavy-tail-sigma-scale 1.1 \
  --priority-mode trace \
  --priority-weight-levels "0.7,1.0,1.3,1.6" \
  --priority-weight-probabilities "0.15,0.55,0.20,0.10" \
  --priority-seed 0 \
  --cpgjs-guard-policy dynamic \
  --cpgjs-guard-multi-node-threshold 0.2 \
  --cpgjs-guard-hybrid-threshold 0.1 \
  --cpgjs-v2-mode v2 \
  --disable-visualization

# Failure Cases

- total failure records: 74

## Counts by Benchmark

| Benchmark | Failure Records |
|---|---:|
| Decisive | 74 |

## Counts by Type

| Failure Type | Count |
|---|---:|
| p95_secondary_budget_exceeded | 17 |
| avg_jct_secondary_budget_exceeded | 15 |
| avg_wait_secondary_budget_exceeded | 12 |
| energy_secondary_budget_exceeded | 12 |
| primary_metric_regression | 10 |
| makespan_secondary_budget_exceeded | 8 |

## Decisive: Counts by Type

| Failure Type | Count |
|---|---:|
| p95_secondary_budget_exceeded | 17 |
| avg_jct_secondary_budget_exceeded | 15 |
| avg_wait_secondary_budget_exceeded | 12 |
| energy_secondary_budget_exceeded | 12 |
| primary_metric_regression | 10 |
| makespan_secondary_budget_exceeded | 8 |

## Decisive: Top Cases

| Type | Severity | Experiment | Seed | Trace | CPGJS | Baseline | Details |
|---|---:|---|---:|---|---:|---:|---|
| primary_metric_regression | 523.283858 | T5 | 0 | trace_T5_load070_seed0_n100 | 9529.507180 | 10052.791038 | CPGJS primary_score below best baseline |
| primary_metric_regression | 483.722233 | T5 | 0 | trace_T5_load110_seed0_n100 | 9000.110883 | 9483.833116 | CPGJS primary_score below best baseline |
| primary_metric_regression | 441.016696 | T3 | 1 | trace_T3_load070_seed1_n100 | 8480.788546 | 8921.805242 | CPGJS primary_score below best baseline |
| primary_metric_regression | 399.083266 | T3 | 0 | trace_T3_load070_seed0_n100 | 8415.105947 | 8814.189213 | CPGJS primary_score below best baseline |
| primary_metric_regression | 318.035897 | T3 | 0 | trace_T3_load050_seed0_n100 | 10350.574435 | 10668.610332 | CPGJS primary_score below best baseline |
| primary_metric_regression | 286.533756 | T2 | 0 | trace_T2_load050_seed0_n100 | 10482.592735 | 10769.126491 | CPGJS primary_score below best baseline |
| primary_metric_regression | 117.763514 | T5 | 0 | trace_T5_load050_seed0_n100 | 10690.763862 | 10808.527376 | CPGJS primary_score below best baseline |
| primary_metric_regression | 100.024332 | T5 | 1 | trace_T5_load110_seed1_n100 | 6809.139697 | 6909.164030 | CPGJS primary_score below best baseline |
| primary_metric_regression | 92.964583 | T5 | 0 | trace_T5_load090_seed0_n100 | 9777.427282 | 9870.391865 | CPGJS primary_score below best baseline |
| primary_metric_regression | 35.109173 | T3 | 0 | trace_T3_load090_seed0_n100 | 7735.924095 | 7771.033268 | CPGJS primary_score below best baseline |
| p95_secondary_budget_exceeded | 2.224452 | T2 | 1 | trace_T2_load050_seed1_n100 | 18.771063 | 5.732582 | CPGJS p95_jct regression exceeds allowed secondary budget (5.0%) |
| avg_wait_secondary_budget_exceeded | 1.202536 | T2 | 0 | trace_T2_load050_seed0_n100 | 1.085773 | 0.492965 | CPGJS avg_waiting_time regression exceeds allowed secondary budget (0.0%) |
| avg_wait_secondary_budget_exceeded | 0.767043 | T5 | 0 | trace_T5_load090_seed0_n100 | 0.187531 | 0.106127 | CPGJS avg_waiting_time regression exceeds allowed secondary budget (0.0%) |
| avg_wait_secondary_budget_exceeded | 0.761548 | T2 | 1 | trace_T2_load050_seed1_n100 | 0.869659 | 0.480064 | CPGJS avg_waiting_time regression exceeds allowed secondary budget (5.0%) |
| avg_wait_secondary_budget_exceeded | 0.549337 | T2 | 0 | trace_T2_load070_seed0_n100 | 2.234544 | 1.397169 | CPGJS avg_waiting_time regression exceeds allowed secondary budget (5.0%) |
| energy_secondary_budget_exceeded | 0.403364 | T3 | 1 | trace_T3_load130_seed1_n100 | 396.324952 | 272.694913 | CPGJS energy regression exceeds allowed secondary budget (5.0%) |
| makespan_secondary_budget_exceeded | 0.383859 | T3 | 1 | trace_T3_load130_seed1_n100 | 105.060719 | 71.769704 | CPGJS makespan regression exceeds dynamic budget (8.0%, primary_gain=12.986%) |
| p95_secondary_budget_exceeded | 0.297940 | T5 | 1 | trace_T5_load130_seed1_n100 | 29.214848 | 21.673704 | CPGJS p95_jct regression exceeds allowed secondary budget (5.0%) |
| avg_wait_secondary_budget_exceeded | 0.252572 | T5 | 1 | trace_T5_load130_seed1_n100 | 2.532866 | 1.944512 | CPGJS avg_waiting_time regression exceeds allowed secondary budget (5.0%) |
| avg_wait_secondary_budget_exceeded | 0.251372 | T5 | 1 | trace_T5_load110_seed1_n100 | 2.626089 | 2.098568 | CPGJS avg_waiting_time regression exceeds allowed secondary budget (0.0%) |


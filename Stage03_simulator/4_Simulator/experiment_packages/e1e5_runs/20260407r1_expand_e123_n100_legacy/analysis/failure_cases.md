# Failure Cases

- total failure records: 56

## Counts by Benchmark

| Benchmark | Failure Records |
|---|---:|
| Legacy | 56 |

## Counts by Type

| Failure Type | Count |
|---|---:|
| avg_jct_secondary_budget_exceeded | 11 |
| energy_secondary_budget_exceeded | 11 |
| primary_metric_regression | 11 |
| makespan_secondary_budget_exceeded | 9 |
| avg_wait_secondary_budget_exceeded | 8 |
| p95_secondary_budget_exceeded | 6 |

## Legacy: Counts by Type

| Failure Type | Count |
|---|---:|
| avg_jct_secondary_budget_exceeded | 11 |
| energy_secondary_budget_exceeded | 11 |
| primary_metric_regression | 11 |
| makespan_secondary_budget_exceeded | 9 |
| avg_wait_secondary_budget_exceeded | 8 |
| p95_secondary_budget_exceeded | 6 |

## Legacy: Top Cases

| Type | Severity | Experiment | Seed | Trace | CPGJS | Baseline | Details |
|---|---:|---|---:|---|---:|---:|---|
| avg_wait_secondary_budget_exceeded | 824.825000 | E1 | 1 | trace_baseline_gpu30_seed1_n100 | 0.099099 | 0.000120 | CPGJS avg_waiting_time regression exceeds allowed secondary budget (0.0%) |
| primary_metric_regression | 703.246621 | E1 | 1 | trace_baseline_gpu50_seed1_n100 | 10018.201898 | 10721.448518 | CPGJS primary_score below best baseline |
| primary_metric_regression | 635.038088 | E3 | 1 | trace_multi_node_gpu30_seed1_n100 | 8149.565525 | 8784.603613 | CPGJS primary_score below best baseline |
| primary_metric_regression | 264.911197 | E1 | 1 | trace_baseline_gpu30_seed1_n100 | 9679.344803 | 9944.256000 | CPGJS primary_score below best baseline |
| primary_metric_regression | 236.228589 | E1 | 0 | trace_baseline_gpu50_seed0_n100 | 11788.695427 | 12024.924015 | CPGJS primary_score below best baseline |
| primary_metric_regression | 203.447894 | E1 | 1 | trace_baseline_gpu20_seed1_n100 | 9102.924688 | 9306.372582 | CPGJS primary_score below best baseline |
| primary_metric_regression | 136.154514 | E1 | 0 | trace_baseline_gpu40_seed0_n100 | 11082.231993 | 11218.386508 | CPGJS primary_score below best baseline |
| primary_metric_regression | 89.956958 | E2 | 1 | trace_mixed_jobs_gpu30_seed1_n100 | 10642.267505 | 10732.224462 | CPGJS primary_score below best baseline |
| primary_metric_regression | 43.381335 | E1 | 0 | trace_baseline_gpu10_seed0_n100 | 9789.265756 | 9832.647092 | CPGJS primary_score below best baseline |
| primary_metric_regression | 26.741683 | E1 | 1 | trace_baseline_gpu40_seed1_n100 | 10295.834728 | 10322.576412 | CPGJS primary_score below best baseline |
| primary_metric_regression | 17.515845 | E1 | 0 | trace_baseline_gpu30_seed0_n100 | 11333.737952 | 11351.253797 | CPGJS primary_score below best baseline |
| avg_wait_secondary_budget_exceeded | 11.829040 | E1 | 0 | trace_baseline_gpu40_seed0_n100 | 0.005478 | 0.000427 | CPGJS avg_waiting_time regression exceeds allowed secondary budget (0.0%) |
| avg_wait_secondary_budget_exceeded | 10.466823 | E1 | 0 | trace_baseline_gpu50_seed0_n100 | 0.122179 | 0.010655 | CPGJS avg_waiting_time regression exceeds allowed secondary budget (0.0%) |
| avg_wait_secondary_budget_exceeded | 2.744000 | E1 | 1 | trace_baseline_gpu20_seed1_n100 | 0.015912 | 0.004250 | CPGJS avg_waiting_time regression exceeds allowed secondary budget (0.0%) |
| avg_wait_secondary_budget_exceeded | 1.656716 | E1 | 0 | trace_baseline_gpu10_seed0_n100 | 0.000534 | 0.000201 | CPGJS avg_waiting_time regression exceeds allowed secondary budget (0.0%) |
| primary_metric_regression | 1.264283 | E1 | 0 | trace_baseline_gpu20_seed0_n100 | 10280.987773 | 10282.252056 | CPGJS primary_score below best baseline |
| avg_wait_secondary_budget_exceeded | 0.589714 | E1 | 1 | trace_baseline_gpu50_seed1_n100 | 0.426452 | 0.268257 | CPGJS avg_waiting_time regression exceeds allowed secondary budget (0.0%) |
| energy_secondary_budget_exceeded | 0.196075 | E1 | 1 | trace_baseline_gpu20_seed1_n100 | 145.806869 | 121.904489 | CPGJS energy regression exceeds allowed secondary budget (0.0%) |
| makespan_secondary_budget_exceeded | 0.125991 | E1 | 1 | trace_baseline_gpu20_seed1_n100 | 40.365816 | 35.849150 | CPGJS makespan regression exceeds dynamic budget (0.0%, primary_gain=-2.186%) |
| makespan_secondary_budget_exceeded | 0.118036 | E1 | 0 | trace_baseline_gpu50_seed0_n100 | 40.413944 | 36.147277 | CPGJS makespan regression exceeds dynamic budget (0.0%, primary_gain=-1.964%) |


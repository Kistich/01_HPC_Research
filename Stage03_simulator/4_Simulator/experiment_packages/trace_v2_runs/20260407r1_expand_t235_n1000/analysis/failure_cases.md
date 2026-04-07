# Failure Cases

- total failure records: 54

## Counts by Benchmark

| Benchmark | Failure Records |
|---|---:|
| Decisive | 54 |

## Counts by Type

| Failure Type | Count |
|---|---:|
| avg_wait_secondary_budget_exceeded | 11 |
| p95_secondary_budget_exceeded | 11 |
| avg_jct_secondary_budget_exceeded | 10 |
| energy_secondary_budget_exceeded | 9 |
| primary_metric_regression | 9 |
| makespan_secondary_budget_exceeded | 4 |

## Decisive: Counts by Type

| Failure Type | Count |
|---|---:|
| avg_wait_secondary_budget_exceeded | 11 |
| p95_secondary_budget_exceeded | 11 |
| avg_jct_secondary_budget_exceeded | 10 |
| energy_secondary_budget_exceeded | 9 |
| primary_metric_regression | 9 |
| makespan_secondary_budget_exceeded | 4 |

## Decisive: Top Cases

| Type | Severity | Experiment | Seed | Trace | CPGJS | Baseline | Details |
|---|---:|---|---:|---|---:|---:|---|
| primary_metric_regression | 2274.169182 | T2 | 1 | trace_T2_load050_seed1_n1000 | 108425.769951 | 110699.939132 | CPGJS primary_score below best baseline |
| primary_metric_regression | 1817.250827 | T3 | 1 | trace_T3_load050_seed1_n1000 | 107258.094251 | 109075.345078 | CPGJS primary_score below best baseline |
| primary_metric_regression | 1709.680811 | T3 | 0 | trace_T3_load070_seed0_n1000 | 97793.930289 | 99503.611101 | CPGJS primary_score below best baseline |
| primary_metric_regression | 902.127399 | T3 | 0 | trace_T3_load090_seed0_n1000 | 86641.289153 | 87543.416552 | CPGJS primary_score below best baseline |
| primary_metric_regression | 682.298781 | T5 | 0 | trace_T5_load090_seed0_n1000 | 74077.191359 | 74759.490140 | CPGJS primary_score below best baseline |
| primary_metric_regression | 516.254051 | T2 | 0 | trace_T2_load070_seed0_n1000 | 95391.867342 | 95908.121394 | CPGJS primary_score below best baseline |
| primary_metric_regression | 481.724719 | T3 | 1 | trace_T3_load090_seed1_n1000 | 84816.774393 | 85298.499113 | CPGJS primary_score below best baseline |
| primary_metric_regression | 181.092533 | T3 | 1 | trace_T3_load070_seed1_n1000 | 96075.654121 | 96256.746654 | CPGJS primary_score below best baseline |
| primary_metric_regression | 154.622343 | T5 | 0 | trace_T5_load050_seed0_n1000 | 105384.566593 | 105539.188936 | CPGJS primary_score below best baseline |
| avg_wait_secondary_budget_exceeded | 1.328612 | T3 | 1 | trace_T3_load050_seed1_n1000 | 0.852251 | 0.365991 | CPGJS avg_waiting_time regression exceeds allowed secondary budget (0.0%) |
| avg_wait_secondary_budget_exceeded | 1.190195 | T2 | 0 | trace_T2_load070_seed0_n1000 | 1.818499 | 0.830291 | CPGJS avg_waiting_time regression exceeds allowed secondary budget (0.0%) |
| p95_secondary_budget_exceeded | 0.511044 | T3 | 1 | trace_T3_load090_seed1_n1000 | 27.251513 | 18.034892 | CPGJS p95_jct regression exceeds allowed secondary budget (0.0%) |
| p95_secondary_budget_exceeded | 0.430960 | T2 | 0 | trace_T2_load070_seed0_n1000 | 18.369934 | 12.837487 | CPGJS p95_jct regression exceeds allowed secondary budget (0.0%) |
| avg_wait_secondary_budget_exceeded | 0.303447 | T3 | 1 | trace_T3_load090_seed1_n1000 | 3.158779 | 2.423404 | CPGJS avg_waiting_time regression exceeds allowed secondary budget (0.0%) |
| avg_jct_secondary_budget_exceeded | 0.250595 | T2 | 0 | trace_T2_load070_seed0_n1000 | 4.058783 | 3.245481 | CPGJS avg_jct regression exceeds allowed secondary budget (0.0%) |
| avg_wait_secondary_budget_exceeded | 0.244871 | T3 | 0 | trace_T3_load070_seed0_n1000 | 0.870883 | 0.699577 | CPGJS avg_waiting_time regression exceeds allowed secondary budget (0.0%) |
| avg_jct_secondary_budget_exceeded | 0.223323 | T3 | 1 | trace_T3_load050_seed1_n1000 | 2.732737 | 2.233863 | CPGJS avg_jct regression exceeds allowed secondary budget (0.0%) |
| avg_wait_secondary_budget_exceeded | 0.206193 | T2 | 1 | trace_T2_load050_seed1_n1000 | 0.429400 | 0.355996 | CPGJS avg_waiting_time regression exceeds allowed secondary budget (0.0%) |
| avg_wait_secondary_budget_exceeded | 0.191814 | T5 | 0 | trace_T5_load090_seed0_n1000 | 4.523544 | 3.795513 | CPGJS avg_waiting_time regression exceeds allowed secondary budget (0.0%) |
| p95_secondary_budget_exceeded | 0.187376 | T3 | 1 | trace_T3_load050_seed1_n1000 | 10.841612 | 9.130731 | CPGJS p95_jct regression exceeds allowed secondary budget (0.0%) |


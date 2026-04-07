# Gate Summary

- total records: 60
- ok records: 60
- error records: 0

Benefit sign convention: positive means CPGJS performs better.

## Case-Level Pass Ratio by Benchmark

| Benchmark | Total Cases | Passed | Failed | Pass Ratio |
|---|---:|---:|---:|---:|
| Decisive | 30 | 14 | 16 | 46.67% |

## Case-Level Pass Ratio by Experiment

| Benchmark | Experiment | Total Cases | Passed | Failed | Pass Ratio |
|---|---|---:|---:|---:|---:|
| Decisive | T2 | 10 | 4 | 6 | 40.00% |
| Decisive | T3 | 10 | 5 | 5 | 50.00% |
| Decisive | T5 | 10 | 5 | 5 | 50.00% |

## Benchmark-Scoped Pair Summary

| Benchmark | Baseline | Metric | Pairs | Mean Rel Benefit(%) | Mean Benefit | 95% CI | p-value(t) | Effect Size dz | Wins/Losses/Ties |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Decisive | SRTF | avg_jct | 30 | 0.216 | 0.210255 | [-0.064282, 0.520119] | 0.17164 | 0.2559 | 14/16/0 |
| Decisive | SRTF | avg_waiting_time | 30 | -5.598 | 0.233220 | [-0.046638, 0.548253] | 0.14601 | 0.2727 | 17/13/0 |
| Decisive | SRTF | completion_rate | 30 | 0.000 | 0.000000 | [0.000000, 0.000000] | 1 | 0.0000 | 0/0/30 |
| Decisive | SRTF | makespan | 30 | -0.382 | -2.424669 | [-6.058412, 0.462213] | 0.149596 | -0.2703 | 7/14/9 |
| Decisive | SRTF | p95_jct | 30 | 1.004 | 2.217335 | [-0.087501, 4.695204] | 0.0778604 | 0.3337 | 16/14/0 |
| Decisive | SRTF | primary_score | 30 | 6.719 | 3702.011934 | [1945.923136, 5698.536312] | 0.00044781 | 0.7227 | 21/9/0 |
| Decisive | SRTF | total_energy_kwh | 30 | -1.440 | -55.715589 | [-62.620457, -49.254928] | 2.96773e-16 | -3.0060 | 0/30/0 |

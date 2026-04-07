# Gate Summary

- total records: 42
- ok records: 42
- error records: 0

Benefit sign convention: positive means CPGJS performs better.

## Case-Level Pass Ratio by Benchmark

| Benchmark | Total Cases | Passed | Failed | Pass Ratio |
|---|---:|---:|---:|---:|
| Legacy | 14 | 14 | 0 | 100.00% |

## Case-Level Pass Ratio by Experiment

| Benchmark | Experiment | Total Cases | Passed | Failed | Pass Ratio |
|---|---|---:|---:|---:|---:|
| Legacy | E1 | 10 | 10 | 0 | 100.00% |
| Legacy | E2 | 2 | 2 | 0 | 100.00% |
| Legacy | E3 | 2 | 2 | 0 | 100.00% |

## Benchmark-Scoped Pair Summary

| Benchmark | Baseline | Metric | Pairs | Mean Rel Benefit(%) | Mean Benefit | 95% CI | p-value(t) | Effect Size dz | Wins/Losses/Ties |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Legacy | SRTF | avg_jct | 14 | 3.861 | 0.751759 | [0.213139, 1.373117] | 0.0331407 | 0.6368 | 11/3/0 |
| Legacy | SRTF | avg_waiting_time | 14 | 6.726 | 0.973230 | [0.338933, 1.722577] | 0.0203289 | 0.7060 | 11/3/0 |
| Legacy | SRTF | completion_rate | 14 | 0.000 | 0.000000 | [0.000000, 0.000000] | 1 | 0.0000 | 0/0/14 |
| Legacy | SRTF | makespan | 14 | -0.211 | -0.320920 | [-2.404907, 1.952123] | 0.785535 | -0.0742 | 5/9/0 |
| Legacy | SRTF | p95_jct | 14 | 7.320 | 7.705067 | [3.826482, 11.892025] | 0.00359601 | 0.9472 | 13/1/0 |
| Legacy | SRTF | primary_score | 14 | 17.936 | 7014.305495 | [4914.030049, 9161.567354] | 2.65145e-05 | 1.6896 | 14/0/0 |
| Legacy | SRTF | total_energy_kwh | 14 | -0.347 | -3.564636 | [-6.952824, 0.059176] | 0.0870773 | -0.4946 | 4/10/0 |

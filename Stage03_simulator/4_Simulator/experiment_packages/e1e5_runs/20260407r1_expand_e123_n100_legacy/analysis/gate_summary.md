# Gate Summary

- total records: 42
- ok records: 42
- error records: 0

Benefit sign convention: positive means CPGJS performs better.

## Case-Level Pass Ratio by Benchmark

| Benchmark | Total Cases | Passed | Failed | Pass Ratio |
|---|---:|---:|---:|---:|
| Legacy | 14 | 2 | 12 | 14.29% |

## Case-Level Pass Ratio by Experiment

| Benchmark | Experiment | Total Cases | Passed | Failed | Pass Ratio |
|---|---|---:|---:|---:|---:|
| Legacy | E3 | 2 | 0 | 2 | 0.00% |
| Legacy | E1 | 10 | 1 | 9 | 10.00% |
| Legacy | E2 | 2 | 1 | 1 | 50.00% |

## Benchmark-Scoped Pair Summary

| Benchmark | Baseline | Metric | Pairs | Mean Rel Benefit(%) | Mean Benefit | 95% CI | p-value(t) | Effect Size dz | Wins/Losses/Ties |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Legacy | SRTF | avg_jct | 14 | -0.563 | 0.000791 | [-0.088730, 0.122362] | 0.989326 | 0.0036 | 2/11/1 |
| Legacy | SRTF | avg_waiting_time | 14 | -6080.095 | 0.024728 | [-0.057851, 0.153739] | 0.691215 | 0.1086 | 2/8/4 |
| Legacy | SRTF | completion_rate | 14 | 0.000 | 0.000000 | [0.000000, 0.000000] | 1 | 0.0000 | 0/0/14 |
| Legacy | SRTF | makespan | 14 | -2.772 | -1.044040 | [-1.954426, -0.289978] | 0.0463131 | -0.5886 | 2/9/3 |
| Legacy | SRTF | p95_jct | 14 | 0.303 | 0.055891 | [-0.261072, 0.401886] | 0.756003 | 0.0848 | 5/6/3 |
| Legacy | SRTF | primary_score | 14 | -0.375 | -48.628928 | [-280.713026, 241.898697] | 0.732385 | -0.0934 | 2/11/1 |
| Legacy | SRTF | total_energy_kwh | 14 | -2.767 | -3.564192 | [-7.280262, -0.693995] | 0.0696586 | -0.5283 | 0/13/1 |

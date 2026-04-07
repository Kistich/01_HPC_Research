# Gate Summary

- total records: 60
- ok records: 60
- error records: 0

Benefit sign convention: positive means CPGJS performs better.

## Case-Level Pass Ratio by Benchmark

| Benchmark | Total Cases | Passed | Failed | Pass Ratio |
|---|---:|---:|---:|---:|
| Decisive | 30 | 10 | 20 | 33.33% |

## Case-Level Pass Ratio by Experiment

| Benchmark | Experiment | Total Cases | Passed | Failed | Pass Ratio |
|---|---|---:|---:|---:|---:|
| Decisive | T5 | 10 | 1 | 9 | 10.00% |
| Decisive | T3 | 10 | 4 | 6 | 40.00% |
| Decisive | T2 | 10 | 5 | 5 | 50.00% |

## Benchmark-Scoped Pair Summary

| Benchmark | Baseline | Metric | Pairs | Mean Rel Benefit(%) | Mean Benefit | 95% CI | p-value(t) | Effect Size dz | Wins/Losses/Ties |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Decisive | SRTF | avg_jct | 30 | -2.698 | -0.061365 | [-0.243563, 0.165000] | 0.563306 | -0.1067 | 8/22/0 |
| Decisive | SRTF | avg_waiting_time | 30 | -8.981 | 0.075204 | [-0.133000, 0.341281] | 0.541145 | 0.1129 | 14/16/0 |
| Decisive | SRTF | completion_rate | 30 | 0.000 | 0.000000 | [0.000000, 0.000000] | 1 | 0.0000 | 0/0/30 |
| Decisive | SRTF | makespan | 30 | -2.330 | -2.070309 | [-4.847742, -0.175093] | 0.105346 | -0.3052 | 6/16/8 |
| Decisive | SRTF | p95_jct | 30 | -13.421 | -1.648623 | [-3.025053, -0.388847] | 0.0266641 | -0.4263 | 5/25/0 |
| Decisive | SRTF | primary_score | 30 | 7.724 | 434.365428 | [180.750631, 709.836605] | 0.00343106 | 0.5819 | 20/10/0 |
| Decisive | SRTF | total_energy_kwh | 30 | -3.120 | -11.857043 | [-21.303191, -6.092142] | 0.00704838 | -0.5294 | 2/28/0 |

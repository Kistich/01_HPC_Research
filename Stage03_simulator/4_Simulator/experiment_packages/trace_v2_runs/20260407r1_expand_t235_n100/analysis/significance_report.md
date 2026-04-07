# Significance Report

Benefit sign convention: positive means CPGJS performs better.

## Decisive: Trace-Level Paired Comparisons

| A | B | Metric | Pairs | Mean Benefit | Mean Rel Benefit(%) | 95% CI | p-value(t) | p-value(wilcoxon) | Effect Size dz | Wins/Losses/Ties |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| CPGJS | SRTF | avg_jct | 30 | -0.061365 | -2.698 | [-0.243563, 0.165000] | 0.563306 | 0.0803274 | -0.1067 | 8/22/0 |
| CPGJS | SRTF | avg_waiting_time | 30 | 0.075204 | -8.981 | [-0.133000, 0.341281] | 0.541145 | 0.556113 | 0.1129 | 14/16/0 |
| CPGJS | SRTF | completion_rate | 30 | 0.000000 | 0.000 | [0.000000, 0.000000] | 1 | 1 | 0.0000 | 0/0/30 |
| CPGJS | SRTF | makespan | 30 | -2.070309 | -2.330 | [-4.847742, -0.175093] | 0.105346 | 0.0362569 | -0.3052 | 6/16/8 |
| CPGJS | SRTF | p95_jct | 30 | -1.648623 | -13.421 | [-3.025053, -0.388847] | 0.0266641 | 0.00186401 | -0.4263 | 5/25/0 |
| CPGJS | SRTF | preemptions | 30 | 52.400000 | 15.297 | [27.163333, 78.567500] | 0.000524343 | 0.00203213 | 0.7121 | 22/8/0 |
| CPGJS | SRTF | primary_score | 30 | 434.365428 | 7.724 | [180.750631, 709.836605] | 0.00343106 | 0.00619499 | 0.5819 | 20/10/0 |
| CPGJS | SRTF | primary_score_phi_0 | 30 | -272.174845 | -2.772 | [-389.522424, -147.795299] | 0.000148009 | 0.000137394 | -0.7966 | 6/24/0 |
| CPGJS | SRTF | primary_score_phi_0p05 | 30 | 81.095291 | 1.366 | [-90.689079, 272.712117] | 0.386477 | 0.670181 | 0.1605 | 14/16/0 |
| CPGJS | SRTF | primary_score_phi_0p1 | 30 | 434.365428 | 7.724 | [180.750631, 709.836605] | 0.00343106 | 0.00619499 | 0.5819 | 20/10/0 |
| CPGJS | SRTF | primary_score_phi_0p2 | 30 | 1140.905701 | 316.155 | [704.417361, 1621.494681] | 3.73078e-05 | 1.59778e-05 | 0.8875 | 24/6/0 |
| CPGJS | SRTF | total_energy_kwh | 30 | -11.857043 | -3.120 | [-21.303191, -6.092142] | 0.00704838 | 4.65661e-08 | -0.5294 | 2/28/0 |


## Decisive: Seed-Level Significance Summary

| A | B | Metric | Seeds | Significant Seeds (p<0.05) | Ratio | Mean Rel Benefit(%) | Mean Effect Size dz |
|---|---|---|---:|---:|---:|---:|---:|
| CPGJS | SRTF | avg_jct | 2 | 0 | 0.000 | -2.698 | -0.0825 |
| CPGJS | SRTF | avg_waiting_time | 2 | 0 | 0.000 | -8.981 | 0.1124 |
| CPGJS | SRTF | completion_rate | 2 | 0 | 0.000 | 0.000 | 0.0000 |
| CPGJS | SRTF | makespan | 2 | 0 | 0.000 | -2.330 | -0.3735 |
| CPGJS | SRTF | p95_jct | 2 | 0 | 0.000 | -13.421 | -0.4275 |
| CPGJS | SRTF | preemptions | 2 | 2 | 1.000 | 15.297 | 0.7052 |
| CPGJS | SRTF | primary_score | 2 | 1 | 0.500 | 7.724 | 0.6562 |
| CPGJS | SRTF | primary_score_phi_0 | 2 | 2 | 1.000 | -2.772 | -0.7904 |
| CPGJS | SRTF | primary_score_phi_0p05 | 2 | 0 | 0.000 | 1.366 | 0.1657 |
| CPGJS | SRTF | primary_score_phi_0p1 | 2 | 1 | 0.500 | 7.724 | 0.6562 |
| CPGJS | SRTF | primary_score_phi_0p2 | 2 | 2 | 1.000 | 316.155 | 1.0363 |
| CPGJS | SRTF | total_energy_kwh | 2 | 1 | 0.500 | -3.120 | -0.8761 |


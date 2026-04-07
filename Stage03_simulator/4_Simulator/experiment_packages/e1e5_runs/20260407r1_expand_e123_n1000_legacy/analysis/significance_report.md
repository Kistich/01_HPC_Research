# Significance Report

Benefit sign convention: positive means CPGJS performs better.

## Legacy: Trace-Level Paired Comparisons

| A | B | Metric | Pairs | Mean Benefit | Mean Rel Benefit(%) | 95% CI | p-value(t) | p-value(wilcoxon) | Effect Size dz | Wins/Losses/Ties |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| CPGJS | SRTF | avg_jct | 14 | 0.751759 | 3.861 | [0.213139, 1.373117] | 0.0331407 | 0.00402832 | 0.6368 | 11/3/0 |
| CPGJS | SRTF | avg_waiting_time | 14 | 0.973230 | 6.726 | [0.338933, 1.722577] | 0.0203289 | 0.00402832 | 0.7060 | 11/3/0 |
| CPGJS | SRTF | completion_rate | 14 | 0.000000 | 0.000 | [0.000000, 0.000000] | 1 | 1 | 0.0000 | 0/0/14 |
| CPGJS | SRTF | makespan | 14 | -0.320920 | -0.211 | [-2.404907, 1.952123] | 0.785535 | 0.760864 | -0.0742 | 5/9/0 |
| CPGJS | SRTF | p95_jct | 14 | 7.705067 | 7.320 | [3.826482, 11.892025] | 0.00359601 | 0.00170898 | 0.9472 | 13/1/0 |
| CPGJS | SRTF | preemptions | 14 | 4693.500000 | 78.931 | [4504.646429, 4884.867857] | 8.181e-16 | 0.00012207 | 12.3653 | 14/0/0 |
| CPGJS | SRTF | primary_score | 14 | 7014.305495 | 17.936 | [4914.030049, 9161.567354] | 2.65145e-05 | 0.00012207 | 1.6896 | 14/0/0 |
| CPGJS | SRTF | primary_score_phi_0 | 14 | -2245.064599 | -3.770 | [-3540.728491, -836.390718] | 0.00830076 | 0.0166016 | -0.8309 | 3/11/0 |
| CPGJS | SRTF | primary_score_phi_0p05 | 14 | 2384.620448 | 5.594 | [723.694878, 4093.888484] | 0.017788 | 0.0134277 | 0.7248 | 11/3/0 |
| CPGJS | SRTF | primary_score_phi_0p1 | 14 | 7014.305495 | 17.936 | [4914.030049, 9161.567354] | 2.65145e-05 | 0.00012207 | 1.6896 | 14/0/0 |
| CPGJS | SRTF | primary_score_phi_0p2 | 14 | 16273.675589 | 63.563 | [13108.339705, 19486.160441] | 2.30912e-07 | 0.00012207 | 2.6156 | 14/0/0 |
| CPGJS | SRTF | total_energy_kwh | 14 | -3.564636 | -0.347 | [-6.952824, 0.059176] | 0.0870773 | 0.104004 | -0.4946 | 4/10/0 |


## Legacy: Seed-Level Significance Summary

| A | B | Metric | Seeds | Significant Seeds (p<0.05) | Ratio | Mean Rel Benefit(%) | Mean Effect Size dz |
|---|---|---|---:|---:|---:|---:|---:|
| CPGJS | SRTF | avg_jct | 2 | 0 | 0.000 | 3.861 | 0.6406 |
| CPGJS | SRTF | avg_waiting_time | 2 | 1 | 0.500 | 6.726 | 0.7228 |
| CPGJS | SRTF | completion_rate | 2 | 0 | 0.000 | 0.000 | 0.0000 |
| CPGJS | SRTF | makespan | 2 | 0 | 0.000 | -0.211 | -0.0567 |
| CPGJS | SRTF | p95_jct | 2 | 0 | 0.000 | 7.320 | 0.9101 |
| CPGJS | SRTF | preemptions | 2 | 2 | 1.000 | 78.931 | 12.7889 |
| CPGJS | SRTF | primary_score | 2 | 2 | 1.000 | 17.936 | 1.6407 |
| CPGJS | SRTF | primary_score_phi_0 | 2 | 1 | 0.500 | -3.770 | -0.8154 |
| CPGJS | SRTF | primary_score_phi_0p05 | 2 | 0 | 0.000 | 5.594 | 0.7064 |
| CPGJS | SRTF | primary_score_phi_0p1 | 2 | 2 | 1.000 | 17.936 | 1.6407 |
| CPGJS | SRTF | primary_score_phi_0p2 | 2 | 2 | 1.000 | 63.563 | 2.5257 |
| CPGJS | SRTF | total_energy_kwh | 2 | 1 | 0.500 | -0.347 | -0.6018 |


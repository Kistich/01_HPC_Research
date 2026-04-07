# Significance Report

Benefit sign convention: positive means CPGJS performs better.

## Legacy: Trace-Level Paired Comparisons

| A | B | Metric | Pairs | Mean Benefit | Mean Rel Benefit(%) | 95% CI | p-value(t) | p-value(wilcoxon) | Effect Size dz | Wins/Losses/Ties |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| CPGJS | SRTF | avg_jct | 14 | 0.000791 | -0.563 | [-0.088730, 0.122362] | 0.989326 | 0.0747355 | 0.0036 | 2/11/1 |
| CPGJS | SRTF | avg_waiting_time | 14 | 0.024728 | -6080.095 | [-0.057851, 0.153739] | 0.691215 | 0.139414 | 0.1086 | 2/8/4 |
| CPGJS | SRTF | completion_rate | 14 | 0.000000 | 0.000 | [0.000000, 0.000000] | 1 | 1 | 0.0000 | 0/0/14 |
| CPGJS | SRTF | makespan | 14 | -1.044040 | -2.772 | [-1.954426, -0.289978] | 0.0463131 | 0.0328542 | -0.5886 | 2/9/3 |
| CPGJS | SRTF | p95_jct | 14 | 0.055891 | 0.303 | [-0.261072, 0.401886] | 0.756003 | 0.789574 | 0.0848 | 5/6/3 |
| CPGJS | SRTF | preemptions | 14 | 17.071429 | -31.805 | [-14.142857, 61.644643] | 0.442409 | 0.504739 | 0.2117 | 8/4/2 |
| CPGJS | SRTF | primary_score | 14 | -48.628928 | -0.375 | [-280.713026, 241.898697] | 0.732385 | 0.0640301 | -0.0934 | 2/11/1 |
| CPGJS | SRTF | primary_score_phi_0 | 14 | -110.191397 | -1.046 | [-293.675510, 73.805689] | 0.281476 | 0.0546239 | -0.3003 | 2/11/1 |
| CPGJS | SRTF | primary_score_phi_0p05 | 14 | -79.410163 | -0.728 | [-285.348771, 155.591127] | 0.510701 | 0.0640301 | -0.1808 | 2/11/1 |
| CPGJS | SRTF | primary_score_phi_0p1 | 14 | -48.628928 | -0.375 | [-280.713026, 241.898697] | 0.732385 | 0.0640301 | -0.0934 | 2/11/1 |
| CPGJS | SRTF | primary_score_phi_0p2 | 14 | 12.933541 | 0.465 | [-270.376115, 404.705612] | 0.945765 | 0.0747355 | 0.0185 | 2/11/1 |
| CPGJS | SRTF | total_energy_kwh | 14 | -3.564192 | -2.767 | [-7.280262, -0.693995] | 0.0696586 | 0.00147378 | -0.5283 | 0/13/1 |


## Legacy: Seed-Level Significance Summary

| A | B | Metric | Seeds | Significant Seeds (p<0.05) | Ratio | Mean Rel Benefit(%) | Mean Effect Size dz |
|---|---|---|---:|---:|---:|---:|---:|
| CPGJS | SRTF | avg_jct | 2 | 1 | 0.500 | -0.563 | -0.3303 |
| CPGJS | SRTF | avg_waiting_time | 2 | 0 | 0.000 | -6080.095 | -0.2290 |
| CPGJS | SRTF | completion_rate | 2 | 0 | 0.000 | 0.000 | 0.0000 |
| CPGJS | SRTF | makespan | 2 | 0 | 0.000 | -2.772 | -0.5661 |
| CPGJS | SRTF | p95_jct | 2 | 0 | 0.000 | 0.303 | 0.0211 |
| CPGJS | SRTF | preemptions | 2 | 0 | 0.000 | -31.805 | -0.0099 |
| CPGJS | SRTF | primary_score | 2 | 1 | 0.500 | -0.375 | -0.3381 |
| CPGJS | SRTF | primary_score_phi_0 | 2 | 0 | 0.000 | -1.046 | -0.3751 |
| CPGJS | SRTF | primary_score_phi_0p05 | 2 | 1 | 0.500 | -0.728 | -0.3510 |
| CPGJS | SRTF | primary_score_phi_0p1 | 2 | 1 | 0.500 | -0.375 | -0.3381 |
| CPGJS | SRTF | primary_score_phi_0p2 | 2 | 1 | 0.500 | 0.465 | -0.3132 |
| CPGJS | SRTF | total_energy_kwh | 2 | 0 | 0.000 | -2.767 | -0.5906 |


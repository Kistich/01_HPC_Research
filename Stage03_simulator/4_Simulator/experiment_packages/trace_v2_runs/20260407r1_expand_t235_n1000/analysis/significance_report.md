# Significance Report

Benefit sign convention: positive means CPGJS performs better.

## Decisive: Trace-Level Paired Comparisons

| A | B | Metric | Pairs | Mean Benefit | Mean Rel Benefit(%) | 95% CI | p-value(t) | p-value(wilcoxon) | Effect Size dz | Wins/Losses/Ties |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| CPGJS | SRTF | avg_jct | 30 | 0.210255 | 0.216 | [-0.064282, 0.520119] | 0.17164 | 0.502761 | 0.2559 | 14/16/0 |
| CPGJS | SRTF | avg_waiting_time | 30 | 0.233220 | -5.598 | [-0.046638, 0.548253] | 0.14601 | 0.32847 | 0.2727 | 17/13/0 |
| CPGJS | SRTF | completion_rate | 30 | 0.000000 | 0.000 | [0.000000, 0.000000] | 1 | 1 | 0.0000 | 0/0/30 |
| CPGJS | SRTF | makespan | 30 | -2.424669 | -0.382 | [-6.058412, 0.462213] | 0.149596 | 0.139622 | -0.2703 | 7/14/9 |
| CPGJS | SRTF | p95_jct | 30 | 2.217335 | 1.004 | [-0.087501, 4.695204] | 0.0778604 | 0.253436 | 0.3337 | 16/14/0 |
| CPGJS | SRTF | preemptions | 30 | 574.733333 | 18.388 | [391.800000, 778.947500] | 2.72492e-06 | 2.6077e-08 | 1.0598 | 28/2/0 |
| CPGJS | SRTF | primary_score | 30 | 3702.011934 | 6.719 | [1945.923136, 5698.536312] | 0.00044781 | 0.0013406 | 0.7227 | 21/9/0 |
| CPGJS | SRTF | primary_score_phi_0 | 30 | -3392.271312 | -3.612 | [-4015.101428, -2753.339404] | 2.32431e-11 | 1.86265e-09 | -1.9108 | 0/30/0 |
| CPGJS | SRTF | primary_score_phi_0p05 | 30 | 154.870311 | 0.551 | [-740.967300, 1179.536612] | 0.748292 | 0.983834 | 0.0591 | 14/16/0 |
| CPGJS | SRTF | primary_score_phi_0p1 | 30 | 3702.011934 | 6.719 | [1945.923136, 5698.536312] | 0.00044781 | 0.0013406 | 0.7227 | 21/9/0 |
| CPGJS | SRTF | primary_score_phi_0p2 | 30 | 10796.295180 | 67.416 | [7143.386397, 14850.445051] | 4.97815e-06 | 1.63913e-07 | 1.0200 | 27/3/0 |
| CPGJS | SRTF | total_energy_kwh | 30 | -55.715589 | -1.440 | [-62.620457, -49.254928] | 2.96773e-16 | 1.86265e-09 | -3.0060 | 0/30/0 |


## Decisive: Seed-Level Significance Summary

| A | B | Metric | Seeds | Significant Seeds (p<0.05) | Ratio | Mean Rel Benefit(%) | Mean Effect Size dz |
|---|---|---|---:|---:|---:|---:|---:|
| CPGJS | SRTF | avg_jct | 2 | 0 | 0.000 | 0.216 | 0.2391 |
| CPGJS | SRTF | avg_waiting_time | 2 | 0 | 0.000 | -5.598 | 0.2503 |
| CPGJS | SRTF | completion_rate | 2 | 0 | 0.000 | 0.000 | 0.0000 |
| CPGJS | SRTF | makespan | 2 | 0 | 0.000 | -0.382 | -0.2857 |
| CPGJS | SRTF | p95_jct | 2 | 0 | 0.000 | 1.004 | 0.3252 |
| CPGJS | SRTF | preemptions | 2 | 2 | 1.000 | 18.388 | 1.0480 |
| CPGJS | SRTF | primary_score | 2 | 2 | 1.000 | 6.719 | 0.7372 |
| CPGJS | SRTF | primary_score_phi_0 | 2 | 2 | 1.000 | -3.612 | -1.9483 |
| CPGJS | SRTF | primary_score_phi_0p05 | 2 | 0 | 0.000 | 0.551 | 0.0031 |
| CPGJS | SRTF | primary_score_phi_0p1 | 2 | 2 | 1.000 | 6.719 | 0.7372 |
| CPGJS | SRTF | primary_score_phi_0p2 | 2 | 2 | 1.000 | 67.416 | 1.0425 |
| CPGJS | SRTF | total_energy_kwh | 2 | 2 | 1.000 | -1.440 | -3.0365 |


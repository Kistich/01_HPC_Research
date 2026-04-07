# Failure Slice Report

- total failures: 74
- top-5 mechanism coverage: 100.00%

## By Mechanism

| Mechanism | Count | Ratio |
|---|---:|---:|
| TOPOLOGY_COUPLING | 56 | 75.68% |
| OBJECTIVE_MISMATCH | 10 | 13.51% |
| FRAGMENTATION_GANG | 8 | 10.81% |

## By Benchmark x Mechanism

| Benchmark | Mechanism | Count |
|---|---|---:|
| Decisive | TOPOLOGY_COUPLING | 56 |
| Decisive | OBJECTIVE_MISMATCH | 10 |
| Decisive | FRAGMENTATION_GANG | 8 |

## By Experiment x Mechanism

| Experiment | Mechanism | Count |
|---|---|---:|
| T2 | TOPOLOGY_COUPLING | 13 |
| T2 | FRAGMENTATION_GANG | 2 |
| T2 | OBJECTIVE_MISMATCH | 1 |
| T3 | TOPOLOGY_COUPLING | 17 |
| T3 | OBJECTIVE_MISMATCH | 4 |
| T3 | FRAGMENTATION_GANG | 3 |
| T5 | TOPOLOGY_COUPLING | 26 |
| T5 | OBJECTIVE_MISMATCH | 5 |
| T5 | FRAGMENTATION_GANG | 3 |

## By Job Class

| Major Job Class | Count |
|---|---:|
| HYBRID | 63 |
| GPU_SINGLE | 8 |
| CPU_MULTI | 3 |

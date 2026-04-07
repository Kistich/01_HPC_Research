# Failure Slice Report

- total failures: 56
- top-5 mechanism coverage: 100.00%

## By Mechanism

| Mechanism | Count | Ratio |
|---|---:|---:|
| TAIL_PRESSURE | 39 | 69.64% |
| OBJECTIVE_MISMATCH | 11 | 19.64% |
| TOPOLOGY_COUPLING | 5 | 8.93% |
| FRAGMENTATION_GANG | 1 | 1.79% |

## By Benchmark x Mechanism

| Benchmark | Mechanism | Count |
|---|---|---:|
| Legacy | TAIL_PRESSURE | 39 |
| Legacy | OBJECTIVE_MISMATCH | 11 |
| Legacy | TOPOLOGY_COUPLING | 5 |
| Legacy | FRAGMENTATION_GANG | 1 |

## By Experiment x Mechanism

| Experiment | Mechanism | Count |
|---|---|---:|
| E1 | TAIL_PRESSURE | 35 |
| E1 | OBJECTIVE_MISMATCH | 9 |
| E2 | TOPOLOGY_COUPLING | 5 |
| E2 | OBJECTIVE_MISMATCH | 1 |
| E3 | TAIL_PRESSURE | 4 |
| E3 | FRAGMENTATION_GANG | 1 |
| E3 | OBJECTIVE_MISMATCH | 1 |

## By Job Class

| Major Job Class | Count |
|---|---:|
| CPU_SINGLE | 45 |
| GPU_SINGLE | 11 |

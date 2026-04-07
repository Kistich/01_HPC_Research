# PT11 Trace-Factor Expanded Matrix Summary (2026-04-07)

## Scope
- Added decisive holdout families: T2/T3/T5 (n=100 and n=1000).
- Added legacy D/E variation set: E1/E2/E3 (n=100 and n=1000).
- Recomputed phase-1/phase-2/negative-control/robustness/holdout tables on merged case set.

## Data Volume
- Previous case rows: 90
- New rows added: 88
- Merged rows: 178

## Interaction CI Tightening (Primary)
- Decisive A x E: CI width 21.04 -> 14.46 (31.30% narrower)
- Decisive A x D: CI width 16.95 -> 16.12 (4.88% narrower)
- Decisive Ctail x D: CI width 19.14 -> 16.83 (12.10% narrower)
- Decisive B x D: CI width 17.11 -> 15.11 (11.71% narrower)
- Legacy A x D: CI width 49.29 -> 33.96 (31.09% narrower)
- Legacy Ctail x D: CI width 55.63 -> 37.40 (32.77% narrower)

## Current Interaction Status
- Decisive D/E interactions still cross zero in 95% CI (no strong interaction-sign significance yet).
- Legacy D/E interactions also remain mostly crossing zero despite narrower CI.

## Holdout Generalization
- Decisive family holdout:
	- T2: n=20, MAE=12.81, sign_acc=0.85
	- T3: n=20, MAE=8.01, sign_acc=0.55
	- T5: n=20, MAE=6.50, sign_acc=0.60
- Legacy gpu_ratio holdout:
	- sign_acc in [0.58, 0.83], MAE range ~6.36 to ~22.50

## Decision
- Primary claim (main effects + cross-scale directional stability): evidence is substantially improved and mostly sufficient for mainline narrative.
- Secondary claim (interaction-strength and gate robustness): evidence is improved but not yet at strong interaction-proof level.
- Recommendation: do not resume broad trace-specific parameter search; prioritize additional sample-size expansion (more seeds / controlled replications) for D/E interaction tightening. If a stronger secondary claim is required, only then do targeted minimal parameter probes on identified weak regimes.


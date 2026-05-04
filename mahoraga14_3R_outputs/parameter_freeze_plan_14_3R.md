# Mahoraga14_3R Parameter Freeze Plan

## Institutional freeze rule
- Freeze the 14.3 architecture exactly as implemented.
- Reduce acceptance review to four interpretable scale knobs.
- Do not retune structural engine / ML / continuation model families inside 14.3R.

## Primary knobs
- `budget_multiplier`: scales effective long budget and cash redeployment around the frozen 14.3 schedule.
- `conviction_multiplier`: scales conviction translation already present in 14.3.
- `leader_multiplier`: scales leader participation already present in 14.3.
- `backoff_strength`: scales how aggressively 14.3 backs off under fragility/break-risk.

## Frozen parameters
- Engine mixture, structural policy, continuation model family and all short logic remain frozen.
- Continuation is reviewed only as a quality filter; it is not promoted to a new tuning axis in 14.3R.

## Selection discipline
- Acceptance prefers a plateau candidate over a point-optimal candidate.
- If the current 14.3 point is not on a stable plateau, 14.1 remains the institutional baseline.
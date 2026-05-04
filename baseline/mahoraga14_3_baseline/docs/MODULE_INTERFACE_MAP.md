# Module Interface Map

## BASE_ALPHA_V2
- inputs: universe OHLCV, fold schedule, shared precomputed state
- outputs: base long-book weights and stitched baseline traces

## PARTICIPATION_ALLOCATOR_V2
- inputs: weekly continuation, breadth, volatility, benchmark and structural context
- outputs: long budget, gate scale, vol multiplier, exp cap, leader blend, conviction multiplier

## CONVICTION_AMPLIFIER_LAYER
- inputs: allocator raw state plus healthy-regime context
- outputs: amplified budget/gate/vol/exp-cap translations

## LEADER_PARTICIPATION_LAYER
- inputs: base long book, allocator state, leader opportunity state
- outputs: conditional leader blend and redeployed cash-drag participation

## RISK_BACKOFF_LAYER_V2
- inputs: fragility, break-risk, benchmark weakness, continuation quality
- outputs: clipped budget / conviction / leader participation under stress

## continuation as quality filter
- inputs: continuation trigger / pressure / break-risk models
- outputs: quality signal used to allow or restrain participation; not a separate thesis in the official baseline
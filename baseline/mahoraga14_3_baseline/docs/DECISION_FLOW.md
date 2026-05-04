# Decision Flow

1. `BASE_ALPHA_V2` proposes the long-only book.
2. `PARTICIPATION_ALLOCATOR_V2` reads market/book state and sets participation budget and caps.
3. `CONVICTION_AMPLIFIER_LAYER` increases translation strength in healthy bull regimes.
4. `LEADER_PARTICIPATION_LAYER` conditionally increases leader participation and reduces cash drag.
5. `RISK_BACKOFF_LAYER_V2` clips the system when the regime deteriorates.
6. continuation remains a quality filter that modulates participation but does not define the thesis.
7. the promoted official freeze applies the accepted robust multipliers `B1.05_C1.10_L1.10_R1.05` over the frozen 14.3R architecture.
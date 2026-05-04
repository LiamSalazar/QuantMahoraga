# Component Audit

## BASE_ALPHA_V2
- does: generates the long-only stock-selection core inherited from the frozen 14.1 baseline.
- does not: decide final bull participation on its own.

## PARTICIPATION_ALLOCATOR_V2
- does: converts healthy regime evidence into budget/cap participation controls.
- does not: invent new alpha or override the book unconditionally.

## CONVICTION_AMPLIFIER_LAYER
- does: scale the translation from conviction to effective participation.
- does not: bypass risk backoff or create a new model family.

## LEADER_PARTICIPATION_LAYER
- does: conditionally lift exposure toward leaders and reduce cash drag.
- does not: add shorts, hedges or permanent tech concentration.

## RISK_BACKOFF_LAYER_V2
- does: harden the system when fragility, break-risk or benchmark weakness rise.
- does not: pursue upside participation.

## continuation
- does: act as a quality filter with positive local edge.
- does not: bear sole responsibility for bull participation in the official baseline.
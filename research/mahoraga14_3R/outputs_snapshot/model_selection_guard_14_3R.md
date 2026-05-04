# Model Selection Guard

- Multiple Mahoraga 14 variants were explored before 14.3; acceptance therefore uses a conservative guard.
- 14.3R only examines a local neighborhood around the frozen 14.3 point, not a new discovery grid.
- White/Reality-Check style family bootstrap p-value (local family max excess vs control): 0.0000
- Base-current candidate bootstrap p-value vs centered null: 0.1700
- Highest robust local candidate under the acceptance family: `B1.05_C1.10_L1.10_R1.05`.
- Decision discipline: a candidate is not promoted to institutional baseline solely because it is the best point in the local family.
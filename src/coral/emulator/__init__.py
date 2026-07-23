"""coral.emulator — the ML flood-depth surrogate (dissertation Ch. 6, Layer 1).

Learns (static maps + scalar forcing) -> 2-D max-depth field, replacing the
expensive GeoClaw+LISFLOOD sweep for adaptation-scenario exploration.

Stage 1 (here): U-Net CNN baseline (dataset.py, models.py, train.py).
Stage 2 (next): GNN in PyTorch Geometric (gnn.py) — mSWE-GNN / FloodGNN-GRU.

See docs/EMULATOR_PLAN.md for the staged plan and the wiki note
"CORAL emulator method candidates" for the architecture rationale.
"""

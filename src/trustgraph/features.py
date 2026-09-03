"""Canonical feature names and column order.

These names come from PROJECT_SPEC.md section 5 and are used verbatim everywhere:
graph construction, the model, and (later) explanation attribution. Fixing the order
in one place means a mismatch between producer and consumer is a shape error rather
than a silently wrong attribution (DECISIONS.md D15).
"""

from __future__ import annotations

# --- Fog node (RSU) features, PROJECT_SPEC.md 5.1 ---
RSU_FEATURES: tuple[str, ...] = (
    "load",              # advertised compute utilization (L8 - may be false)
    "queue_depth",       # advertised pending tasks (L8)
    "cert_valid",        # binary SCMS certificate validity
    "success_ewma",      # EWMA over advertised-vs-observed discrepancy (L8)
    "latency_dev",       # observed minus advertised latency (L8)
    "uptime_stability",  # observed restarts / dropped sessions
)

# --- Vehicle features, PROJECT_SPEC.md 5.2 ---
VEHICLE_FEATURES: tuple[str, ...] = (
    "task_demand",
    "speed",
    "dwell_estimate",
)

# --- Edge features, PROJECT_SPEC.md 5.3 ---
# Single homogeneous edge type covering vehicle-RSU and RSU-RSU links (L6).
EDGE_FEATURES: tuple[str, ...] = (
    "link_latency",
    "signal_strength",
    "link_age",
    "same_segment",  # both endpoints share a backhaul_segment_id (L5/L6)
)

# Node feature matrix layout (DECISIONS.md D16): one homogeneous `x` whose width is
# the RSU block followed by the vehicle block. An RSU row fills the RSU block and
# zeroes the vehicle block; a vehicle row does the reverse.
NODE_FEATURE_NAMES: tuple[str, ...] = tuple(
    [f"rsu.{name}" for name in RSU_FEATURES]
    + [f"veh.{name}" for name in VEHICLE_FEATURES]
)

RSU_BLOCK = slice(0, len(RSU_FEATURES))
VEHICLE_BLOCK = slice(len(RSU_FEATURES), len(RSU_FEATURES) + len(VEHICLE_FEATURES))

NODE_FEATURE_DIM = len(NODE_FEATURE_NAMES)
EDGE_FEATURE_DIM = len(EDGE_FEATURES)

# Column index of a named RSU feature within `x`. Used by the selection rule to read
# advertised `load` straight out of the graph rather than carrying a parallel array.
RSU_COL = {name: i for i, name in enumerate(RSU_FEATURES)}
EDGE_COL = {name: i for i, name in enumerate(EDGE_FEATURES)}

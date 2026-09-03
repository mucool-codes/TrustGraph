"""Sanity visualisations.

Two plots, both diagnostic rather than decorative. They exist to answer questions that
summary statistics can hide:

  * `plot_topology` - are the backhaul segments actually contiguous stretches of road,
    or has the clustering scattered them? L5 only buys anything if they are
    geographically coherent, and a table of segment sizes cannot show that.
  * `plot_coverage_over_time` - is handoff really happening, or do vehicles sit in one
    coverage zone for the whole trace? A mean handoff rate can look healthy while a
    handful of vehicles do all the moving.

matplotlib is imported with the non-interactive Agg backend so these render the same
way headless as on a desktop.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .stats import CoverageSeries  # noqa: E402
from .topology import Topology  # noqa: E402
from .trace import Trace  # noqa: E402

# Qualitative, colourblind-safe (Okabe-Ito). Segments are a categorical variable, so
# a sequential colormap would imply an ordering between backhaul links that does not
# exist.
SEGMENT_COLOURS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)


def _segment_colour(segment_id: int) -> str:
    return SEGMENT_COLOURS[int(segment_id) % len(SEGMENT_COLOURS)]


def plot_topology(
    topology: Topology,
    trace: Trace,
    path: str | Path,
    timestep: int = 0,
    dpi: int = 150,
) -> Path:
    """Graph snapshot: roads, RSUs coloured by backhaul segment, coverage, vehicles.

    RSU-RSU edges are drawn in the segment colour when both endpoints share a segment
    and in grey when they do not - that is the `same_segment` edge feature made
    visible, and it is the thing message passing is meant to exploit (L5/L6).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    road = topology.road
    fig, ax = plt.subplots(figsize=(9.0, 9.0))

    # --- roads ---------------------------------------------------------------
    for a, b in road.segments():
        pa, pb = road.intersections[a], road.intersections[b]
        ax.plot(
            [pa[0], pb[0]],
            [pa[1], pb[1]],
            color="#d9d9d9",
            linewidth=3.0,
            solid_capstyle="round",
            zorder=1,
        )

    # --- coverage discs ------------------------------------------------------
    for r in range(topology.num_rsus):
        ax.add_patch(
            plt.Circle(
                tuple(topology.positions[r]),
                topology.coverage_radius_m,
                facecolor=_segment_colour(topology.backhaul_segment_id[r]),
                edgecolor="none",
                alpha=0.07,
                zorder=2,
            )
        )

    # --- RSU-RSU coordination edges (L6) -------------------------------------
    for i, j in topology.rsu_edges:
        shared = (
            topology.backhaul_segment_id[i] == topology.backhaul_segment_id[j]
        )
        pi, pj = topology.positions[i], topology.positions[j]
        ax.plot(
            [pi[0], pj[0]],
            [pi[1], pj[1]],
            color=_segment_colour(topology.backhaul_segment_id[i])
            if shared
            else "#9e9e9e",
            linewidth=1.8 if shared else 0.7,
            linestyle="-" if shared else (0, (4, 3)),
            alpha=0.9 if shared else 0.5,
            zorder=3,
        )

    # --- vehicles ------------------------------------------------------------
    positions = trace.positions[timestep]
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=14,
        marker="o",
        facecolor="#333333",
        edgecolor="none",
        alpha=0.75,
        label=f"vehicles (n={trace.num_vehicles})",
        zorder=4,
    )

    # --- RSUs ----------------------------------------------------------------
    for seg in range(topology.num_segments):
        mask = topology.backhaul_segment_id == seg
        ax.scatter(
            topology.positions[mask, 0],
            topology.positions[mask, 1],
            s=170,
            marker="^",
            color=_segment_colour(seg),
            edgecolor="white",
            linewidth=1.2,
            label=f"segment {seg} ({int(mask.sum())} RSUs)",
            zorder=5,
        )
    for r in range(topology.num_rsus):
        ax.annotate(
            str(r),
            tuple(topology.positions[r]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=7,
            color="#444444",
            zorder=6,
        )

    margin = 0.06 * road.extent_m
    ax.set_xlim(-margin, road.extent_m + margin)
    ax.set_ylim(-margin, road.extent_m + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"RSU topology at t={timestep}  -  "
        f"{topology.num_rsus} RSUs, {topology.num_segments} backhaul segments, "
        f"coverage {topology.coverage_radius_m:.0f} m\n"
        "solid coloured links: same_segment = 1   |   dashed grey: same_segment = 0",
        fontsize=10,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    ax.grid(True, color="#f0f0f0", linewidth=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_coverage_over_time(
    series: CoverageSeries,
    trace: Trace,
    path: str | Path,
    num_tracked_vehicles: int = 6,
    dpi: int = 150,
) -> Path:
    """Vehicle-RSU edge count over time, plus which RSU a few vehicles are attached to.

    The top panel is the aggregate the exit condition asks for. The bottom panel is
    the check the aggregate cannot make: a step in a vehicle's serving-RSU line is one
    handoff, so if those lines are flat, no handoff is happening no matter how the
    edge count fluctuates.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    times = np.arange(trace.num_steps) * trace.dt_s
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11.0, 7.0), sharex=True, height_ratios=[1.0, 1.15]
    )

    # --- aggregate edge count -------------------------------------------------
    top.plot(times, series.edge_count, color="#0072B2", linewidth=1.4)
    mean_edges = float(series.edge_count.mean())
    top.axhline(
        mean_edges,
        color="#D55E00",
        linestyle="--",
        linewidth=1.0,
        label=f"mean {mean_edges:.1f}",
    )
    top.set_ylabel("vehicle-RSU edges")
    top.set_title(
        "Vehicle-RSU edge count over time  -  the graph is genuinely dynamic",
        fontsize=11,
    )
    top.legend(loc="upper right", fontsize=8, frameon=False)
    top.grid(True, color="#f0f0f0", linewidth=0.5)
    top.set_axisbelow(True)

    # --- per-vehicle serving RSU ---------------------------------------------
    # The most mobile vehicles, so the panel shows handoff where it happens rather
    # than whichever vehicles happen to have the lowest ids.
    changes = (np.diff(series.serving, axis=0) != 0).sum(axis=0)
    tracked = np.argsort(-changes)[: min(num_tracked_vehicles, trace.num_vehicles)]
    tracked = np.sort(tracked)

    for k, v in enumerate(tracked):
        column = series.serving[:, v].astype(float)
        # A gap in coverage is a break in the line, not a drop to RSU -1.
        column[series.serving[:, v] < 0] = np.nan
        bottom.step(
            times,
            column,
            where="post",
            linewidth=1.3,
            color=SEGMENT_COLOURS[k % len(SEGMENT_COLOURS)],
            label=f"veh {int(v)} ({int(changes[v])} changes)",
        )

    bottom.set_xlabel("time (s)")
    bottom.set_ylabel("serving RSU index")
    bottom.set_title(
        "Serving RSU for the most mobile vehicles  -  each step is a handoff, "
        "each break is a coverage gap",
        fontsize=11,
    )
    bottom.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    bottom.grid(True, color="#f0f0f0", linewidth=0.5)
    bottom.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path

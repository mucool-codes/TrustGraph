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

from dataclasses import dataclass
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


# --- S2 figure palette -------------------------------------------------------------
# Categorical slots 1-3 of the reference data-viz palette, in fixed order. The first
# three slots validate on *all* pairs (not just adjacent ones), which matters here
# because the three group means cross each other during the onset ramp.
HEALTHY_COLOUR = "#2a78d6"       # slot 1: healthy RSUs on untouched segments
DEGRADED_COLOUR = "#eb6834"      # slot 2: degraded RSUs
SEGMENT_MATE_COLOUR = "#1baf7a"  # slot 3: healthy RSUs sharing a degraded segment
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
RAMP_WASH = "#f0efec"
# Sequential single-hue blue ramp, light -> dark, for the heatmap: success_ewma is a
# magnitude, so one hue - never a rainbow.
BLUE_RAMP = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)


@dataclass(frozen=True)
class EwmaPanel:
    """One column of the S2 exit figure. Plain arrays only - this module never imports
    the sealed ground truth; the calling script unseals and passes masks in (L4).

    Attributes:
        title: column heading.
        success_ewma: (num_steps, num_rsus) the observable feature.
        degraded: (num_rsus,) bool, from the sealed ground truth.
        segment_id: (num_rsus,) backhaul segment per RSU.
        fault_progress: (num_steps,) mean true degradation level over degraded RSUs.
        onset_step, ramp_steps, dt_s: the ramp window, for shading.
    """

    title: str
    success_ewma: np.ndarray
    degraded: np.ndarray
    segment_id: np.ndarray
    fault_progress: np.ndarray
    onset_step: int
    ramp_steps: int
    dt_s: float


def _style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=8)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def plot_success_ewma_over_time(
    panels: list[EwmaPanel], path: str | Path, dpi: int = 150
) -> Path:
    """Mean success_ewma over time, degraded vs healthy, one column per scenario.

    Top row: the group means. Healthy RSUs are split into those sharing a segment with
    a degraded RSU and those on untouched segments, so correlated (segment-level)
    degradation is visible as a difference between the two healthy lines and between
    columns. Thin lines are the individual degraded RSUs; the dotted grey line is the
    sealed ground truth `1 - fault progress`, so the feature's lag behind the true
    onset can be read directly. The shaded band is the onset ramp.

    Bottom row: success_ewma for every RSU, rows grouped by backhaul segment with a
    surface-coloured gap between segments. Degraded RSUs carry an orange marker at the
    left edge. At high rho the degraded rows sit together inside one segment block;
    at rho = 0 they are scattered across blocks.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.transforms import blended_transform_factory

    cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
    ncol = len(panels)
    fig, axes = plt.subplots(
        2,
        ncol,
        figsize=(6.8 * ncol, 10.0),
        height_ratios=[1.0, 1.15],
        squeeze=False,
        facecolor=SURFACE,
    )
    image = None

    for col, p in enumerate(panels):
        top, bottom = axes[0, col], axes[1, col]
        num_steps, num_rsus = p.success_ewma.shape
        times = np.arange(num_steps) * p.dt_s
        onset_s = p.onset_step * p.dt_s
        full_s = (p.onset_step + p.ramp_steps) * p.dt_s

        touched = np.isin(p.segment_id, p.segment_id[p.degraded])
        groups = (
            ("healthy, untouched segments", ~touched, HEALTHY_COLOUR),
            ("healthy, same segment as a degraded RSU", touched & ~p.degraded, SEGMENT_MATE_COLOUR),
            ("degraded (mean)", p.degraded, DEGRADED_COLOUR),
        )

        # --- top: group means -------------------------------------------------
        _style_axis(top)
        top.axvspan(onset_s, full_s, color=RAMP_WASH, lw=0, zorder=0)
        top.text(
            (onset_s + full_s) / 2, 1.06, "onset ramp", ha="center", va="bottom",
            fontsize=8, color=INK_SECONDARY,
        )
        top.grid(True, axis="y", color=GRIDLINE, linewidth=0.6)
        top.set_axisbelow(True)
        for r in np.flatnonzero(p.degraded):
            top.plot(times, p.success_ewma[:, r], color=DEGRADED_COLOUR, lw=0.8, alpha=0.3, zorder=2)
        top.plot(
            times, 1.0 - p.fault_progress, color=INK_MUTED, lw=1.3, ls=":",
            label="1 - true fault progress (sealed)", zorder=3,
        )
        end_values: list[float] = []
        for label, mask, colour in groups:
            if not mask.any():
                continue
            mean = p.success_ewma[:, mask].mean(axis=1)
            top.plot(times, mean, color=colour, lw=2.0, label=f"{label}, n={int(mask.sum())}", zorder=4)
            end_values.append(float(mean[-1]))
        # End-of-run value labels, spread apart vertically so near-equal means (the two
        # healthy groups usually are) do not print on top of each other.
        order_idx = np.argsort(end_values)
        placed: list[float] = []
        for i in order_idx:
            y = end_values[i]
            if placed and y - placed[-1] < 0.05:
                y = placed[-1] + 0.05
            placed.append(y)
            top.annotate(
                f"{end_values[i]:.2f}", (times[-1], y), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=8,
                color=INK_SECONDARY, annotation_clip=False,
            )
        top.set_ylim(0.0, 1.12)
        top.set_xlim(0, times[-1])
        top.set_ylabel("mean success_ewma")
        top.set_title(p.title, fontsize=11, color=INK_PRIMARY, loc="left")
        top.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=INK_SECONDARY)

        # --- bottom: per-RSU heatmap, grouped by segment ------------------------
        _style_axis(bottom)
        order = np.lexsort((np.arange(num_rsus), p.segment_id))
        image = bottom.imshow(
            p.success_ewma[:, order].T,
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            extent=(0.0, num_steps * p.dt_s, num_rsus - 0.5, -0.5),
        )
        seg_sorted = p.segment_id[order]
        for i in np.flatnonzero(np.diff(seg_sorted)) + 1:
            bottom.axhline(i - 0.5, color=SURFACE, lw=2.5)
        for x in (onset_s, full_s):
            bottom.axvline(x, color=INK_PRIMARY, lw=0.8, ls=(0, (4, 3)), alpha=0.6)
        bottom.set_yticks(np.arange(num_rsus))
        bottom.set_yticklabels(
            [f"s{int(p.segment_id[r])}  RSU {int(r):02d}" for r in order], fontsize=7
        )
        rows = np.flatnonzero(p.degraded[order])
        bottom.scatter(
            np.full(rows.size, -0.012), rows, marker="s", s=26, color=DEGRADED_COLOUR,
            transform=blended_transform_factory(bottom.transAxes, bottom.transData),
            clip_on=False, zorder=5,
        )
        bottom.set_xlabel("time (s)")
        bottom.set_title(
            "success_ewma per RSU, grouped by backhaul segment  (square = degraded)",
            fontsize=9, color=INK_SECONDARY, loc="left",
        )

    if image is not None:
        bar = fig.colorbar(image, ax=axes[1, :].tolist(), fraction=0.025, pad=0.02)
        bar.set_label("success_ewma", color=INK_SECONDARY)
        bar.ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=8)
        bar.outline.set_visible(False)

    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


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

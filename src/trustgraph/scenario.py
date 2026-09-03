"""Assembling a scenario from a config.

One place where (config, seed) becomes the static world - road layout, RSU placement,
backhaul segments, link model - and where a mobility trace is generated. Everything
else (the pipeline, the statistics, the plots, the scripts) builds its world through
here, so there is exactly one construction order and one set of RNG streams, and no
two consumers can disagree about what seed 20260903 means.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .graph import SnapshotBuilder
from .links import LinkModel, build_link_model
from .mobility import build_mobility
from .roads import RoadNetwork, build_road_network
from .topology import Topology, build_topology
from .trace import Trace


@dataclass(frozen=True)
class World:
    """Everything about a run that does not change once vehicles start moving."""

    road: RoadNetwork
    topology: Topology
    link_model: LinkModel


def build_world(cfg: Config) -> World:
    """Build the static world for `cfg`.

    Uses only the `topology` RNG stream, so the world is identical for two runs that
    differ in how much randomness mobility or feature generation consumed
    (DECISIONS.md D17).
    """
    road = build_road_network(cfg.road)
    topology = build_topology(cfg.topology, road, cfg.seeds.generator("topology"))
    return World(road=road, topology=topology, link_model=build_link_model(cfg.link))


def generate_trace(cfg: Config, world: World | None = None) -> Trace:
    """Simulate vehicle motion for the configured horizon.

    Called by `scripts/generate_trace.py`, which writes the result to disk. The
    pipeline never calls this - it reads the trace back (see `trace.py`).
    """
    world = world or build_world(cfg)
    source = build_mobility(cfg.mobility, world.road, cfg.seeds.generator("mobility"))
    return source.generate(int(cfg.scenario["num_steps"]), seed=cfg.seed)


def build_snapshot_builder(
    cfg: Config, world: World, trace: Trace
) -> SnapshotBuilder:
    """The graph constructor for a (world, trace) pair."""
    return SnapshotBuilder(
        topology=world.topology,
        trace=trace,
        link_model=world.link_model,
        cfg_graph=cfg.graph,
        rng=cfg.seeds.generator("features"),
    )

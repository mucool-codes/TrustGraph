"""Assembling a scenario from a config.

One place where (config, seed) becomes the static world - road layout, RSU placement,
backhaul segments, link model - and where a mobility trace is generated. Everything
else (the pipeline, the statistics, the plots, the scripts) builds its world through
here, so there is exactly one construction order and one set of RNG streams, and no
two consumers can disagree about what seed 20260903 means.

This module is on the training path and must stay free of the sealed ground truth:
the simulator that injects degradation lives in `simulator.py`, which imports this
module, never the other way round (L4, `tests/test_sealing.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .graph import SnapshotBuilder
from .links import LinkModel, build_link_model
from .mobility import build_mobility
from .observed import ObservedScenario
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
    cfg: Config, world: World, trace: Trace, observed: ObservedScenario
) -> SnapshotBuilder:
    """The graph constructor for a (world, trace, observable scenario) triple."""
    if observed.num_steps != trace.num_steps:
        raise ValueError(
            f"scenario has {observed.num_steps} steps but the trace has "
            f"{trace.num_steps}; they were generated from different configs"
        )
    return SnapshotBuilder(
        topology=world.topology,
        trace=trace,
        link_model=world.link_model,
        cfg_graph=cfg.graph,
        rsu_features=observed.rsu_features,
        rsu_active=observed.rsu_active,
        vehicle_task_demand=observed.vehicle_task_demand,
    )

"""Bus coordinates for drawing grids.

Checked 19 Oct 2026: neither the datakit raw tables nor the downloaded files carry bus/substation coordinates for
case118 or Texas2k, so the layout is a graph embedding (kamada_kawai seeded by spring_layout), cached in
static_bus.x/y by the caller (scripts/stress_and_accuracy_map.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _normalise(xy: np.ndarray) -> np.ndarray:
    xy = xy - xy.min(0)
    span = xy.max(0).max()
    return xy / span if span > 0 else xy


def layout(static_bus: pd.DataFrame, static_branch: pd.DataFrame, *, method: str = "auto",
           coords: pd.DataFrame | None = None, seed: int = 0) -> np.ndarray:
    """[B, 2] coordinates.

    method='auto': use ``coords`` (columns bus, x, y; e.g. substation lat/lon) if given and complete,
    else kamada_kawai seeded by spring_layout (networkx). Normalise to the unit box, keeping the aspect ratio.
    Deterministic for a given seed. Cache by writing back to static_bus.x/y in the caller.
    """
    import networkx as nx

    buses = static_bus.bus.to_numpy()
    if method == "auto" and coords is not None:
        c = coords.set_index("bus").reindex(buses)
        if c[["x", "y"]].notna().all().all():
            return _normalise(c[["x", "y"]].to_numpy(np.float64))
    G = nx.Graph()
    G.add_nodes_from(buses.tolist())
    G.add_edges_from(zip(static_branch.from_bus.tolist(), static_branch.to_bus.tolist()))
    init = nx.spring_layout(G, seed=seed, iterations=100)
    pos = nx.kamada_kawai_layout(G, pos=init)
    return _normalise(np.array([pos[b] for b in buses], dtype=np.float64))

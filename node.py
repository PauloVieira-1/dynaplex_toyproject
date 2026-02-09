from __future__ import annotations
from dataclasses import dataclass

# A single node in the supply chain: static configuration only.
# Based on Airplane in MDP examples from DynaPlex.

@dataclass(slots=True)
class Node:
    # Static configuration
    id: int
    name: str
    capacity: int
    node_type: str
    holding_cost: float
    backlog_cost: float
    order_cost: float
    lead_time: int
    upstream_ids: list[int]
    downstream_ids: list[int]

    def __init__(
        self,
        id: int,
        name: str,
        capacity: int,
        node_type: str,
        holding_cost: float,
        backlog_cost: float,
        order_cost: float,
        lead_time: int,
        upstream_ids: list[int],
        downstream_ids: list[int],
    ):
        # Validate input
        assert capacity > 0, "Capacity must be positive."
        assert holding_cost >= 0, "Holding cost must be non-negative."
        assert backlog_cost >= 0, "Backlog cost must be non-negative."
        assert order_cost >= 0, "Order cost must be non-negative."
        assert lead_time >= 0, "Lead time must be non-negative."

        self.id = id
        self.name = name
        self.capacity = capacity
        self.node_type = node_type
        self.holding_cost = holding_cost
        self.backlog_cost = backlog_cost
        self.order_cost = order_cost
        self.lead_time = lead_time
        self.upstream_ids = upstream_ids  # For multi-node systems
        self.downstream_ids = downstream_ids  # For multi-node systems


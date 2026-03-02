from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class Node:
    # Static configuration
    id: int
    name: str
    capacity: int
    holding_cost: float
    backlog_cost: float
    order_cost: float
    lead_time: int    
    # required_inventory: list[int] # This is the items required items per upstream node to produce a unit (eg: [1,3,4] --> needs 1 from upstream_ids[0]...)
    upstream_ids: list[int]
    downstream_ids: list[int]  ## Maybe not neccesary to store downstream??

    def __init__(
        self,
        id: int,
        name: str,
        capacity: int,
        holding_cost: float,
        backlog_cost: float,
        order_cost: float,
        lead_time: int,
        # required_inventory: list[int],
        upstream_ids: list[int],
        downstream_ids: list[int],
    ):

        assert capacity > 0, "Capacity must be positive."
        assert holding_cost >= 0, "Holding cost must be non-negative."
        assert backlog_cost >= 0, "Backlog cost must be non-negative."
        assert order_cost >= 0, "Order cost must be non-negative."
        assert lead_time >= 0, "Lead time must be non-negative."
        assert all(upstream_id > 0 for upstream_id in upstream_ids), "Upstream IDs must be positive."
        assert all(downstream_id > 0 for downstream_id in downstream_ids), "Downstream IDs must be positive."
        # assert all(required_inventory > 0 for required_inventory in required_inventory), "Required inventory must be positive."
        # assert len(required_inventory) == len(upstream_ids), "Required inventory and upstream IDs must have the same length."


        self.id = id
        self.name = name
        self.capacity = capacity
        self.holding_cost = holding_cost
        self.backlog_cost = backlog_cost
        self.order_cost = order_cost
        # self.required_inventory = required_inventory
        self.lead_time = lead_time
        self.upstream_ids = upstream_ids  
        self.downstream_ids = downstream_ids  


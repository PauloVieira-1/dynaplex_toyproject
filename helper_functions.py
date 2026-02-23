from typing import List, Tuple
from custom_types import SupplyChainState, NodeInfo
from dynaplex.modelling import TrajectoryContext, StateCategory
from node import Node

import numpy as np


def assert_state_valid(mdp, state: SupplyChainState):

    assert state.category == StateCategory.AWAIT_EVENT, "Not expecting an event right now."
    assert state.remaining_time > 0, "Simulation already finished."
    assert len(state.node_infos) == len(mdp.nodes), "State node infos length mismatch with number of nodes."


def process_inventory_and_pipeline(mdp, state: SupplyChainState) -> Tuple[List[int], List[int]]:
  
    inventories = []
    backorders_list = []

    for node, info in zip(mdp.nodes, state.node_infos):

        inventory = max(0, info.inventory_level)
        backorders = max(0, -info.inventory_level)

        # Process pipeline arrivals
        if node.lead_time > 0 and info.pipeline:

            arrived = info.pipeline.pop(0)
            inventory = min(node.capacity, inventory + arrived) # if too much then discarded (lost sales)

        # Fulfill backorders if possible
        fulfilled_backlog = min(inventory, backorders)
        inventory -= fulfilled_backlog
        backorders -= fulfilled_backlog

        inventories.append(inventory)
        backorders_list.append(backorders)

    return inventories, backorders_list


def process_demand(mdp, state: SupplyChainState, context: TrajectoryContext,
                   inventories: List[int], backorders_list: List[int]) -> None:


    last_node_index = len(mdp.nodes) - 1
    demand = context.rng.poisson(lam=5)  #! Example demand distribution for now (to be chnaged)

    inventory = inventories[last_node_index]
    backorders = backorders_list[last_node_index]

    fulfilled = min(demand, inventory)
    inventory -= fulfilled
    backorders += demand - fulfilled

    inventories[last_node_index] = inventory
    backorders_list[last_node_index] = backorders


def fulfill_upstream_orders(mdp, state: SupplyChainState, inventories: List[int]) -> None:

    # Iterate in reverse to ensure upstream nodes fulfill orders before downstream nodes

    for i in reversed(range(len(mdp.nodes))):

        node = mdp.nodes[i]
        info = state.node_infos[i]

        if not node.upstream_ids:  # Infinite supply for source node
            continue

        downstream_order = state.pending_orders[i]

        # Find upstream node index
        upstream_index = node.upstream_ids[0]  # Assuming single upstream node (to be extended for multiple upstream nodes)
        shipped = min(inventories[upstream_index], downstream_order)

        inventories[upstream_index] -= shipped

        # Add to pipeline or immediate inventory
        if node.lead_time > 0:
            if len(info.pipeline) < node.lead_time:
                info.pipeline.extend([0] * (node.lead_time - len(info.pipeline)))
            info.pipeline[-1] += shipped
        else:
            inventories[i] += shipped


def update_node_infos_and_costs(mdp, state: SupplyChainState, context: TrajectoryContext,
                                inventories: List[int], backorders_list: List[int]) -> None:


    for idx, (node, info) in enumerate(zip(mdp.nodes, state.node_infos)):
        
        inventory = inventories[idx]
        backorders = backorders_list[idx]

        info.inventory_level = inventory - backorders
        context.cumulative_cost += node.holding_cost * inventory
        context.cumulative_cost += node.backlog_cost * backorders
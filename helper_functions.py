from typing import List
from custom_types import SupplyChainState
from dynaplex.modelling import TrajectoryContext, StateCategory
from custom_types import Node, NodeInfo


# Section 1 - Helpers for the modify_state_with_event function
# -----------------------------------------------------------------------------------------------------------------


# This is a work in progress 
# I am thinking of abstracting some of the other validation and sanity check logic here or into something similaar
def assert_state_valid(mdp, state: SupplyChainState):

    assert state.category == StateCategory.AWAIT_EVENT, "Not expecting an event right now."
    assert state.remaining_time > 0, "Remaining time must be positive."
    assert len(state.node_infos) == len(mdp.nodes), "State node infos length mismatch with number of nodes."
    assert len(state.pending_orders) == len(mdp.nodes)
    assert state.current_node_index < len(mdp.nodes)


def advance_all_pipelines(mdp, state: SupplyChainState) -> None:

    for i, node in enumerate(mdp.nodes):
        info = state.node_infos[i]

        if node.lead_time > 0:

            while len(info.pipeline) < node.lead_time:
                info.pipeline.append(0)

            arrived_today = info.pipeline.pop(0)
            info.inventory_level = min(node.capacity, info.inventory_level + arrived_today)

            info.pipeline.append(0)


def fulfill_upstream_orders(mdp, state: SupplyChainState) -> None:
    
    # Iterate in reverse so upstream nodes fulfill orders before downstream nodes
    for i in reversed(range(len(mdp.nodes))):
        node = mdp.nodes[i]
        info = state.node_infos[i]

        # Skip source node
        if not node.upstream_ids:
            continue

        # Pending orders for this node (orders placed by this node to its upstream)
        order_qty = state.pending_orders[i]

        requested = order_qty
        if requested <= 0:
            continue

        available_components = []

        for upstream_id in node.upstream_ids:
            upstream_idx = upstream_id - 1
            upstream_info = state.node_infos[upstream_idx]

            available = max(0, upstream_info.inventory_level)
            available_components.append(available)

        assembled = min(min(available_components), requested)

        # Remove components from upstream inventories
        for upstream_id in node.upstream_ids:
            upstream_idx = upstream_id - 1
            state.node_infos[upstream_idx].inventory_level -= assembled

        unfulfilled = requested - assembled

        if node.lead_time > 0:

            '''
            The position (index) in the pipeline indicates when items arrive at the stockpoint:

                pipeline[0] = 3 means 3 units arrive tomorrow
                pipeline[1] = 5 means 5 units arrive in 2 days
            '''

            info.pipeline[0] += assembled
        else:
            info.inventory_level = min(
                info.inventory_level + assembled,
                node.capacity
            )

        state.pending_orders[i] = unfulfilled  # handles the remainder


def process_demand(mdp, state: SupplyChainState, context: TrajectoryContext) -> None:
    
    final_node_index = next(i for i, node in enumerate(mdp.nodes) if not node.downstream_ids)
    last_node_info = state.node_infos[final_node_index]

    #! Example demand distribution for now (to be changed later)
    # ASML has a pyramid like structure, should be put in a function at later stage

    demand = context.rng.poisson(lam=2)
    last_node_info.inventory_level -= demand


def update_node_infos_and_costs(mdp, state: SupplyChainState, context: TrajectoryContext) -> None:

    for node, info in zip(mdp.nodes, state.node_infos):

        inventory = max(0, info.inventory_level)
        backlog = max(0, -info.inventory_level)

        context.cumulative_cost += node.holding_cost * inventory
        context.cumulative_cost += node.backlog_cost * backlog


# Section 2 - Helpers for the modify_state_with_action function
# -----------------------------------------------------------------------------------------------------------------


def process_node_order(state: SupplyChainState, nodes: List[Node], action: int, context: TrajectoryContext) -> None:

    current_node_info: NodeInfo = state.node_infos[state.current_node_index]
    current_node: Node = nodes[state.current_node_index]

    # backorders: int = max(0, -current_node_info.inventory_level) 
    inventory: int = max(0, current_node_info.inventory_level)

    if action > 0:

        max_order = max(current_node.capacity - inventory, 0)
        order_qty = min(action, max_order)

        context.cumulative_cost += current_node.order_cost * order_qty

        # Orders always represent a request to upstream now 
        # In past implementation, if the node had no upstream, the system structure was ignored 

        if len(current_node.upstream_ids) > 0: # To distinguish between first node (infinite supply) and rest           

            state.pending_orders[state.current_node_index] += order_qty

        else:

            # if lead_time > 0, items always enter pipeline[-1]
            # if lead_time == 0, items arrive immediately

            if current_node.lead_time > 0:

                # Ensure pipeline is the correct length before inserting
                while len(current_node_info.pipeline) < current_node.lead_time:
                    current_node_info.pipeline.append(0)                         

                # Items enter the back of the pipeline; they arrive after lead_time days
                current_node_info.pipeline[-1] += order_qty

            else:

                # Added capacity cap for zero-lead-time source node 
                # no lead time means the items arrive immediately
                current_node_info.inventory_level = min(
                    current_node.capacity,
                    current_node_info.inventory_level + order_qty
                )


def modify_state_category(state: SupplyChainState, nodes: List[Node]) -> None:
    
    if state.current_node_index < len(nodes) - 1:

        state.current_node_index += 1
        state.category = StateCategory.AWAIT_ACTION

    else:
        state.category = StateCategory.AWAIT_EVENT

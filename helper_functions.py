from typing import List, Tuple
from custom_types import SupplyChainState
from dynaplex.modelling import TrajectoryContext, StateCategory



# This is a work in progress 
# I am thinking of abstracting some of the other validation and sanity check logic here or into something similaar
# -----------------------------------------------------------------------------------------------------------------

def assert_state_valid(mdp, state: SupplyChainState):

    assert state.category == StateCategory.AWAIT_EVENT, "Not expecting an event right now."
    assert state.remaining_time > 0, "Simulation already finished."
    assert len(state.node_infos) == len(mdp.nodes), "State node infos length mismatch with number of nodes."
    assert len(state.pending_orders) == len(mdp.nodes)
    assert state.current_node_index < len(mdp.nodes)

# -----------------------------------------------------------------------------------------------------------------



def process_inventory_and_pipeline(mdp, state: SupplyChainState) -> None:

    for node, info in zip(mdp.nodes, state.node_infos):

        # Separate inventory and backlog from NodeInfo
        # Pipeline arrivals are handled in fulfill_upstream_orders — do not process them here
        inventory = max(0, info.inventory_level)
        backorders = max(0, -info.inventory_level)

        info.inventory_level = inventory - backorders


def process_demand(mdp, state: SupplyChainState, context: TrajectoryContext) -> None:


    last_node_index = len(mdp.nodes) - 1 #!!! demand is only generated at last node !!!
    last_node_info = state.node_infos[last_node_index]


    #! Example demand distribution for now (to be chnaged later) 
    # ASML has a pyramid like structure, should be put in a function at later stage 

    demand = context.rng.poisson(lam=5)  

    available_inventory = max(0, last_node_info.inventory_level)
    fulfilled = min(available_inventory, demand)

    last_node_info.inventory_level = available_inventory - fulfilled - (demand - fulfilled)


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


        # Multi-node system (work in progress) ----> assumes each item from upstream nodes in the same 
        
        #-------------------------------------------------------------------

        total_shipped = 0

        for upstream_id in node.upstream_ids:
            upstream_idx = upstream_id - 1
            upstream_info = state.node_infos[upstream_idx]

            # Ship as much as possible (up to available inventory)
            shipped = min(upstream_info.inventory_level, order_qty - total_shipped)
            upstream_info.inventory_level -= shipped
            total_shipped += shipped

            if total_shipped >= order_qty:
                break

        # Any unfulfilled order becomes backlog (negative inventory)
        unfulfilled = order_qty - total_shipped
        info.inventory_level -= unfulfilled

        #-------------------------------------------------------------------

        # Single-node assumption
        #-------------------------------------------------------------------

        # upstream_index = node.upstream_ids[0] - 1  
        # shipped = min(inventories[upstream_index], downstream_order)

        # # Reduce upstream inventory by shipped quantity
        # inventories[upstream_index] -= shipped

        #-------------------------------------------------------------------

        if node.lead_time > 0:
            # Ensure pipeline has at least lead_time places 

            '''
            The position (index) in the pipeline indicates when items arrive at the stockpoint:

                pipeline[0] = 3 means 3 units arrive tomorrow
                pipeline[1] = 5 means 5 units arrive in 2 days
            
            '''

        # Shift pipeline daily
        if node.lead_time > 0 and info.pipeline:
            arrived_today = info.pipeline.pop(0)

            # Fill backorders first, then add remainder to inventory
            net = info.inventory_level + arrived_today
            info.inventory_level = min(node.capacity, net)

            while len(info.pipeline) < node.lead_time:
                info.pipeline.append(0)

            # Add shipped items to pipeline for future arrival
            info.pipeline[-1] += total_shipped
        else:
            # Immediate arrival for zero lead time
            info.inventory_level += total_shipped
            info.inventory_level = min(info.inventory_level, node.capacity)

        # Reset pending orders
        state.pending_orders[i] = 0



def update_node_infos_and_costs(mdp, state: SupplyChainState, context: TrajectoryContext) -> None:

    for node, info in zip(mdp.nodes, state.node_infos):

        inventory = max(0, info.inventory_level)
        backlog = max(0, -info.inventory_level)

        context.cumulative_cost += node.holding_cost * inventory
        context.cumulative_cost += node.backlog_cost * backlog
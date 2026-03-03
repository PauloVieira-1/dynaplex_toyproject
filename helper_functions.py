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

def advance_source_pipelines(mdp, state: SupplyChainState) -> None:

    for i, node in enumerate(mdp.nodes):
        if node.upstream_ids:  # not a source node!!!
            continue
        info = state.node_infos[i]

        # This is a source node meaning it has no upstream nodes 
        if node.lead_time > 0:

            while len(info.pipeline) < node.lead_time:
                info.pipeline.append(0)

            arrived_today = info.pipeline.pop(0)
            info.inventory_level = min(node.capacity, info.inventory_level + arrived_today)

            # Keep pipeline at correct length for future arrivals
            while len(info.pipeline) < node.lead_time:
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

        # Multi-node system (work in progress) ----> assumes each item from upstream nodes in the same 
        
        #-------------------------------------------------------------------
     

        requested = order_qty
        if requested <= 0:
            state.pending_orders[i] = 0
            continue

        possible = requested
        upstream_inventories = []

        for upstream_id in node.upstream_ids:
            upstream_idx = upstream_id - 1  
            upstream_info = state.node_infos[upstream_idx]
            available = max(0, upstream_info.inventory_level)   # never use negative inventory
            possible = min(possible, available)
            upstream_inventories.append((upstream_idx, available))

        # Now ship 'possible' units → subtract from EACH upstream
        shipped = possible

        for up_idx, _ in upstream_inventories:
            state.node_infos[up_idx].inventory_level -= shipped

        # Any unfulfilled order becomes backlog (negative inventory at this node)
        unfulfilled = requested - shipped

        #-------------------------------------------------------------------

        # Single-node assumption  (your original single upstream comment block - kept for reference)
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

            net = info.inventory_level + arrived_today - unfulfilled
            info.inventory_level = min(node.capacity, net)

            while len(info.pipeline) < node.lead_time:
                info.pipeline.append(0)

            # Add shipped items to pipeline for future arrival
            info.pipeline[-1] += shipped

        else:

            # Immediate arrival for zero lead time
            net = info.inventory_level + shipped - unfulfilled
            info.inventory_level = min(net, node.capacity)

        # Reset pending orders
        state.pending_orders[i] = 0

def process_demand(mdp, state: SupplyChainState, context: TrajectoryContext) -> None:
    
    final_node_index = next(i for i, node in enumerate(mdp.nodes) if not node.downstream_ids)
    last_node_info = state.node_infos[final_node_index]


    #! Example demand distribution for now (to be chnaged later) 
    # ASML has a pyramid like structure, should be put in a function at later stage 

    demand = context.rng.poisson(lam=5)  
    last_node_info.inventory_level -= demand



def update_node_infos_and_costs(mdp, state: SupplyChainState, context: TrajectoryContext) -> None:

    for node, info in zip(mdp.nodes, state.node_infos):

        inventory = max(0, info.inventory_level)
        backlog = max(0, -info.inventory_level)

        context.cumulative_cost += node.holding_cost * inventory
        context.cumulative_cost += node.backlog_cost * backlog
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



def process_inventory_and_pipeline(mdp, state: SupplyChainState) -> Tuple[List[int], List[int]]:
  
    inventories = []
    backorders_list = []


    for node, info in zip(mdp.nodes, state.node_infos):

        inventory = max(0, info.inventory_level)
        backorders = max(0, -info.inventory_level)

        if node.lead_time > 0 and info.pipeline:

            arrived = 0
            if info.pipeline:
                arrived = info.pipeline.pop(0)
                
            # if too much arrives then discarded (lost sales). 
            # there are other checks for this so i may remove these lines

            inventory = min(node.capacity, inventory + arrived)
  

    
        # Fulfill backorders if possible
        fulfilled_backlog = min(inventory, backorders)
        inventory -= fulfilled_backlog
        backorders -= fulfilled_backlog

        inventories.append(inventory)
        backorders_list.append(backorders)

    return inventories, backorders_list


def process_demand(mdp, state: SupplyChainState, context: TrajectoryContext,
                   inventories: List[int], backorders_list: List[int]) -> None:


    last_node_index = len(mdp.nodes) - 1 #!!! demand is only generated at last node !!!

    #! Example demand distribution for now (to be chnaged later) 
    # ASML has a pyramid like structure, should be put in a function at later stage 
    demand = context.rng.poisson(lam=5)  

    inventory = inventories[last_node_index]
    backorders = backorders_list[last_node_index]

    fulfilled = min(demand, inventory)
    inventory -= fulfilled
    backorders += demand - fulfilled

    inventories[last_node_index] = inventory
    backorders_list[last_node_index] = backorders


def fulfill_upstream_orders(mdp, state: SupplyChainState, inventories: List[int]) -> None:
    
    # Iterate in reverse so upstream nodes fulfill orders before downstream nodes
    for i in reversed(range(len(mdp.nodes))):
        node = mdp.nodes[i]
        info = state.node_infos[i]

        # Skip source node
        if not node.upstream_ids:
            continue

        # Pending orders for this node (orders placed by this node to its upstream)
        order_qty = state.pending_orders[i]

        # if downstream_order == 0:
        #     continue


        # Multi-node system (work in progress) ----> assumes each item from upstream nodes in the same 
        
        #-------------------------------------------------------------------

        total_shipped = 0

        for upstream_node_id in node.upstream_ids:

            # print(f"{upstream_node_id} being fulfilled")
            
            upstream_index = upstream_node_id - 1
            available = inventories[upstream_index]
            shipped = min(available, order_qty - total_shipped)

            inventories[upstream_index] -= shipped
            total_shipped += shipped

            if total_shipped == order_qty:
                break

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

                pipeline[0] = 3  means 3 units arrive tomorrow
                pipeline[1] = 5 means 5 units arrive in 2 days
            
            '''


        # Shift pipeline daily

        if node.lead_time > 0:
            arrived_today = info.pipeline.pop(0) if info.pipeline else 0
            inventories[i] = min(node.capacity, inventories[i] + arrived_today)
            
            if node.lead_time > 0:
                while len(info.pipeline) < node.lead_time - 1:
                    info.pipeline.append(0)
                info.pipeline.append(total_shipped)
            

        else:
            inventories[i] = min(node.capacity, inventories[i] + total_shipped)

        # Reset pending orders after fulfillment
        state.pending_orders[i] = 0



def update_node_infos_and_costs(mdp, state: SupplyChainState, context: TrajectoryContext,
                                inventories: List[int], backorders_list: List[int]) -> None:


    for index, (node, info) in enumerate(zip(mdp.nodes, state.node_infos)):
        
        inventory = inventories[index]
        backorders = backorders_list[index]

        info.inventory_level = inventory - backorders

        context.cumulative_cost += node.holding_cost * inventory
        context.cumulative_cost += node.backlog_cost * backorders
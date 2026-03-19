
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from custom_types import SupplyChainState, NodeInfo, Node
import numpy as np
from numpy.typing import NDArray
from dynaplex.modelling import StateCategory, TrajectoryContext, HorizonType, discover_num_features, Features
from helper_functions import *




@dataclass(slots=True)
class SupplyChainMDP:

    nodes: List[Node]
    initial_horizon: int
    horizon_type: HorizonType
    num_features: int
    num_actions: int
    action_dims: List[int]
    

    def __init__(self, nodes: List[Node], initial_horizon: int):


        # Maybe abstract this into a function later
        # ---------------------------------------------------------------------------------------------
        
        assert len(nodes) > 0, "Supply chain must have at least one node."
        assert initial_horizon > 0, "Initial horizon must be positive."
        assert all(node.capacity > 0 for node in nodes), "All nodes must have positive capacity."
        assert all(node.holding_cost >= 0 for node in nodes), "All nodes must have non-negative holding cost."
        assert all(node.backlog_cost >= 0 for node in nodes), "All nodes must have non-negative backlog cost."
        assert all(node.order_cost >= 0 for node in nodes), "All nodes must have non-negative order cost."
        assert all(node.lead_time >= 0 for node in nodes), "All nodes must have non-negative lead time."

        # ---------------------------------------------------------------------------------------------

        self.nodes = nodes
        self.initial_horizon = initial_horizon
        self.horizon_type = HorizonType.INFINITE  


        self.action_dims = [node.capacity + 1 for node in nodes] # Each node can order from 0 up to its capacity

        # Max possible actions for any single node. 
        # Maximum because the PPO output layer needs to accommodate the largest action space among the nodes, even if some nodes have smaller action spaces
        self.num_actions = max(node.capacity for node in nodes) + 1 

        self.num_features = discover_num_features(self)



    def get_initial_state(self, context: TrajectoryContext) -> SupplyChainState:
        
        node_infos = []
        
        for node in self.nodes:
            node_infos.append(
                NodeInfo(
                    inventory_level=node.capacity // 2, # initially half full
                    pipeline=[0 for _ in range(node.lead_time)],
                )
            )

        return SupplyChainState(
            node_infos=node_infos,
            day=0,
            remaining_time=self.initial_horizon,
            category=StateCategory.AWAIT_ACTION, 
            current_node_index=0,
            pending_orders=[0 for x in self.nodes],
        )


    def modify_state_with_event(self, state: SupplyChainState, context: TrajectoryContext) -> None:
        """
        This is a function that modifies the state of the supply chain based on the current event, which is the arrival of an order.
        """

        # I decided to abstract each "step" into a functions in helper_functions.py because modify_state_with_event grew far too big 
        # I will possibly be doing the same for modify_state_with_action

        advance_all_pipelines(self, state) # The functionality to advance the pipeline for all nodes has been abstracted here 

        fulfill_upstream_orders(self, state)

        process_demand(self, state, context)

        update_node_infos_and_costs(self, state, context)

        assert_state_valid(self, state)

        state.current_node_index = 0
        state.day += 1
        state.remaining_time -= 1

        if state.remaining_time > 0:
            state.category = StateCategory.AWAIT_ACTION
        else:
            state.category = StateCategory.FINAL



    def modify_state_with_action(self, state: SupplyChainState, context: TrajectoryContext, action: int) -> None:
        
        
        # ---------------------------------------------------------------------------------------------------

        # Before, I encoded the single integer action into a list of order quantities for each node. 
        # Now, since we consdier the system sequentially and each node makes its decision one at a time, 
        # we can directly use the action as the order quantity for the current node without encoding/decoding.

            # action_list = decode_action(action, self.action_dims)

        # ---------------------------------------------------------------------------------------------------
        
            # Validates the action, then adds the order quantity either into
            # pending_orders (if the node has upstream suppliers) or directly into the pipeline
            # or inventory (if it is a source node with infinite supply).


        process_node_order(state, self.nodes, action, context)


        # This function is abstracted out of modify_state_with_action
        # It sets the state category to either AWAIT_EVENT or AWAIT_ACTION depending on if all nodes have been processed 


        modify_state_category(state, self.nodes)
            

    
    # OPTION 1 - features for all nodes at once
    # -----------------------------------------------

    def write_features(self, state: SupplyChainState, features: Features) -> None:
        max_lt = max(node.lead_time for node in self.nodes)
        
        for node_static, node_dynamic in zip(self.nodes, state.node_infos):
            inventory = max(0, node_dynamic.inventory_level)
            backlog = max(0, -node_dynamic.inventory_level)
            features.append(inventory / node_static.capacity)
            features.append(backlog / node_static.capacity)

            for s in range(max_lt):
                val = node_dynamic.pipeline[s] if s < len(node_dynamic.pipeline) else 0
                features.append(val / node_static.capacity)


        features.append(state.current_node_index / len(self.nodes))
        features.append(state.remaining_time / self.initial_horizon)



    # OPTION 2 - features for each node
    # -----------------------------------------------

    # def write_features(self, state: SupplyChainState, features: Features) -> None:
            
    #     index_node = state.current_node_index
    #     node_static = self.nodes[index_node]
    #     node_dynamic = state.node_infos[index_node]

    #     inventory = max(0, node_dynamic.inventory_level)
    #     backlog = max(0, -node_dynamic.inventory_level)
        
    #     features.append(inventory / node_static.capacity)
    #     features.append(backlog / node_static.capacity)
    #     features.append(sum(node_dynamic.pipeline) / node_static.capacity)

    #     features.append(index_node / len(self.nodes))
    #     features.append(state.remaining_time / self.initial_horizon)



    def write_action_validity(self, state: SupplyChainState, valid: NDArray[np.bool_]) -> None:

            current_node_static: Node = self.nodes[state.current_node_index]
            current_node_dynamic: NodeInfo = state.node_infos[state.current_node_index]

            current_inv = max(0, current_node_dynamic.inventory_level)
            max_order = max(0, current_node_static.capacity - current_inv)

            for action_qty in range(current_node_static.capacity + 1):

                if action_qty <= max_order:
                    valid[action_qty] = True
                else:                    
                    valid[action_qty] = False




#--------------------------------------------------------------------------------------------------



from __future__ import annotations
from dataclasses import dataclass
from typing import List
from custom_types import SupplyChainState, NodeInfo

import numpy as np
from numpy.typing import NDArray

from dynaplex.modelling import StateCategory, TrajectoryContext, HorizonType, discover_num_features, Features
from node import Node
from policy import BasePolicy, BaseStockPolicy
from graph import create_graph_window
from PPO import decode_action, encode_action, train_PPO

from helper_functions import *



# Supply Chain MDP
# -----------------

@dataclass(slots=True)
class SupplyChainMDP:
    nodes: List[Node]
    initial_horizon: int
    horizon_type: HorizonType
    num_features: int
    num_actions: int
    action_dims: List[int]

    def __init__(self, nodes: List[Node], initial_horizon: int):


        # Maybe abstract this functionality into a function
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
        self.horizon_type = HorizonType.FINITE # should work on making finite 


        self.action_dims = [node.capacity + 1 for node in nodes] # Each node can order from 0 up to its capacity

        # Max possible actions for any single node. 
        # Maximum because the PPO output layer needs to accommodate the largest action space among the nodes, even if some nodes have smaller action spaces.
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
            remaining_time=self.initial_horizon,
            day=0,
            category=StateCategory.AWAIT_ACTION,
            current_node_index=0,
            pending_orders=[0 for x in self.nodes],
        )


    def modify_state_with_event(self, state: SupplyChainState, context: TrajectoryContext) -> None:

        # I decided to abstract each "step" to a functions in helper_functions.py because this function grew far too big 
        # Considering doing the same for modify_state_with_action


        inventories, backorders_list = process_inventory_and_pipeline(self, state)

    # 1) Process demand at the last node
        process_demand(self, state, context, inventories, backorders_list)

    # 2) Fulfill upstream orders
        fulfill_upstream_orders(self, state, inventories)

    # 3) Update node infos and compute costs
        update_node_infos_and_costs(self, state, context, inventories, backorders_list)

    # Check state validity after all updates
        assert_state_valid(self, state)


        # --------------------------------------------------------------------
        # Reset to first node for next day (should be infinite, rather than finite system??)
        # This happens once only after all nodes have processed daily events 

        state.current_node_index = 0
        state.day += 1
        state.remaining_time -= 1

        # --------------------------------------------------------------------


        if state.remaining_time <= 0:
            state.category = StateCategory.FINAL

        else:
            state.category = StateCategory.AWAIT_ACTION



    def modify_state_with_action(self, state: SupplyChainState, context: TrajectoryContext, action: int) -> None:
        

        # assert state.category == StateCategory.AWAIT_ACTION, "Not expecting an action."
        # assert state.remaining_time > 0, "Simulation finished."

        # ---------------------------------------------------------------------------------------------------

        # Before, there I encoded the single integer action into a list of order quantities for each node. 
        # Now, since we consdier the system sequentially and each node makes its decision one at a time, 
        # we can directly use the action as the order quantity for the current node without encoding/decoding.

        # action_list = decode_action(action, self.action_dims)

        # ---------------------------------------------------------------------------------------------------


        current_node_info: NodeInfo = state.node_infos[state.current_node_index]
        current_node: Node = self.nodes[state.current_node_index]

        # backorders: int = max(0, -current_node_info.inventory_level) 
        inventory: int = max(0, current_node_info.inventory_level)

        if action > 0:

            max_order = max(current_node.capacity - inventory, 0)
            order_qty = min(action, max_order)

            context.cumulative_cost += current_node.order_cost * order_qty

            # Orders always represent a request to upstream now 
            # In past implementation, if the node had no upstream, the system structure was ignored 

            if len(current_node.upstream_ids) > 0: # To distinguish between first node and rest

                upstream_node_index = current_node.upstream_ids[0] - 1 # Assuming single upstream (will chnage later for multiple)
                state.pending_orders[upstream_node_index] += order_qty 

                # print(f"{upstream_node_index}, {state.pending_orders}")

            else:

                if current_node.lead_time > 0:

                    # Always go into pipeline (including source node)
                    if len(current_node_info.pipeline) < current_node.lead_time:
                        current_node_info.pipeline.extend(
                            [0] * (current_node.lead_time - len(current_node_info.pipeline))
                        )
                        current_node_info.inventory_level = min(current_node.capacity, current_node_info.inventory_level + order_qty)               
                    else:

                        # Immediate arrival (even for source)
                        current_node_info.inventory_level += order_qty



        # Transition logic might be incorrect? 
        # Should be checked 
        # --------------------------------------------------------------------

        if state.current_node_index < len(self.nodes) - 1:
            state.current_node_index += 1
            state.category = StateCategory.AWAIT_ACTION
            # print(f"Moving to node {state.current_node_index}.....") 
        else:
            state.category = StateCategory.AWAIT_EVENT
            
        # --------------------------------------------------------------------


    def write_features(self, state: SupplyChainState, features: Features) -> None:

        for node_static, node_dynamic in zip(self.nodes, state.node_infos):

            inventory = max(0, node_dynamic.inventory_level)
            backlog = max(0, -node_dynamic.inventory_level)
            
            features.append(inventory / node_static.capacity)
            features.append(backlog / node_static.capacity)
            features.append(sum(node_dynamic.pipeline) / node_static.capacity)
            
        features.append(state.remaining_time / self.initial_horizon)


    def write_action_validity(self, state: SupplyChainState, valid: NDArray[np.bool_]) -> None:

            current_node_static: Node = self.nodes[state.current_node_index]
            current_node_dynamic: NodeInfo = state.node_infos[state.current_node_index]

            current_inv = max(0, current_node_dynamic.inventory_level)
            max_order = max(0, current_node_static.capacity - current_inv)

            for action_qty in range(self.num_actions):

                if action_qty <= max_order:
                    valid[action_qty] = True
                else:                    
                    valid[action_qty] = False


#--------------------------------------------------------------------------------------------------



# Simulation
# ------------

def simulate_episode(mdp: SupplyChainMDP, policy, *, seed: int = 42) -> None:
    
    context = TrajectoryContext(rng=np.random.default_rng(seed))
    state = mdp.get_initial_state(context)

    print("=" * 80)
    print("Simulating episode...")
    print("-" * 80)

    while state.category != StateCategory.FINAL:

        if state.category == StateCategory.AWAIT_EVENT:

            print(f"\n--- Day {state.day} ---\n")
            
            mdp.modify_state_with_event(state, context)
            
            print(f"    Post-Event State: {state}")

        elif state.category == StateCategory.AWAIT_ACTION:
            current_node_idx = state.current_node_index
            current_node_name = mdp.nodes[current_node_idx].name
            
            # Temporary solution (currently each node starts with its own policy, 
            # but maybe it is better to have all nodes share a single policy at start too???
            # In this scenary Lists would be used for values like target inventory and safety stock, rather than having a policy instance for each node.?

            # ------------------------------------------------------------------------

            if isinstance(policy, list): 

                current_node_policy = policy[current_node_idx]
                current_node_info = state.node_infos[current_node_idx]
                action = current_node_policy.get_action(current_node_info)

            else:
                action = policy.get_action(state)

            # ------------------------------------------------------------------------

            print(f"         Decision for {current_node_name} (Index {current_node_idx}): Order {action}")

            mdp.modify_state_with_action(state, context, action)

        else:
            raise RuntimeError(f"Unexpected state category: {state.category}")

    print("-" * 80)
    print(f"Episode finished. \n Total cost: {context.cumulative_cost:.2f}")



def main() -> None:

    # Initialize MDP and Policy
    # -----------------------

    node_1 = Node(
        id=1,
        name="Node_1",
        capacity=20,
        holding_cost=1.0,
        backlog_cost=5.0,
        order_cost=2.0,
        lead_time=2,
        upstream_ids=[],
        downstream_ids=[],
    )

    policy_node_1 = BaseStockPolicy(node=node_1, target_inventory=10, safety_stock=5, price_per_unit=20.0)

    node_2 = Node(
        id=2,
        name="Node_2",
        capacity=15,
        holding_cost=0.5,
        backlog_cost=3.0,
        order_cost=1.5,
        lead_time=1,
        upstream_ids=[1],
        downstream_ids=[],
    )

    policy_node_2 = BaseStockPolicy(node=node_2, target_inventory=8, safety_stock=3, price_per_unit=20.0)

    node_3 = Node(
        id=3,
        name="Node_3",
        capacity=10,
        holding_cost=0.2,
        backlog_cost=10.0,
        order_cost=1.0,
        lead_time=0,
        upstream_ids=[2],
        downstream_ids=[],
    )

    policy_node_3 = BaseStockPolicy(node=node_3, target_inventory=5, safety_stock=2, price_per_unit=20.0)

    mdp = SupplyChainMDP(
        nodes=[node_1, node_2, node_3],   
        initial_horizon=15,
    )


    # Generate Visualization
    # ------------------------------------------------


    # state = mdp.get_initial_state(TrajectoryContext(rng=np.random.default_rng(1)))

    # nodes = mdp.nodes

    # connections = []
    # for node in nodes:
    #     for upstream_id in node.upstream_ids:
    #         connections.append((next(n for n in nodes if n.id == upstream_id), node))

    # state_by_id = {node.id: node_info for node, node_info in zip(nodes, state.node_infos)}
    # # create_graph_window(nodes, connections, state_by_id)




    # Run baseline simulation with the initial policy
    # ------------------------------------------------

    policy_list = [policy_node_1, policy_node_2, policy_node_3]

    simulate_episode(mdp, policy_list, seed=42)

    # Initialize PPO Trainer and Train Policy
    # ------------------------------------------------

    number_iterations = 50
    trained_policy = train_PPO(mdp, number_iterations=number_iterations)

    # Run simulation with the trained policy
    # ------------------------------------------------

    print("Simulating episode with trained PPO policy...")
    simulate_episode(mdp, trained_policy, seed=54)


if __name__ == "__main__":
    main()

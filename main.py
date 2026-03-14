from __future__ import annotations
from dataclasses import dataclass
from typing import List
from custom_types import SupplyChainState, NodeInfo, Node

import numpy as np
from numpy.typing import NDArray

from dynaplex.modelling import StateCategory, TrajectoryContext, HorizonType, discover_num_features, Features
from models.PPO import train_PPO
from evaluation.record import EpisodeRecorder
from evaluation.plots import plot_results


from helper_functions import *

from assembly_tree import AssemblyTree

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
        self.horizon_type = HorizonType.FINITE # should work on making infinite 


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
            remaining_time=self.initial_horizon,
            day=0,
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


        # ----------------------------------------------------------------------------------------
        # Reset to first node for next day 
        # This happens once only after all nodes have processed daily events 

        # I am not sure how something like this would work in an infinite system 

        state.current_node_index = 0
        state.day += 1
        state.remaining_time -= 1

        # -----------------------------------------------------------------------------------------------


        if state.remaining_time <= 0:
            state.category = StateCategory.FINAL

        else:
            state.category = StateCategory.AWAIT_ACTION



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
            

    
    # OPTION 2 - features for all nodes at once
    # -----------------------------------------------

    def write_features(self, state: SupplyChainState, features: Features) -> None:

        for node_static, node_dynamic in zip(self.nodes, state.node_infos):

            inventory = max(0, node_dynamic.inventory_level)
            backlog = max(0, -node_dynamic.inventory_level)
            
            features.append(inventory / node_static.capacity)
            features.append(backlog / node_static.capacity)
            features.append(sum(node_dynamic.pipeline) / node_static.capacity)
            
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

            for action_qty in range(self.num_actions):

                if action_qty <= max_order:
                    valid[action_qty] = True
                else:                    
                    valid[action_qty] = False




#--------------------------------------------------------------------------------------------------



# Simulation
# ------------

def simulate_episode(mdp: SupplyChainMDP, policy, *, seed: int = 42, name: str = "episode", recorder: EpisodeRecorder = None) -> None:
    
    context = TrajectoryContext(rng=np.random.default_rng(seed))
    state = mdp.get_initial_state(context)

    if recorder is not None:                                                    
        recorder.initialise([n.name for n in mdp.nodes])                     

    print("=" * 80)
    print(f'Simulating episode: {name}')
    print("-" * 80)

    while state.category != StateCategory.FINAL:

        if state.category == StateCategory.AWAIT_EVENT:

            print(f"\n--- Day {state.day} ---\n")
            
            mdp.modify_state_with_event(state, context)
            
            print(f"    Post-Event State: {state}")

            if recorder is not None:                                           
                recorder.record_state(state, context.cumulative_cost)  

        elif state.category == StateCategory.AWAIT_ACTION:
            
            current_node_idx = state.current_node_index
            current_node_name = mdp.nodes[current_node_idx].name
            
            # Temporary solution belown (currently each node starts with its own policy, 
            # but maybe it is better to have all nodes share a single policy at start too???)
            # In this scenario lists or dictionaries would be used for values like target inventory and safety stock, rather than having a policy instance for each node?

            # ------------------------------------------------------------------------

            if isinstance(policy, list): 

                current_node_policy = policy[current_node_idx]
                current_node_info = state.node_infos[current_node_idx]
                action = current_node_policy.get_action(current_node_info)

            else:
                action = policy.get_action(state)

            # ------------------------------------------------------------------------

            if recorder is not None:
                recorder.record_action(current_node_idx, action)

            print(f"         Decision for {current_node_name} (Index {current_node_idx}): Order {action}")

            mdp.modify_state_with_action(state, context, action)

        else:
            raise RuntimeError(f"Unexpected state category: {state.category}")

    print("-" * 80)
    print(f"Episode finished. \n Total cost: {context.cumulative_cost:.2f}")

    if recorder is not None:
        recorder.save()



def main() -> None:

    # Run simulation with random orders 
    # ------------------------------------------------

    tree = AssemblyTree("config/chain_2.json")
    tree.create_tree_from_json()
    
    policy_list = tree.get_policy_list()
    node_list = tree.get_assembly_tree()

    mdp = SupplyChainMDP(
        nodes = node_list,
        initial_horizon=15,
    )

    recorder = EpisodeRecorder("results/random.csv")
    simulate_episode(mdp, policy_list, seed=42, name="Random", recorder=recorder)



    # Run baseline simulation with the initial policy
    # ------------------------------------------------    

    tree = AssemblyTree("config/chain_1.json")
    tree.create_tree_from_json()
    
    policy_list_2 = tree.get_policy_list()
    node_list_2 = tree.get_assembly_tree()

    mdp_2 = SupplyChainMDP(
        nodes = node_list_2,
        initial_horizon=15,
    )

    recorder = EpisodeRecorder("results/base_stock.csv")
    simulate_episode(mdp_2, policy_list_2, seed=42, name="Base Stock", recorder=recorder)


    # Run simulation with the trained policy
    # ------------------------------------------------

    number_iterations = 50
    trained_policy = train_PPO(mdp, number_iterations=number_iterations)

    print("Simulating episode with trained PPO policy...")

    recorder = EpisodeRecorder("results/PPO_trained.csv")
    simulate_episode(mdp, trained_policy, seed=54, name="PPO", recorder=recorder)


    # Generate plots of results
    # ------------------------------------------------

    plot_results(
        {
            "Random": "results/random.csv",
            "Base Stock": "results/base_stock.csv",
            "PPO": "results/PPO_trained.csv",
        }
    )



if __name__ == "__main__":
    main()
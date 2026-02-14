from __future__ import annotations
from dataclasses import dataclass
from typing import List
from custom_types import SupplyChainState, NodeInfo

import numpy as np

from dynaplex.modelling import StateCategory, TrajectoryContext, HorizonType, discover_num_features, Features
from node import Node
from policy import BasePolicy, BaseStockPolicy
from graph import create_graph_window
from PPO import decode_action, encode_action, train_PPO
from numpy.typing import NDArray


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

        assert len(nodes) > 0, "Supply chain must have at least one node."
        assert initial_horizon > 0, "Initial horizon must be positive."
        assert all(node.capacity > 0 for node in nodes), "All nodes must have positive capacity."
        assert all(node.holding_cost >= 0 for node in nodes), "All nodes must have non-negative holding cost."
        assert all(node.backlog_cost >= 0 for node in nodes), "All nodes must have non-negative backlog cost."
        assert all(node.order_cost >= 0 for node in nodes), "All nodes must have non-negative order cost."
        assert all(node.lead_time >= 0 for node in nodes), "All nodes must have non-negative lead time."

        self.nodes = nodes
        self.initial_horizon = initial_horizon
        self.horizon_type = HorizonType.FINITE

        self.action_dims = [node.capacity + 1 for node in nodes] # Each node can order from 0 up to its capacity, inclusive
        self.num_actions = np.prod(self.action_dims)

        self.num_features = discover_num_features(self)




    def get_initial_state(self, context: TrajectoryContext) -> SupplyChainState:
        
        node_infos = []
        
        for node in self.nodes:
            node_infos.append(
                NodeInfo(
                    inventory_level=node.capacity // 2,
                    pipeline=[0 for _ in range(node.lead_time)],
                )
            )
        return SupplyChainState(
            node_infos=node_infos,
            remaining_time=self.initial_horizon,
            day=0,
            category=StateCategory.AWAIT_EVENT,
            num_actions=self.num_actions,
            horizon_type=self.horizon_type, 
        )


    def modify_state_with_event(self, state: SupplyChainState, context: TrajectoryContext) -> None:
        """
        Process exogenous events.
        receive arrivals + realize demand.
        Moves state from AWAIT_EVENT -> AWAIT_ACTION.
        """

        # Sanity checks
        assert state.category == StateCategory.AWAIT_EVENT, "Not expecting an event right now."
        assert state.remaining_time > 0, "Simulation already finished."

        for node, node_info in zip(self.nodes, state.node_infos): # Iterate through nodes to process events

            # Calculate inventory and backorders based on inventory_level
            inventory: int = max(0, node_info.inventory_level)
            backorders: int = max(0, -node_info.inventory_level)

            # Receive arrivals from the pipeline (lead time)
            if node.lead_time > 0 and node_info.pipeline:
                arrived = node_info.pipeline.pop(0)
                inventory = min(node.capacity, inventory + arrived)  # Orders that arrive but exceed capacity are lost

            # clear backorders if able to
            fulfilled_backlog = min(inventory, backorders)
            inventory -= fulfilled_backlog
            backorders -= fulfilled_backlog

            # Demand realization
            demand = int(context.rng.integers(low=0, high=10))  # Comparable to "price" from dynaplex example

            fulfilled = min(demand, inventory)
            inventory -= fulfilled
            backorders += demand - fulfilled

            # Update node state
            node_info.inventory_level = inventory - backorders

            # Holding and backlog costs for the day
            context.cumulative_cost += node.holding_cost * inventory
            context.cumulative_cost += node.backlog_cost * backorders

        # Advance time
        state.day += 1
        state.remaining_time -= 1
        context.time_elapsed += 1

        if state.remaining_time <= 0:
            state.category = StateCategory.FINAL
        else:
            state.category = StateCategory.AWAIT_ACTION



    def modify_state_with_action(self, state: SupplyChainState, context: TrajectoryContext, action: int) -> None:
        """
        Apply actions (order quantities).
        Moves state from AWAIT_ACTION -> AWAIT_EVENT or FINAL.
        """

        assert state.category == StateCategory.AWAIT_ACTION, "Not expecting an action."
        assert state.remaining_time > 0, "Simulation already finished."


        action_list = decode_action(action, self.action_dims)


        for node, node_info, action_qty in zip(self.nodes, state.node_infos, action_list): # Iterate through nodes to process actions

            # print(f"Processing action for node {node.name}: order {action} units!!!!")
            # print(f"{node_info}!")
            # print(f"{action}!!!")
            # print(f'{list(zip(self.nodes, state.node_infos, actions))}')

            # Calculate inventory and backorders
            backorders: int = max(0, -node_info.inventory_level)
            inventory: int = max(0, node_info.inventory_level)

            if action_qty > 0:
                max_order = max(node.capacity - inventory, 0)
                order_qty = min(action_qty, max_order)

                context.cumulative_cost += node.order_cost * order_qty

                if node.lead_time <= 0:
                    inventory += order_qty
                    fulfilled_backlog = min(inventory, backorders)
                    inventory -= fulfilled_backlog
                    backorders -= fulfilled_backlog
                    node_info.inventory_level = inventory - backorders

                else:
                    if len(node_info.pipeline) < node.lead_time:
                        node_info.pipeline.extend([0] * (node.lead_time - len(node_info.pipeline)))
                    node_info.pipeline[-1] += order_qty

        if state.remaining_time <= 0:
            state.category = StateCategory.FINAL
        else:
            state.category = StateCategory.AWAIT_EVENT


    def write_features(self, state: SupplyChainState, features: Features) -> None:
            
            # Iterate through nodes to extract data from the state
            for node_static, node_dynamic in zip(self.nodes, state.node_infos):

                inventory = max(0, node_dynamic.inventory_level)
                backlog = max(0, -node_dynamic.inventory_level)
                
                features.append(inventory / node_static.capacity) # Normalize inventory by capacity
                features.append(backlog / node_static.capacity)
                features.append(sum(node_dynamic.pipeline) / node_static.capacity)
                
            features.append(state.remaining_time / self.initial_horizon)


    def write_action_validity(self, state: SupplyChainState, valid: NDArray[np.bool_]) -> None:

        for node_static, node_dynamic in zip(self.nodes, state.node_infos):

            current_inv = max(0, node_dynamic.inventory_level)
            max_order = max(0, node_static.capacity - current_inv)

            for action_qty in range(self.num_actions):
                valid[action_qty] = (action_qty <= max_order)


# Simulation
# ------------

def simulate_episode(mdp: SupplyChainMDP, policy: List[BasePolicy], *, seed: int = 42) -> None:

    context = TrajectoryContext(rng=np.random.default_rng(seed))
    state = mdp.get_initial_state(context)

    step = 0

    print("=" * 80)
    print("SIMULATION STARTING...")
    print(f"Initial state: {state}")
    print("-" * 80)

    while state.category != StateCategory.FINAL:

        if state.category == StateCategory.AWAIT_EVENT:
            mdp.modify_state_with_event(state, context)
            print(f"  State after event: {state}")

        elif state.category == StateCategory.AWAIT_ACTION:

            flat_action = None
            
            if isinstance(policy, list):
                actions_list = [p.get_action(ni) for p, ni in zip(policy, state.node_infos)]
                flat_action = encode_action(actions_list, mdp.action_dims)
            else:
                flat_action = policy.get_action(state)       

            mdp.modify_state_with_action(state, context, flat_action)

            print(f"  Action taken: {flat_action})")
            print(f"  State after action: {state}\n")
            step += 1

        else:
            raise RuntimeError(f"Unexpected state category: {state.category}")

    print("-" * 80)
    print(f"Episode finished: {step} steps, total cost: {context.cumulative_cost:.2f}")





def main() -> None:

    # Initialize MDP and Policy
    # -----------------------

    node_1 = Node(
        id=1,
        name="Node_1",
        capacity=20,
        node_type="Plant",
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
        node_type="Warehouse",
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
        node_type="Retailer",
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

    state = mdp.get_initial_state(TrajectoryContext(rng=np.random.default_rng(1)))

    nodes = mdp.nodes

    connections = []
    for node in nodes:
        for upstream_id in node.upstream_ids:
            connections.append((next(n for n in nodes if n.id == upstream_id), node))

    state_by_id = {node.id: node_info for node, node_info in zip(nodes, state.node_infos)}
    create_graph_window(nodes, connections, state_by_id)


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

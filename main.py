from __future__ import annotations
from dataclasses import dataclass
from typing import List
from custom_types import SupplyChainState, NodeInfo

import numpy as np

from dynaplex.modelling import StateCategory, TrajectoryContext, HorizonType, discover_num_features, Features
from node import Node
from policy import BaseStockPolicy
from graph import create_graph_window
from PPO import train_PPO
from numpy.typing import NDArray


# Supply Chain MDP
# -----------------


@dataclass(slots=True)
class SupplyChainMDP:
    NodeInfo: List[Node]
    initial_horizon: int
    horizon_type: HorizonType
    num_features: int
    num_actions: int

    def __init__(self, NodeInfo: List[Node], initial_horizon: int):
        self.NodeInfo = NodeInfo
        self.initial_horizon = initial_horizon
        self.horizon_type = HorizonType.FINITE
        # The agent chooses an order quantity between 0 and max capacity
        self.num_actions = max(node.capacity for node in NodeInfo) + 1
        
        # Discover features by probing the state
        context = TrajectoryContext(rng=np.random.default_rng(0))
        self.num_features = discover_num_features(self)

    def get_initial_state(self, context: TrajectoryContext) -> SupplyChainState:
        node_infos = []
        for node in self.NodeInfo:
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
        Process exogenous event
        receive arrivals + realize demand.
        Moves state from AWAIT_EVENT -> AWAIT_ACTION.
        """

        # Sanity checks
        assert state.category == StateCategory.AWAIT_EVENT, "Not expecting an event right now."
        assert state.remaining_time > 0, "Simulation already finished."

        for node, node_info in zip(self.NodeInfo, state.node_infos):

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

    # -------------------------------------------------------------------------
    # ACTION TRANSITION
    # -------------------------------------------------------------------------

    def modify_state_with_action(self, state: SupplyChainState, context: TrajectoryContext, actions: List[int]) -> None:
        """
        Apply actions (order quantities).
        Moves state from AWAIT_ACTION -> AWAIT_EVENT or FINAL.
        """

        assert state.category == StateCategory.AWAIT_ACTION, "Not expecting an action right now."
        assert state.remaining_time > 0, "Simulation already finished."
        assert len(actions) == len(self.NodeInfo), "Action dimension mismatch."

        for node, node_info, action in zip(self.NodeInfo, state.node_infos, actions):

            # Calculate inventory and backorders
            backorders: int = max(0, -node_info.inventory_level)
            inventory: int = max(0, node_info.inventory_level)

            if action > 0:
                max_order = max(node.capacity - inventory, 0)
                order_qty = min(action, max_order)

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


    # ---------------------------------------------------------
    # FEATURES & ACTION VALIDITY
    # ---------------------------------------------------------

    def write_features(self, state: SupplyChainState, features: Features) -> None:
            """
            DynaPlex calls this with (state, features). 
            We must extract node info from the state ourselves.
            """
            # Iterate through nodes to extract data from the state
            for node_static, node_dynamic in zip(self.NodeInfo, state.node_infos):
                inventory = max(0, node_dynamic.inventory_level)
                backlog = max(0, -node_dynamic.inventory_level)
                
                # Use append to add to the feature vector
                features.append(inventory / node_static.capacity)
                features.append(backlog / node_static.capacity)
                features.append(sum(node_dynamic.pipeline) / node_static.capacity)
                
            # Optional: add global features like remaining time
            features.append(state.remaining_time / self.initial_horizon)


    def write_action_validity(self, state: SupplyChainState, valid: NDArray[np.bool_]) -> None:
            """
            DynaPlex calls this with (state, valid).
            'valid' is a boolean array of size self.num_actions.
            """
            # 1. Get the static config and dynamic state for the node
            # (Assuming single-node for now based on your main script)
            node_static = self.NodeInfo[0]
            node_dynamic = state.node_infos[0]

            # 2. Determine physical constraints
            # Example: can't order more than (Capacity - Current Inventory)
            current_inv = max(0, node_dynamic.inventory_level)
            max_order = max(0, node_static.capacity - current_inv)

            # 3. Populate the 'valid' mask
            # Note: action index corresponds to order quantity
            for action_qty in range(self.num_actions):
                valid[action_qty] = (action_qty <= max_order)


# Simulation
# ------------

def simulate_episode(mdp: SupplyChainMDP, policy: BaseStockPolicy, *, seed: int = 42) -> None:

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
            
            # Only one node for now
            node_info = state.node_infos[0]
            action = policy.get_action(state=node_info)

            mdp.modify_state_with_action(state, context, [action])

            print(f"Step {step}: ACTION {action} -> State after action: {state}")
            step += 1

        else:
            raise RuntimeError(f"Unexpected state category: {state.category}")

    print("-" * 80)
    print(f"Episode finished: {step} steps, total cost: {context.cumulative_cost:.2f}")





def main() -> None:

    # Initialize MDP and Policy
    # -----------------------

    node = Node(
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

    mdp = SupplyChainMDP(
        NodeInfo=[node],   
        initial_horizon=15,
    )

    policy = BaseStockPolicy(node=node, target_inventory=15, safety_stock=5, price_per_unit=25.0)

    # Run baseline simulation with the initial policy
    # ------------------------------------------------

    simulate_episode(mdp, policy, seed=42)

    # Initialize PPO Trainer and Train Policy
    # ------------------------------------------------

    number_iterations = 50
    trained_policy = train_PPO(mdp, number_iterations=number_iterations)

    # Run simulation with the trained policy
    # ------------------------------------------------

    print("Simulating episode with trained PPO policy...")
    simulate_episode(mdp, trained_policy, seed=54)

    # Generate Visualization
    # ------------------------------------------------

    # final_state = mdp.get_initial_state(TrajectoryContext(rng=np.random.default_rng(1)))

    # nodes = mdp.NodeInfo
    # connections = []  # no edges for single node
    # state_by_id = {node.id: node_info for node, node_info in zip(nodes, final_state.node_infos)}

    # create_graph_window(nodes, connections, state_by_id)


if __name__ == "__main__":
    main()

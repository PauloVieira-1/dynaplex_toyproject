from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from dynaplex.modelling import StateCategory, TrajectoryContext
from node import Node
from policy import BaseStockPolicy
from graph import create_graph_window

from PPO import train_PPO


def simulate_episode(mdp: Node, policy: BaseStockPolicy, *, seed: int = 42) -> None:
    
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
            action = policy.get_action(state=state)
            mdp.modify_state_with_action(state, context, action)
            print(f"Step {step}: ACTION {action} -> State after action: {state}")
            step += 1
        else:
            raise RuntimeError(f"Unexpected state category: {state.category}")

    print("-" * 80)
    print(f"Episode finished: {step} steps, total cost: {context.cumulative_cost:.2f}")


def main() -> None:

    # Initialize MDP and Policy
    # -----------------------
    
    mdp = Node(
        id=1,
        name="Plant-1",
        capacity=20,
        node_type="Node_1",
        holding_cost=1.0,
        backlog_cost=5.0,
        order_cost=2.0,
        lead_time=2,
        upstream_ids=[],
        downstream_ids=[],
        initial_horizon=15,
    )

    policy = BaseStockPolicy(node=mdp, target_inventory=15, safety_stock=5, price_per_unit=25.0)

    # Run baseline simulation with the initial policy
    # ------------------------------------------------

    simulate_episode(mdp, policy, seed=42)

    context = TrajectoryContext(rng=np.random.default_rng(42))
    state = mdp.get_initial_state(context)

    # Initialize PPO Trainer and Train Policy
    # ------------------------------------------------

    number_iterations = 50
    trained_policy = train_PPO(mdp, number_iterations=number_iterations)


    # Run simulation with the trained policy
    # ------------------------------------------------

    print("Simulating episode with trained PPO policy...")
    simulate_episode(mdp, trained_policy, seed=42)

    # Generate Visualization
    # ------------------------------------------------

    nodes = [mdp]
    connections = []  # no edges for single node
    state_by_id = {mdp.id: state}

    # create_graph_window(nodes, connections, state_by_id)



if __name__ == "__main__":
    main()

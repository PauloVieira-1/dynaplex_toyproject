from __future__ import annotations
from typing import List
from custom_types import SupplyChainState
from custom_types.custom_types import ReorderAction
import copy
import numpy as np
from dynaplex.modelling import StateCategory, TrajectoryContext
from models.attention_PPO import train_attention
from models.PPO import train_PPO
from evaluation.record import EpisodeRecorder
from evaluation.plots import plot_results
from evaluation.plot_conversion import plot_training_comparison
from helper_functions import *
from assembly_tree import AssemblyTree
from mdp_assembly import SupplyChainMDP
from helper_functions import get_max_simulation_iterations, get_attention_training_episodes, get_ppo_training_timesteps


# Simulation
# ------------

def simulate_episode(mdp: SupplyChainMDP, policy, seed: int = 50, name: str = "episode", recorder: EpisodeRecorder = None,
    initial_state: SupplyChainState = None,
    max_steps: int = None
) -> None:
    
    context = TrajectoryContext(rng=np.random.default_rng(seed))
    state = initial_state if initial_state is not None else mdp.get_initial_state(context)

    if max_steps is None:
        max_steps = mdp.initial_horizon

    if recorder is not None:                                                    
        recorder.initialise([n.name for n in mdp.nodes])        


    print("=" * 80)
    print(f'Simulating episode: {name}')
    print("-" * 80)


    steps = 0

    while state.category != StateCategory.FINAL and steps < max_steps:

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
                if hasattr(policy, "get_action"):
                    action = policy.get_action(state)
                elif callable(policy):
                    action = policy(state)
                else:
                    raise TypeError("Policy must be callable or have get_action()")

            # ------------------------------------------------------------------------

            if recorder is not None:
                recorder.record_action(current_node_idx, action)

            print(f"         Decision for {current_node_name} (Index {current_node_idx}): Order {action}")

            mdp.modify_state_with_action(state, context, action)

        else:
            raise RuntimeError(f"Unexpected state category: {state.category}")

        steps += 1

    if steps >= max_steps:
        print(f"Episode truncated at day {state.day} after reaching max_steps={max_steps}")

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
        initial_horizon=get_max_simulation_iterations(),
    )

    recorder = EpisodeRecorder("results/random.csv")
    simulate_episode(mdp, policy_list, seed=50, name="Random", recorder=recorder, max_steps=get_max_simulation_iterations())



    # Run baseline simulation with the initial policy
    # ------------------------------------------------    

    # tree = AssemblyTree("config/chain_1.json")
    # tree.create_tree_from_json()
    
    # policy_list_2 = tree.get_policy_list()
    # node_list_2 = tree.get_assembly_tree()

    # mdp_2 = SupplyChainMDP(
    #     nodes = node_list_2,
    #     initial_horizon=get_max_simulation_iterations(),
    # )

    # recorder = EpisodeRecorder("results/base_stock.csv")
    # simulate_episode(mdp_2, policy_list_2, seed=50, name="Base Stock", recorder=recorder, max_steps=get_max_simulation_iterations())


    # Run simulation with the trained PPO policy
    # ------------------------------------------------

    number_iterations = get_ppo_training_timesteps()
    trained_policy, steps_to_train_ppo, time_to_train_ppo = train_PPO(mdp, load_policy=False, total_timesteps=number_iterations)

    print(f"Time taken to train PPO: {time_to_train_ppo}")
    print(f"Steps taken to train PPO: {steps_to_train_ppo}")

    print("Simulating episode with trained PPO policy...")

    recorder = EpisodeRecorder("results/PPO_trained.csv")
    simulate_episode(mdp, trained_policy, seed=50, name="PPO", recorder=recorder, max_steps=get_max_simulation_iterations())


    # Run simulation with trained Attention policy
    # ------------------------------------------------

    reorder_actions = [
        ReorderAction(order_quantity=q)
        for q in range(mdp.num_actions) 
    ]

    max_demand = 6

    trained_policy, steps_to_train_attention, time_to_train_attention = train_attention(
        mdp,
        number_iterations=get_attention_training_episodes(),
        max_steps=get_max_simulation_iterations(),
        reorder_actions=reorder_actions,
        max_demand=max_demand, 
        node_infos=[copy.deepcopy(node_info) for node_info in mdp.get_initial_state(
            TrajectoryContext(rng=np.random.default_rng(50))
        ).node_infos]
    )

    print(f"Time taken to train Attention: {time_to_train_attention}")
    print(f"Steps taken to train Attention: {steps_to_train_attention}")

    print("Simulating episode with trained Attention policy...")

    recorder = EpisodeRecorder("results/attention_trained.csv")
    initial_ctx = TrajectoryContext(rng=np.random.default_rng(50))

    initial_state = SupplyChainState(
        node_infos=[copy.deepcopy(n) for n in mdp.get_initial_state(initial_ctx).node_infos],
        remaining_time=get_max_simulation_iterations(),
        day=0,
        category=StateCategory.AWAIT_ACTION,
        current_node_index=0,
        pending_orders=[0 for _ in mdp.nodes],
    )

    simulate_episode(
        mdp,
        trained_policy,
        seed=50,
        name="Attention", 
        recorder=recorder,
        initial_state=initial_state,
        max_steps=get_max_simulation_iterations()
    )


    # Generate plots of results
    # ------------------------------------------------

    plot_results(
        {
            "Random": "results/random.csv",
            "Base Stock": "results/base_stock.csv",
            "PPO": "results/PPO_trained.csv",
            "Attention": "results/attention_trained.csv",
        }
    )

    plot_training_comparison(
        csv_files={
            "Random":     "results/random.csv",
            "Base Stock": "results/base_stock.csv",
            "PPO":        "results/PPO_trained.csv",
            "Attention":  "results/attention_trained.csv",
        },
        training_stats={
            "PPO":       (steps_to_train_ppo, time_to_train_ppo),
            "Attention": (steps_to_train_attention, time_to_train_attention),
        },
        save_dir="results/",
    )



if __name__ == "__main__":
    main()
from custom_types.custom_types import ReorderAction, SupplyChainState, NodeInfo, GraphAction
import copy
from mdp_assembly import SupplyChainMDP  
from dynaplex.modelling import TrajectoryContext, StateCategory
from dynaplex_playgroud.simple_SC_attention_training import _action_to_key
from typing import List
import numpy as np
from dynaplex_playgroud.data.action_base import GlobalState
from dataclasses import dataclass

try:
    from dynaplex_playgroud.algorithms.ppo_dynaplex_adapter import (
        AttentionPPOConfig,
        train_attention_ppo,
        select_action_attention,
    )
    USE_DYNAPLEX_ADAPTER = False # use false 
except Exception:
    from dynaplex_playgroud.algorithms.ppo_attention import (
        AttentionPPOConfig,
        train_attention_ppo,
        select_action_attention,
    )
    USE_DYNAPLEX_ADAPTER = True


# ----------------------------------------
# Allows the attention to see 5 pipeline slots at once 
# Extra orders are ignored
@dataclass(slots=True)
class NodeFeature(GlobalState):
    inventory: float
    backlog: float
    pending: float
    pl_0: float
    pl_1: float
    pl_2: float
    pl_3: float
    pl_4: float

# ----------------------------------------


# ----------------------------------------

# The global state is different than for the other models
@dataclass(slots=True)
class SCGlobalState(GlobalState):
    current_node_index: float
    remaining_time: float
    category: StateCategory
    nodes: list[NodeFeature]

# Other modesl:

@dataclass(slots=True)
class SupplyChainState:
    
    node_infos: List[NodeInfo]
    day: int
    category: StateCategory
    current_node_index: int
    remaining_time: int
    pending_orders: List[int]  # This can grow up to len(mdp.nodes)

# ----------------------------------------




MAX_PIPELINE_LENGTH = 5

@dataclass(slots=True)
class ActionSet:
    global_state: SCGlobalState   
    actions: list[GraphAction]



def _node_to_feature(node_static, node_dynamic, pending: float) -> NodeFeature: 

    """
    Convert a single node's static + dynamic info into a single NodeFeature object.

    """
    
    inventory = max(0, node_dynamic.inventory_level) / node_static.capacity
    backlog = max(0, -node_dynamic.inventory_level) / node_static.capacity

    pipeline_slots = []
    for s in range(MAX_PIPELINE_LENGTH):
        val = node_dynamic.pipeline[s] if s < len(node_dynamic.pipeline) else 0
        pipeline_slots.append(val / node_static.capacity)

    # pending = pending / node_static.capacity
    return NodeFeature(
        inventory=inventory,
        backlog=backlog,
        pending=pending / node_static.capacity,  
        pl_0=pipeline_slots[0],
        pl_1=pipeline_slots[1],
        pl_2=pipeline_slots[2],
        pl_3=pipeline_slots[3],
        pl_4=pipeline_slots[4],
    )




def build_action_set(state: SupplyChainState, mdp: SupplyChainMDP) -> ActionSet:

    """
    Create an action set with a global state node and action nodes.

    Respoonible for encoding V and K for the Transformer.
    
    """

    # Q (Query) comes from the global state ??
    global_state = SCGlobalState(
        current_node_index=state.current_node_index / len(mdp.nodes),
        remaining_time=state.remaining_time / mdp.initial_horizon,
        category=state.category,
        nodes=[
            _node_to_feature(node_static, node_dynamic, pending)
            for node_static, node_dynamic, pending in zip(mdp.nodes, state.node_infos, state.pending_orders)
        ],
    )

    current_idx = state.current_node_index
    node_static = mdp.nodes[current_idx]
    node_dynamic = state.node_infos[current_idx]

    current_inventory = max(0, node_dynamic.inventory_level)
    max_order = max(0, node_static.capacity - current_inventory)

    # K (Key) comes from the action nodes??
    # V is not explicit but is computed inside the trasformer ??
    graph_actions = [GraphAction(order_quantity=q) for q in range(max_order + 1)]

    if not graph_actions:
        graph_actions = [GraphAction(order_quantity=0)]

    return ActionSet(global_state=global_state, actions=graph_actions)


def evaluate_policy_multinode(
    name: str,
    mdp: SupplyChainMDP,
    action_selector,
    node_infos,         
    num_seeds: int = 20,
    num_steps: int = 50,
):
    costs = []
    for seed in range(num_seeds):
        rng = np.random.default_rng(seed)

        state = SupplyChainState(
            node_infos=copy.deepcopy(node_infos),
            remaining_time=num_steps,
            day=0,
            category=StateCategory.AWAIT_ACTION,
            current_node_index=0,
            pending_orders=[0 for _ in mdp.nodes],
        )

        total_cost = 0.0
        for _ in range(num_steps):
            if state.category == StateCategory.FINAL:
                break

            act = action_selector(state)

            # WHy did you choose to do one step costs instead of cumulative costs?
            ctx = TrajectoryContext(rng=rng, cumulative_cost=0.0, time_elapsed=0)
            mdp.modify_state_with_action(state, ctx, act)

            if state.category == StateCategory.AWAIT_EVENT:
                mdp.modify_state_with_event(state, ctx)

            total_cost += ctx.cumulative_cost

        costs.append(total_cost)

    avg = np.mean(costs)
    lo, hi = np.min(costs), np.max(costs)
    
    print(f"{name:30s} avg_cost={avg:7.2f}  min={lo:.2f}  max={hi:.2f}  ({num_seeds} seeds)")



# ---------------------------------------------------------------------------------------------------------------

def train_attention(mdp: SupplyChainMDP, number_iterations: int, max_steps: int,
                         reorder_actions: List[ReorderAction], max_demand: int, node_infos: List[NodeInfo]):

    config = AttentionPPOConfig(
        num_episodes=number_iterations,
        num_envs=16,
        learning_rate=3e-4,
        entropy_coef=0.03,
        gamma=0.97,
        clip_ratio=0.2,
        d_model=64,
        nhead=4,
        num_layers=2,
        max_steps_per_episode=max_steps,
        seed=123,
        log_every=10,
        ppo_epochs=4,
        minibatch_size=64,
        gae_lambda=0.95,
    )

    
    def _make_step(mdp: SupplyChainMDP, reorder_actions: list[ReorderAction], max_demand: int):

        def step_env(state, action_key: int, rng):


            # Deep copy was not working so im doing a shallow copy
            next_state = SupplyChainState(
                node_infos=[
                    NodeInfo(
                        inventory_level=n.inventory_level,
                        pipeline=list(n.pipeline),
                    )
                    for n in state.node_infos
                ],
                remaining_time=state.remaining_time,
                day=state.day,
                category=state.category,
                current_node_index=state.current_node_index,
                pending_orders=list(state.pending_orders),
            )

            ctx = TrajectoryContext(rng=rng, cumulative_cost=0.0, time_elapsed=0)

            mdp.modify_state_with_action(next_state, ctx, action_key)

            # If statement to check if the next state is an event but not really necessary
            if next_state.category == StateCategory.AWAIT_EVENT:
                mdp.modify_state_with_event(next_state, ctx)

            # Not infinite horizon, done when we reach the final state (in other file I have max time steps)
            done = next_state.category == StateCategory.FINAL
            return next_state, float(ctx.cumulative_cost), done

        return step_env

    def make_state(rng=None): # for now because I'm not using rng

        return SupplyChainState(
            node_infos=copy.deepcopy(node_infos),
            remaining_time=max_steps,
            day=0,
            category=StateCategory.AWAIT_ACTION,
            current_node_index=0,
            pending_orders=[0 for _ in mdp.nodes],
        )

    def make_action_set(state):
        return build_action_set(state, mdp)
    
    
    print("\n---Training with Transformer Cross-Attention PPO ---")

    step_env = _make_step(mdp, reorder_actions, max_demand)

    policy, seq_builder, _ = train_attention_ppo(
        make_state=make_state,
        make_action_set=make_action_set,
        step_env=step_env,
        action_to_key=_action_to_key,
        config=config,
        mdp=mdp,  
    )

    def learned_selector(state: SupplyChainState) -> int:
        return select_action_attention(
            trained_policy=policy,
            seq_builder=seq_builder,
            state=state,
            make_action_set=make_action_set,
            action_to_key=_action_to_key,
        )

    evaluate_policy_multinode("Attention PPO", mdp,learned_selector, node_infos, max_steps)

    return learned_selector
"""
Train Simple Supply Chain MDP using pure Transformer self-data PPO.

This replaces the previous graph-based (HANConv) training approach with the
generic Transformer cross-data PPO loop.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from dynaplex.modelling import TrajectoryContext, StateCategory
from dynaplex_playgroud.mdps.simple_SC import (
    SupplyChainMDP,
    State,
    ReorderAction,
    build_action_set,
    one_step_cost,
    GreedyPolicy,
    OrderUpToPolicy,
)
# Try to use dynaplex PPO trainer adapter if available, otherwise fallback
# to the local algorithms.ppo_attention implementation.
try:
    from dynaplex_playgroud.algorithms.ppo_dynaplex_adapter import (
        AttentionPPOConfig,
        train_attention_ppo,
        select_action_attention,
        _seq_to_action_probs,
    )
    USE_DYNAPLEX_ADAPTER = True
except Exception:
    from dynaplex_playgroud.algorithms.ppo_attention import (
        AttentionPPOConfig,
        train_attention_ppo,
        select_action_attention,
        _seq_to_action_probs,
    )
    USE_DYNAPLEX_ADAPTER = False


# ============================================================================
# Action encoding
# ============================================================================

def _action_to_key(action_feats: torch.Tensor) -> int:
    """Recover raw order_quantity from the action feature vector.

    ``GraphAction`` has a single int field ``order_quantity`` which is
    stored as-is (not normalised) in the token features.
    """
    return int(round(action_feats[0].item()))


# ============================================================================
# Environment step
# ============================================================================

def _make_step_fn(mdp: SupplyChainMDP, reorder_actions: list[ReorderAction], max_demand: int):
    """Build the environment step function for Transformer PPO.

    The supply chain is infinite-horizon with an event→action cycle.
    After applying the action we immediately generate the next event
    (next demand arrival) so the agent sees a continuous stream of decisions.
    """
    def step_env(state: State, action_key: int, rng):
        next_state = copy.deepcopy(state)
        ctx = TrajectoryContext(rng=rng, cumulative_cost=0.0, time_elapsed=0)

        # Apply action (order units, fulfil demand, incur costs)
        mdp.modify_state_with_action(next_state, ctx, action_key)

        # Generate next event (sample next demand)
        mdp.modify_state_with_event(next_state, ctx)

        # Infinite horizon — never done
        done = False
        return next_state, float(ctx.cumulative_cost), done

    return step_env


# ============================================================================
# Reward shaping
# ============================================================================

def _reward_shaping(state, next_state, action_key: int, base_reward: float) -> float:
    # base_reward = -cost from the generic PPO loop.
    # Give a small bonus for keeping inventory near a sensible level
    # (not too high → holding cost, not too low → stockout risk).
    reward = base_reward

    # Penalise ordering nothing when inventory is very low and demand is coming
    if action_key == 0 and state.inventory_level <= 1 and state.incoming_order > 0:
        reward -= 1.0

    return reward


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_policy(name: str, mdp: SupplyChainMDP, action_selector, num_seeds: int = 20, num_steps: int = 50):
    costs = []
    for seed in range(num_seeds):
        rng = np.random.default_rng(seed)
        state = State(
            inventory_level=mdp.max_order_quantity,
            incoming_order=0,
            category=StateCategory.AWAIT_EVENT,
        )
        total_cost = 0.0
        for _ in range(num_steps):
            # Event: sample demand
            state.incoming_order = int(rng.choice(mdp.demands, p=mdp.demand_probs))
            state.category = StateCategory.AWAIT_ACTION
            # Action
            act = action_selector(state)
            step_cost = one_step_cost(
                inventory_level=state.inventory_level,
                incoming_order=state.incoming_order,
                action=act,
                order_cost_per_unit=mdp.order_cost_per_unit,
                holding_cost_per_unit=mdp.holding_cost_per_unit,
                stockout_cost=mdp.stockout_cost,
            )
            total_cost += step_cost
            ctx = TrajectoryContext(rng=rng, cumulative_cost=0.0, time_elapsed=0)
            mdp.modify_state_with_action(state, ctx, act)
        costs.append(total_cost)
    avg = np.mean(costs)
    lo, hi = np.min(costs), np.max(costs)
    print(f"{name:30s} avg_cost={avg:7.2f}  min={lo:.2f}  max={hi:.2f}  ({num_seeds} seeds)")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 60)
    print("Simple Supply Chain — Transformer Attention PPO")
    print("=" * 60)
 
    # ------------------------------------------------------------------
    # Scenario
    # ------------------------------------------------------------------
    scenario = {
        "max_order_quantity": 8,
        "order_cost_per_unit": 3.0,
        "holding_cost_per_unit": 1.2,
        "stockout_cost": 25.0,
        "demands": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64),
        "demand_probs": np.array(
            [0.02, 0.03, 0.05, 0.08, 0.12, 0.16, 0.20, 0.20, 0.14],
            dtype=np.float64,
        ),
    }

    mdp = SupplyChainMDP(
        max_order_quantity=scenario["max_order_quantity"],
        order_cost_per_unit=scenario["order_cost_per_unit"],
        holding_cost_per_unit=scenario["holding_cost_per_unit"],
        stockout_cost=scenario["stockout_cost"],
        demand_probs=scenario["demand_probs"],
        demands=scenario["demands"],
    )

    reorder_actions = [
        ReorderAction(order_quantity=q)
        for q in range(mdp.max_order_quantity + 1)
    ]
    max_demand = int(np.max(mdp.demands)) if len(mdp.demands) > 0 else 1

    # ------------------------------------------------------------------
    # Heuristic baselines
    # ------------------------------------------------------------------
    print("\n=== Heuristic Baselines ===")
    greedy = GreedyPolicy(mdp=mdp, reorder_point=4, target_level=8)
    order_up_to = OrderUpToPolicy(mdp=mdp, target_level=7)
    evaluate_policy("GreedyPolicy", mdp, greedy.get_action)
    evaluate_policy("OrderUpToPolicy", mdp, order_up_to.get_action)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    config = AttentionPPOConfig(
        num_episodes=200,
        num_envs=16,
        learning_rate=3e-4,
        entropy_coef=0.03,
        gamma=0.97,
        clip_ratio=0.2,
        d_model=64,
        nhead=4,
        num_layers=2,
        max_steps_per_episode=50,
        seed=123,
        log_every=10,
        ppo_epochs=4,
        minibatch_size=64,
        gae_lambda=0.95,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    step_env = _make_step_fn(mdp, reorder_actions, max_demand)

    def make_state(rng):
        # Sample initial demand so the agent starts in AWAIT_ACTION
        demand = int(rng.choice(mdp.demands, p=mdp.demand_probs))
        return State(
            inventory_level=mdp.max_order_quantity,
            incoming_order=demand,
            category=StateCategory.AWAIT_ACTION,
        )

    def make_action_set(state):
        return build_action_set(state, mdp, reorder_actions, max_demand)

    print("\n=== Training with Transformer Cross-Attention PPO ===")
    if USE_DYNAPLEX_ADAPTER:
        trained_policy, seq_builder, _ = train_attention_ppo(
            make_state=make_state,
            make_action_set=make_action_set,
            step_env=step_env,
            action_to_key=_action_to_key,
            config=config,
            reward_shaping=_reward_shaping,
            mdp=mdp,
        )
    else:
        policy_net, seq_builder, _ = train_attention_ppo(
            make_state=make_state,
            make_action_set=make_action_set,
            step_env=step_env,
            action_to_key=_action_to_key,
            config=config,
            reward_shaping=_reward_shaping,
        )

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    print("\n=== Debug: Sample Policy Output ===")
    test_state = make_state(np.random.default_rng(999))
    seq = seq_builder.build(make_action_set(test_state))

    if USE_DYNAPLEX_ADAPTER:
        with torch.no_grad():
            probs = _seq_to_action_probs(trained_policy, seq)
            print(f"Action probabilities: {[f'{p:.3f}' for p in probs.tolist()]}")
            print(f"Chosen action (order qty): {_action_to_key(torch.tensor([probs.argmax().item()]))}")
    else:
        policy_net.eval()
        with torch.no_grad():
            probs = _seq_to_action_probs(policy_net, seq)
            print(f"Action probabilities: {[f'{p:.3f}' for p in probs.tolist()]}")
            print(f"Chosen action (order qty): {_action_to_key(torch.tensor([probs.argmax().item()]))}")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print("\n=== Evaluation ===")

    def learned_selector(state: State) -> int:
        if USE_DYNAPLEX_ADAPTER:
            return select_action_attention(
                trained_policy,
                seq_builder,
                state,
                make_action_set=make_action_set,
                action_to_key=_action_to_key,
                deterministic=True,
            )
        else:
            return select_action_attention(
                policy_net=policy_net,
                seq_builder=seq_builder,
                state=state,
                make_action_set=make_action_set,
                action_to_key=_action_to_key,
            )

    evaluate_policy("GreedyPolicy", mdp, greedy.get_action)
    evaluate_policy("OrderUpToPolicy", mdp, order_up_to.get_action)
    evaluate_policy("Attention PPO", mdp, learned_selector)


if __name__ == "__main__":
    main()

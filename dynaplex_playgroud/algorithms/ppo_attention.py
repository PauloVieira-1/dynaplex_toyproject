"""PPO with Transformer self-data policy (no GNN / graph structure).

This is a clean re-implementation of the PPO loop that uses SequenceBuilder +
TransformerPolicyNet/ValueNet instead of ActionGraphGenerator + HANConv.

The training loop is *generic* — problem-specific logic (action mapping,
reward shaping, environment stepping) is injected via callbacks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple
import copy

import numpy as np
import torch
from torch.optim import Adam

from data.sequence_model import SequenceBuilder
from networks.transformer_networks import (
    TransformerPolicyNet,
    TransformerValueNet,
    build_transformer_policy_net,
    build_transformer_value_net,
)


@dataclass(slots=True)
class AttentionPPOConfig:
    """Config for Transformer-based PPO."""
    num_episodes: int = 2000
    num_envs: int = 16
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    gamma: float = 0.99
    log_every: int = 50
    seed: int = 123
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    max_steps_per_episode: int = 100
    ppo_epochs: int = 4
    minibatch_size: int = 64
    gae_lambda: float = 0.95


# ============================================================================
# Action feature extraction helpers
# ============================================================================

def _get_action_root_features(seq: Dict[str, Any], action_idx: int) -> torch.Tensor:
    """Return the raw feature vector of the *root* token of action group ``action_idx``.

    When actions are flat (group size 1) this is the same as the old
    ``_get_action_features``.  When actions contain nested children the root
    token is identified via ``is_group_root``.
    """
    group_root_indices = seq["is_group_root"].nonzero(as_tuple=True)[0]
    global_idx = group_root_indices[action_idx].item()
    tname = seq["token_type_names"][global_idx]
    local_idxs = seq["indices_by_type"][tname]
    local_i = local_idxs.index(global_idx)
    return seq["type_features"][tname][local_i]


def _get_all_action_root_features(seq: Dict[str, Any]) -> List[torch.Tensor]:
    """Return list of root-token feature vectors for every action group."""
    n = seq["num_action_groups"]
    return [_get_action_root_features(seq, i) for i in range(n)]


# Backward-compatible aliases
_get_action_features = _get_action_root_features
_get_all_action_features = _get_all_action_root_features


# ============================================================================
# Network forward helpers
# ============================================================================

def _seq_to_action_probs(
    policy_net: TransformerPolicyNet,
    seq: Dict[str, Any],
) -> torch.Tensor:
    """Run the policy net on a sequence dict and return action probabilities."""
    logits = policy_net.forward_logits(
        type_features=seq["type_features"],
        type_ids=seq["type_ids"],
        action_mask=seq["action_mask"],
        indices_by_type=seq["indices_by_type"],
        action_group_ids=seq.get("action_group_ids"),
        num_action_groups=seq.get("num_action_groups"),
        intra_group_pos=seq.get("intra_group_pos"),
    )
    return torch.softmax(logits, dim=0)


def _seq_to_value(
    value_net: TransformerValueNet,
    seq: Dict[str, Any],
) -> torch.Tensor:
    """Run the value net on a sequence dict and return scalar value."""
    return value_net(
        type_features=seq["type_features"],
        type_ids=seq["type_ids"],
        indices_by_type=seq["indices_by_type"],
        action_mask=seq["action_mask"],
    )


# ============================================================================
# Generic PPO training
# ============================================================================

def train_attention_ppo(
    make_state: Callable[[np.random.Generator], Any],
    make_action_set: Callable[[Any], Any],
    step_env: Callable[[Any, Any, np.random.Generator], Tuple[Any, float, bool]],
    action_to_key: Callable[[torch.Tensor], Any],
    config: AttentionPPOConfig,
    reward_shaping: Callable[[Any, Any, Any, float], float] | None = None,
) -> Tuple[TransformerPolicyNet, SequenceBuilder]:
    """Train a Transformer-based PPO agent (generic).

    Args:
        make_state: Creates a fresh initial state from an RNG.
        make_action_set: Builds an ActionSet from a state.
        step_env: ``(state, action_key, rng) -> (next_state, cost, done)``
                  Applies the action to a **copy** of state. Returns the new
                  state, the immediate non-negative cost, and whether the
                  episode is done.
        action_to_key: Converts raw action feature tensor to a hashable key
                       used to identify / match actions (e.g. ``(dx, dy)``
                       for order picking, ``int(bin_index)`` for bin packing).
        config: Training configuration.
        reward_shaping: Optional ``(state, next_state, action_key, base_reward) -> shaped_reward``.

    Returns:
        (policy_net, sequence_builder)
    """
    # Build sequence builder and discover types from a sample
    seq_builder = SequenceBuilder(skip_root=True)
    rng_init = np.random.default_rng(config.seed)
    sample_state = make_state(rng_init)
    sample_action_set = make_action_set(sample_state)
    sample_seq = seq_builder.build(sample_action_set)

    raw_dims = seq_builder.raw_dims
    type_names = seq_builder.type_names

    print(f"Sequence builder types: {type_names}")
    print(f"Raw dims: {raw_dims}")
    print(f"Sample sequence length: {sample_seq['seq_len']}")
    print(f"Sample action groups: {sample_seq['num_action_groups']}")

    # Build networks
    policy_net = build_transformer_policy_net(
        raw_dims=raw_dims,
        type_names=type_names,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
    )
    value_net = build_transformer_value_net(
        raw_dims=raw_dims,
        type_names=type_names,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
    )

    optimizer = Adam(
        list(policy_net.parameters()) + list(value_net.parameters()),
        lr=config.learning_rate,
    )

    rng = np.random.default_rng(config.seed)
    episode_rewards: List[float] = []

    for episode in range(config.num_episodes):
        batch_transitions: List[tuple] = []
        batch_total_rewards: List[float] = []

        for env_idx in range(config.num_envs):
            state = make_state(rng)
            traj: List[tuple] = []
            total_reward = 0.0

            for step in range(config.max_steps_per_episode):
                action_set = make_action_set(state)
                seq = seq_builder.build(action_set)

                with torch.no_grad():
                    probs = _seq_to_action_probs(policy_net, seq)
                    sampled_idx = torch.multinomial(probs, 1).item()
                    old_log_prob = torch.log(probs[sampled_idx] + 1e-8).item()
                    value = _seq_to_value(value_net, seq).item()

                # Get action key from root-token features
                action_feats = _get_action_root_features(seq, sampled_idx)
                action_key = action_to_key(action_feats)

                # Step environment
                next_state, cost, done = step_env(state, action_key, rng)
                reward = -cost

                # Optional reward shaping
                if reward_shaping is not None:
                    reward = reward_shaping(state, next_state, action_key, reward)

                total_reward += reward
                traj.append((copy.deepcopy(state), action_key, reward, old_log_prob, value))
                state = next_state
                if done:
                    break

            # GAE(λ) returns and advantages
            returns: List[float] = []
            advantages: List[float] = []
            gae = 0.0
            next_value = 0.0  # bootstrap value (0 if terminal)
            for _, _, r, _, v in reversed(traj):
                delta = r + config.gamma * next_value - v
                gae = delta + config.gamma * config.gae_lambda * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + v)
                next_value = v

            for i, (s, a, r, olp, v) in enumerate(traj):
                batch_transitions.append((s, a, returns[i], olp, advantages[i]))
            batch_total_rewards.append(total_reward)

        if not batch_transitions:
            continue

        # Normalize advantages
        all_adv = np.array([t[4] for t in batch_transitions])
        adv_mean = all_adv.mean()
        adv_std = all_adv.std() + 1e-8

        # PPO update: multiple epochs over shuffled mini-batches
        num_transitions = len(batch_transitions)
        mb_size = min(config.minibatch_size, num_transitions)

        for ppo_epoch in range(config.ppo_epochs):
            indices = np.random.permutation(num_transitions)

            for mb_start in range(0, num_transitions, mb_size):
                mb_indices = indices[mb_start:mb_start + mb_size]

                policy_loss = 0.0
                value_loss = 0.0
                mb_count = 0

                for idx in mb_indices:
                    stored_state, action_key, ret, old_log_prob, advantage = batch_transitions[idx]
                    action_set = make_action_set(stored_state)
                    seq = seq_builder.build(action_set)

                    # Policy forward
                    probs = _seq_to_action_probs(policy_net, seq)

                    # Find action by matching key on root features
                    all_feats = _get_all_action_root_features(seq)
                    found = False
                    for k_action, feats in enumerate(all_feats):
                        if action_to_key(feats) == action_key:
                            new_log_prob = torch.log(probs[k_action] + 1e-8)
                            norm_adv = (advantage - adv_mean) / adv_std

                            ratio = torch.exp(new_log_prob - old_log_prob)
                            surr1 = ratio * norm_adv
                            surr2 = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * norm_adv
                            ppo_loss = -torch.min(surr1, surr2)

                            entropy = -(probs * torch.log(probs + 1e-8)).sum()
                            policy_loss = policy_loss + ppo_loss - config.entropy_coef * entropy
                            found = True
                            break

                    if not found:
                        continue

                    # Value forward
                    value_pred = _seq_to_value(value_net, seq).squeeze()
                    value_loss = value_loss + (value_pred - ret) ** 2
                    mb_count += 1

                if mb_count > 0:
                    optimizer.zero_grad()
                    total_loss = (policy_loss + 0.5 * value_loss) / mb_count
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
                    torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=0.5)
                    optimizer.step()

        episode_rewards.extend(batch_total_rewards)

        if (episode + 1) % config.log_every == 0:
            recent = episode_rewards[-config.log_every * config.num_envs:]
            avg_reward = np.mean(recent)
            print(f"episode={episode + 1:4d} avg_reward={avg_reward:7.2f}")

    # Return the collected episode rewards for plotting/analysis
    return policy_net, seq_builder, episode_rewards


# ============================================================================
# Generic action selector
# ============================================================================

def select_action_attention(
    policy_net: TransformerPolicyNet,
    seq_builder: SequenceBuilder,
    state: Any,
    make_action_set: Callable[[Any], Any],
    action_to_key: Callable[[torch.Tensor], Any],
    deterministic: bool = True,
) -> Any:
    """Select an action using the trained Transformer policy.

    Returns:
        The action key (whatever ``action_to_key`` returns).
    """
    policy_net.eval()
    action_set = make_action_set(state)
    seq = seq_builder.build(action_set)

    with torch.no_grad():
        probs = _seq_to_action_probs(policy_net, seq)

        if deterministic:
            chosen = int(torch.argmax(probs).item())
        else:
            chosen = int(torch.multinomial(probs, 1).item())

        feats = _get_action_root_features(seq, chosen)
        key = action_to_key(feats)

    policy_net.train()
    return key


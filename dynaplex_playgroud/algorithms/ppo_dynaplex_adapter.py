"""
Minimal adapter exposing the old attention-training API while delegating
training to dynaplex's PPOTrainer.

This file is intentionally small and defensive: if the dynaplex trainer
API is not present, importing this module will raise ImportError and the
training scripts will fall back to the original `algorithms.ppo_attention`.

Notes:
- The adapter does not change dynaplex internals (ppo_trainer, gymnasium_adapter).
- It returns (trained_policy, seq_builder, episode_rewards) to be compatible
  with the scripts that expect a Transformer policy and sequence builder.
- The trained_policy returned is whatever `PPOTrainer.train()` returns
  (we expect it to have a `get_action(state, deterministic=True)` method).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from dynaplex_playgroud.data.sequence_model import SequenceBuilder

# We delay import of dynaplex.algorithms.ppo_trainer and the gymnasium adapter
# until train_attention_ppo is called. This avoids import-time failures when
# optional dependencies like `gymnasium` are not installed. The trainer
# itself expects a DynaPlexVectorEnv available under dynaplex.utilities.
def _import_trainer():
    """Import PPOTrainer and ensure DynaPlexVectorEnv is available.

    Returns: (PPOTrainer, PPOTrainerConfig)
    Raises informative ImportError if gymnasium or trainer cannot be imported.
    """
    try:
        # Try direct import first
        from dynaplex.algorithms.ppo_trainer import PPOTrainer, PPOTrainerConfig
        return PPOTrainer, PPOTrainerConfig
    except Exception:
        # Attempt to import gymnasium adapter and inject missing symbol
        try:
            import importlib
            ga = importlib.import_module('dynaplex.utilities.gymnasium_adapter')
        except ModuleNotFoundError as e:
            raise ImportError(
                "dynaplex PPO trainer requires the 'gymnasium' package and the "
                "dynaplex gymnasium adapter. Please install gymnasium (pip install gymnasium) "
                "and ensure dynaplex.utilities.gymnasium_adapter is importable. Original error: "
                + str(e)
            )
        except Exception as e:
            raise ImportError("Could not import dynaplex.utilities.gymnasium_adapter: " + str(e))

        # Inject DynaPlexVectorEnv into dynaplex.utilities namespace if not exported
        import dynaplex.utilities as _dutils
        if not hasattr(_dutils, 'DynaPlexVectorEnv') and hasattr(ga, 'DynaPlexVectorEnv'):
            setattr(_dutils, 'DynaPlexVectorEnv', getattr(ga, 'DynaPlexVectorEnv'))

        # Now import trainer
        try:
            from dynaplex.algorithms.ppo_trainer import PPOTrainer, PPOTrainerConfig
            return PPOTrainer, PPOTrainerConfig
        except Exception as e:
            raise ImportError("Could not import dynaplex.algorithms.ppo_trainer after injecting adapter: " + str(e))


@dataclass
class AttentionPPOConfig:
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


def _map_config(cfg: AttentionPPOConfig) -> dict:
    # Return a kwargs dict suitable for constructing PPOTrainerConfig after
    # the trainer class has been imported. This avoids referencing the type
    # before dynamic imports.
    total_timesteps = int(max(1, cfg.num_episodes) * max(1, cfg.num_envs) * max(1, cfg.max_steps_per_episode))
    num_steps = min(256, max(16, cfg.max_steps_per_episode))

    return dict(
        minibatch_size=max(1, cfg.minibatch_size),
        num_envs=cfg.num_envs,
        max_episode_steps=cfg.max_steps_per_episode,
        total_timesteps=total_timesteps,
        num_steps=num_steps,
        update_epochs=cfg.ppo_epochs,
        anneal_lr=False,
        lr=cfg.learning_rate,
        gamma=cfg.gamma,
        hidden_sizes=(64, 64),
        eps_clip=cfg.clip_ratio,
        vf_coef=0.5,
        ent_coef=cfg.entropy_coef,
        max_grad_norm=0.5,
        gae_lambda=cfg.gae_lambda,
        norm_adv=True,
        clip_vloss=True,
        target_kl=None,
        device="cpu",
        seed=cfg.seed,
        logdir=None,
        test_seed_offset=10000,
        num_eval_episodes=50,
        compile_agent=False,
        save_best=False,
        save_checkpoint=False,
    )


def train_attention_ppo(
    make_state: Callable[[np.random.Generator], Any],
    make_action_set: Callable[[Any], Any],
    step_env: Callable[[Any, Any, np.random.Generator], Tuple[Any, float, bool]],
    action_to_key: Callable[[torch.Tensor], Any],
    config: AttentionPPOConfig,
    reward_shaping: Optional[Callable[[Any, Any, Any, float], float]] = None,
    *,
    mdp: Optional[Any] = None,
) -> Tuple[Any, SequenceBuilder, List[float]]:
    """Train using dynaplex PPOTrainer and return a compatibility tuple.

    The adapter requires the `mdp` argument because dynaplex trainer expects
    an MDP instance to construct environments.
    """
    if mdp is None:
        raise TypeError("train_attention_ppo requires mdp=<mdp_instance> when using dynaplex adapter")

    # Build sequence builder for debugging and compatibility with existing scripts
    seq_builder = SequenceBuilder(skip_root=True)
    rng_init = np.random.default_rng(config.seed)
    sample_state = make_state(rng_init)
    sample_as = make_action_set(sample_state)
    sample_seq = seq_builder.build(sample_as)

    print(f"Sequence builder types: {seq_builder.type_names}")
    print(f"Raw dims: {seq_builder.raw_dims}")
    print(f"Sample sequence length: {sample_seq['seq_len']}")
    print(f"Sample action groups: {sample_seq.get('num_action_groups', 0)}")

    # Import trainer classes now (may raise informative error if gymnasium missing)
    PPOTrainer, PPOTrainerConfig = _import_trainer()
    trainer_cfg_kwargs = _map_config(config)
    trainer_cfg = PPOTrainerConfig(**trainer_cfg_kwargs)
    trainer = PPOTrainer(mdp=mdp, config=trainer_cfg)

    trained_policy = trainer.train()

    # We don't have per-episode rewards to return; keep empty list for compatibility
    return trained_policy, seq_builder, []


def select_action_attention(
    trained_policy: Any,
    seq_builder: SequenceBuilder,
    state: Any,
    make_action_set: Callable[[Any], Any],
    action_to_key: Callable[[torch.Tensor], Any],
    deterministic: bool = True,
) -> Any:
    """Return an action key using the trained dynaplex policy.

    We expect trained_policy to expose .get_action(state, deterministic=True).
    """
    if trained_policy is None:
        raise TypeError("select_action_attention expects trained_policy as first arg")

    return trained_policy.get_action(state, deterministic=deterministic)


def _seq_to_action_probs(trained_policy: Any, seq: Dict[str, Any]) -> torch.Tensor:
    """Best-effort: return a vector of action probabilities for a sequence.

    The dynaplex TrainedPolicy may not expose token-level logits. For debug
    uses we return a uniform distribution sized to the number of action groups
    in `seq` when we can't reconstruct exact logits.
    """
    if trained_policy is None:
        num_actions = seq.get('num_action_groups', 1)
        return torch.ones(num_actions, dtype=torch.float32) / float(num_actions)

    # Try to infer number of actions from seq; otherwise fallback to 1
    num_actions = seq.get('num_action_groups') or seq.get('action_mask').sum().item() if 'action_mask' in seq else 1
    return torch.ones(num_actions, dtype=torch.float32) / float(num_actions)





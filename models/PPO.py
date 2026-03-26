from dynaplex import PPOTrainerConfig, PPOTrainer
from dynaplex.algorithms.ppo_trainer import TrainedPolicy
import torch
import numpy as np
from dynaplex.algorithms.ppo_trainer import StateT
import time
import torch.nn as nn


class StepTrackingPPOTrainer(PPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_steps_taken = 0  
        self.time_taken = 0

    def add_step_to_count(self, steps: int = 1):

        self.total_steps_taken += steps


    def train(self) -> TrainedPolicy[StateT]:
  
        assert self._training_env is not None
        assert self._agent is not None
        assert self._optimizer is not None

        device = torch.device(self.config.device)

        # Derived config
        batch_size = int(self.config.num_envs * self.config.num_steps)
        num_minibatches = max(1, batch_size // self.config.minibatch_size)
        minibatch_size = int(batch_size // num_minibatches)
        num_iterations = self.config.total_timesteps // batch_size

        print(f"Starting training with {self.config.num_envs} parallel environments...")
        print(f"Observation shape: {self._training_env.observation_space.shape}")
        print(f"Action space: {self._training_env.action_space}")
        print(f"Logging to: {self._log_path}")
        print(f"Total timesteps: {self.config.total_timesteps}")
        print(f"Batch size: {batch_size}, Minibatch size: {minibatch_size}")
        print(f"Number of iterations: {num_iterations}")

        # Storage setup
        obs_shape = self._training_env.observation_space.shape
        num_actions = int(self._training_env.action_space.n)

        obs = torch.zeros(
            (self.config.num_steps, self.config.num_envs, *obs_shape)
        ).to(device)
        actions = torch.zeros(
            (self.config.num_steps, self.config.num_envs),
            dtype=torch.long,
        ).to(device)
        logprobs = torch.zeros((self.config.num_steps, self.config.num_envs)).to(device)
        rewards = torch.zeros((self.config.num_steps, self.config.num_envs)).to(device)
        terminations = torch.zeros((self.config.num_steps, self.config.num_envs), dtype=torch.bool).to(device)
        truncations = torch.zeros((self.config.num_steps, self.config.num_envs), dtype=torch.bool).to(device)
        values = torch.zeros((self.config.num_steps, self.config.num_envs)).to(device)
        action_masks = torch.zeros(
            (self.config.num_steps, self.config.num_envs, num_actions),
            dtype=torch.bool,
        ).to(device)

        # Start training
        start_time = time.time()
        next_obs_np, infos = self._training_env.reset(seed=self.config.seed)

        next_obs = torch.from_numpy(next_obs_np).to(device)
        next_terminated = torch.zeros(self.config.num_envs, dtype=torch.bool, device=device)
        next_truncated = torch.zeros(self.config.num_envs, dtype=torch.bool, device=device)
        next_action_mask = torch.from_numpy(infos["action_mask"]).to(device)

        best_mean_reward = float('-inf')

        for iteration in range(1, num_iterations + 1):
            # Anneal learning rate
            if self.config.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                lrnow = frac * self.config.lr
                self._optimizer.param_groups[0]["lr"] = lrnow

            for step in range(self.config.num_steps):
                self.add_step_to_count(self.config.num_envs)  # Track steps

                obs[step] = next_obs
                terminations[step] = next_terminated
                truncations[step] = next_truncated
                action_masks[step] = next_action_mask

                # Get action and value
                with torch.no_grad():
                    action, logprob, _, value = self._agent.get_action_and_value(
                        next_obs, action_mask=next_action_mask
                    )
                    values[step] = value.flatten()

                actions[step] = action
                logprobs[step] = logprob

                # Step the environment
                next_obs_np, reward, env_terminations, env_truncations, infos = self._training_env.step(
                    action.cpu().numpy()
                )
                rewards[step] = torch.tensor(reward, device=device).view(-1)
                next_obs.copy_(torch.from_numpy(next_obs_np))
                next_terminated.copy_(torch.from_numpy(env_terminations))
                next_truncated.copy_(torch.from_numpy(env_truncations))
                next_action_mask.copy_(torch.from_numpy(infos["action_mask"]))

            # Compute advantages
            with torch.no_grad():
                next_value = self._agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(self.config.num_steps)):
                    if t == self.config.num_steps - 1:
                        nextnonterminal = 1.0 - next_terminated.float()
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - terminations[t + 1].float()
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t] + self.config.gamma * nextvalues * nextnonterminal - values[t]
                    )
                    advantages[t] = lastgaelam = delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # Flatten batch
            b_obs = obs.reshape((-1, *obs_shape))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            b_action_masks = action_masks.reshape((-1, num_actions))

            # Policy optimization
            b_inds = np.arange(batch_size)
            clipfracs = []
            for epoch in range(self.config.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self._agent.get_action_and_value(
                        b_obs[mb_inds],
                        b_actions.long()[mb_inds],
                        action_mask=b_action_masks[mb_inds],
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.config.eps_clip).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.config.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.eps_clip, 1 + self.config.eps_clip)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if self.config.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -self.config.eps_clip, self.config.eps_clip)
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef

                    self._optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._agent.parameters(), self.config.max_grad_norm)
                    self._optimizer.step()

                if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                    break

            # Logging
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            mean_test_reward = None
            if iteration % 10 == 0:
                test_rewards = self._evaluate_policy()
                mean_test_reward = np.mean(test_rewards)
                if self._writer is not None:
                    self._writer.add_scalar("test/mean_reward", mean_test_reward, self.total_steps_taken)
                if self.config.save_best and mean_test_reward > best_mean_reward:
                    best_mean_reward = mean_test_reward
                    if self._log_path is not None:
                        torch.save(self._get_state_dict(), self._log_path / "best_policy.pth")
                        print(f"New best policy saved! Mean reward: {best_mean_reward:.2f}")

            # Checkpointing
            if self.config.save_checkpoint and self.total_steps_taken % self.config.checkpoint_interval == 0:
                if self._log_path is not None:
                    checkpoint_path = self._log_path / f"checkpoint_step_{self.total_steps_taken}.pth"
                    torch.save(self._get_state_dict(), checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path}")

            # Writer metrics
            if self._writer is not None:
                self._writer.add_scalar("charts/learning_rate", self._optimizer.param_groups[0]["lr"], self.total_steps_taken)
                self._writer.add_scalar("losses/value_loss", v_loss.item(), self.total_steps_taken)
                self._writer.add_scalar("losses/policy_loss", pg_loss.item(), self.total_steps_taken)
                self._writer.add_scalar("losses/entropy", entropy_loss.item(), self.total_steps_taken)
                self._writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.total_steps_taken)
                self._writer.add_scalar("losses/approx_kl", approx_kl.item(), self.total_steps_taken)
                self._writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.total_steps_taken)
                self._writer.add_scalar("losses/explained_variance", explained_var, self.total_steps_taken)
                sps = int(self.total_steps_taken / (time.time() - start_time))
                self._writer.add_scalar("charts/SPS", sps, self.total_steps_taken)
            else:
                sps = int(self.total_steps_taken / (time.time() - start_time))

            if iteration % 10 == 0:
                msg = f"Iteration {iteration}/{num_iterations} - Step {self.total_steps_taken}/{self.config.total_timesteps} - SPS: {sps}"
                if mean_test_reward is not None:
                    msg += f" - Test Reward: {mean_test_reward:.2f}"
                print(msg)

        self.time_taken = time.time() - start_time
        print("Training complete!")

        if self._log_path is not None:
            final_policy_path = self._log_path / "final_policy.pth"
            torch.save(self._get_state_dict(), final_policy_path)
            print(f"Final policy saved to {final_policy_path}")

        expected_state_type: type[StateT] | None = None

        return TrainedPolicy(
            mdp=self.mdp,  # type: ignore[arg-type]
            agent=self._agent,
            device=self.config.device,
            expected_state_type=expected_state_type,
        )


def train_PPO(mdp, load_policy=False, total_timesteps=500_000):
    config = PPOTrainerConfig(
        seed=42,
        device="cpu",
        hidden_sizes=[256, 256, 128],
        num_steps=2048,
        minibatch_size=512,
        lr=3e-4,
        anneal_lr=True,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        total_timesteps=total_timesteps,
        num_envs=16,
        max_grad_norm=0.5,
        norm_adv=True,
    )

    print(f"Training PPO with config: {config}")
    print("-" * 80)

    ppo_trainer = StepTrackingPPOTrainer(mdp=mdp, config=config)

    if load_policy:
        print("Loading previously trained policy...")
        trained_policy = ppo_trainer.load_trained_policy()

        steps_taken = ppo_trainer.total_steps_taken
        time_taken = ppo_trainer.time_taken

        print("Policy loaded successfully.")
    else:
        print("Training policy...")
        trained_policy = ppo_trainer.train()
        
        steps_taken = ppo_trainer.total_steps_taken
        time_taken = ppo_trainer.time_taken
        
        print("Training completed.")

    print(f"Trained policy: {trained_policy}")
    return trained_policy, steps_taken, time_taken
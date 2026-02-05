from dynaplex import PPOTrainerConfig, PPOTrainer

def train_PPO(mdp, number_iterations=50):

    config = PPOTrainerConfig(
        seed = 42,
        device = "cpu",
        hidden_sizes=[64, 64],
        num_steps= 2 * number_iterations,
        minibatch_size=64, # num_envs * num_steps should be divisible by minibatch_size
        ent_coef=0.01,
        gamma=0.99,
        lr=3e-4,
        total_timesteps=100000, # Total n. interactions across all iterations
        logdir=None,  
        num_envs=8, # Number of parallel environments for data collection (speeds up training)
    )


    print(f"Training PPO with config: {config}")
    print(f"-" * 80)

    ppo_trainer = PPOTrainer(mdp=mdp, config=config)
    load_policy = False

    if load_policy:
        print("Loading previously trained policy...")
        trained_policy = ppo_trainer.load_trained_policy()
        print("Policy loaded successfully!")
    else:
        print("Training policy...")
        trained_policy = ppo_trainer.train()
        print("Training completed!")

    print(f"Trained policy: {trained_policy}")

    return trained_policy




# ======================================================================================
# EXPLAINING THE "NUMBER OF ENVIRONMENTS" COST DISCREPANCY
# ======================================================================================

"""
1. THE BATCH SIZE EFFECT:
   In PPO, the total Batch Size = num_envs * num_steps.
   - When you increased num_envs from 8 to 16 (keeping num_steps the same), 
     you effectively doubled the amount of data the agent sees before every update.
   - However, with a fixed minibatch_size (e.g., 66), the agent is now performing 
     fewer updates relative to the total amount of data collected.
   - Essentially, you made the updates "slower" and more conservative, which often 
     requires increasing total_timesteps to reach the same level of performance.

2. LEARNING RATE & NOISE:
   - More environments provide a "smoother" gradient (less noise). While this sounds 
     good, RL sometimes benefits from a bit of noise to escape local optima 
     (like your agent getting stuck on "Action 7").
   - By doubling the data without adjusting the learning rate (lr), the agent 
     might be learning too slowly to reach a good policy within 100,000 steps.

3. EVALUATION VARIANCE:
   - PPO performance is measured by 'Test Reward' during training.
   - If the seed used for evaluation is different, or if the agent hasn't 
     converged yet, you might just be seeing a "unlucky" training run. 
     RL training is notoriously stochastic (random).
"""

# ======================================================================================
# CRITICAL GUIDELINES FOR CHANGING PPO PARAMETERS
# ======================================================================================

"""
AWARENESS CHECKLIST WHEN MODIFYING CONFIGS:

- RATIO OF BATCH TO MINIBATCH:
  Always ensure (num_envs * num_steps) / minibatch_size results in a reasonable 
  number of updates per iteration. If this ratio gets too high, the agent 
  overfits; if too low, it learns nothing.

- THE "HORIZON" VS "NUM_STEPS":
  In your inventory problem, an episode is 14-15 steps.
  If num_steps is 128, the agent is collecting ~9 full episodes per environment 
  before updating. This is usually good. If you set num_steps < 15, 
  the agent would only see partial episodes, which makes learning very difficult.

- ENTROPY COEFFICIENT (ent_coef):
  If your agent is stuck "spamming" one action (like ACTION 7), increase 
  ent_coef (e.g., from 0.01 to 0.05). This forces the agent to keep 
  exploring other order quantities.

- DISCOUNT FACTOR (gamma):
  For inventory, gamma (0.99) is critical. It tells the agent 
  that ordering too much today is bad because of HOLDING costs tomorrow, 
  and ordering too little is bad because of BACKLOG costs later.

- ANNEAL_LR:
  When this is True, the learning rate shrinks to 0 at the end of total_timesteps.
  If you increase total_timesteps, the "cooling down" of the learning 
  process happens over a longer period, usually leading to better stability.
"""
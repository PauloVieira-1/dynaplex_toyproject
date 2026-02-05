from dynaplex import PPOTrainerConfig, PPOTrainer

def train_PPO(mdp, number_iterations=50):

    config = PPOTrainerConfig(
        seed = 42,
        device = "cpu",
        hidden_sizes=[128, 128],
        num_steps= 2 * number_iterations,
        minibatch_size=64,
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




    
from dynaplex import PPOTrainerConfig, PPOTrainer


def train_PPO(mdp, load_policy=False, total_timesteps=500000):

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
    print(f"-" * 80)

    ppo_trainer = PPOTrainer(mdp=mdp, config=config)

    if load_policy:
        print("Loading previously trained policy...")
        trained_policy = ppo_trainer.load_trained_policy()
        print("Policy loaded successfully.")
    else:
        print("Training policy...")
        trained_policy = ppo_trainer.train()
        print("Training completed.")

    print(f"Trained policy: {trained_policy}")

    return trained_policy



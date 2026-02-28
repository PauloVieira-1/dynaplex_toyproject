from typing import List
from dynaplex import PPOTrainerConfig, PPOTrainer


"""
https://en.wikipedia.org/wiki/Mixed_radix --> Helpful for later 

The encoder takes a list of action quantities (one per node) 
and the corresponding action dimensions (max order quantity + 1 for each node) 
and converts it into a single integer index that represents the combined action across all nodes.

The decoder then reconstructs the origional list of actions.


For exmaple:

action_quantities = [2, 3, 1]
action_dims = [5, 5, 5]  # Each node can order between 0 and 4 units (5 possible actions)

The encoder calculates the flat index as follows:
    - Start with flat_index = 0 and multiplier = 1
    - For the last node (1 unit, dim 5): flat_index += 1 * 1 = 1; multiplier *= 5 = 5
    - For the second node (3 units, dim 5): flat_index += 3 * 5 = 15; multiplier *= 5 = 25
    - For the first node (2 units, dim 5): flat_index += 2 * 25 = 50; multiplier *= 5 = 125 

Final flat_index = 1 + 15 + 50 = 66

The decoder takes the flat index (66) and action dimensions to reconstruct the original action list:
    - Start with action_index = 66 and action_dims = [5, 5, 5]
    - For the last node (dim 5): action = 66 % 5 = 1; action_index //= 5 = 13
    - For the second node (dim 5): action = 13 % 5 = 3; action_index //= 5 = 2
    - For the first node (dim 5): action = 2 % 5 = 2; action_index //= 5 = 0      
    - Reverses the action list

Final action_quantities = [2, 3, 1]


"""



def encode_action(order_quantities: List[int], action_dims: List[int]) -> int:

    flat_index = 0
    multiplier = 1

    for action, dim in reversed(list(zip(order_quantities, action_dims))):
        flat_index += action * multiplier
        multiplier *= dim

    return flat_index



def decode_action(action_index: int, action_dims: List[int]) -> List[int]:

    order_quantities = []
    for dim in reversed(action_dims):
        order_quantities.append(action_index % dim)
        action_index //= dim
    return list(reversed(order_quantities))


def train_PPO(mdp, number_iterations=50):

    config = PPOTrainerConfig(
        seed = 42,
        device = "cpu",
        hidden_sizes=[64, 64],
        num_steps= 2 * number_iterations,
        minibatch_size=64,
        ent_coef=0.01,
        gamma=0.99,
        lr=3e-4,
        total_timesteps=100000, # Total n. interactions across all iterations
        logdir=None,  
        num_envs=8, # Number of parallel environments for data collection 
    )



    print(f"Training PPO with config: {config}")
    print(f"-" * 80)

    ppo_trainer = PPOTrainer(mdp=mdp, config=config)
    load_policy = False

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



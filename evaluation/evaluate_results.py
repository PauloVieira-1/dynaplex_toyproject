import torch
import matplotlib.pyplot as plt
import os
import glob

def load_and_plot_training(log_dir: str):

    checkpoints = sorted(glob.glob(os.path.join(log_dir, 'checkpoint_step_*.pth')))
    
    steps = []
    rewards = []
    
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print(f"Keys in {os.path.basename(ckpt_path)}: {ckpt.keys() if isinstance(ckpt, dict) else type(ckpt)}")
        
        step = int(ckpt_path.split('checkpoint_step_')[1].replace('.pth', ''))
        steps.append(step)
        
        reward = ckpt.get('mean_reward') or ckpt.get('episode_reward') or ckpt.get('reward')
        rewards.append(reward)
    
    for name in ['best_policy.pth', 'final_policy.pth']:
        path = os.path.join(log_dir, name)
        
        if os.path.exists(path):
            ckpt = torch.load(path, map_location='cpu')
            print(f"\n{name} keys: {ckpt.keys() if isinstance(ckpt, dict) else type(ckpt)}")
    
    if any(r is not None for r in rewards):

        plt.figure(figsize=(10, 5))
        plt.plot(steps, rewards, marker='o')
        plt.xlabel('Training Step')
        plt.ylabel('Mean Reward')
        plt.title(f'PPO Training Progress — {os.path.basename(log_dir)}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'training_curve.png'))
        plt.show()

    else:
        print("failed.")

load_and_plot_training('../log/Node/ppo')
load_and_plot_training('../log/SupplyChainMDP/ppo')
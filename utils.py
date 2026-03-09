"""
Utility functions for saving, loading, and logging
"""
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

def create_checkpoint_dir():
    """Create timestamped checkpoint directory"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = Path("checkpoints") / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir

def get_latest_checkpoint_dir():
    """Get the most recent checkpoint directory"""
    checkpoints_root = Path("checkpoints")
    if not checkpoints_root.exists():
        return None
    subdirs = [d for d in checkpoints_root.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda d: d.stat().st_mtime)

def save_checkpoint(agent, episode, rewards_history, rps_history, 
                   loss_history, epsilon_history,
                   checkpoint_dir, filename="checkpoint.pth"):
    """
    Save training checkpoint with extended metrics
    """
    checkpoint_path = Path(checkpoint_dir) / filename
    
    checkpoint = {
        'episode': episode,
        'agent_state': agent.get_state_dict(),
        'rewards_history': rewards_history,
        'rps_history': rps_history,
        'loss_history': loss_history,       # Added
        'epsilon_history': epsilon_history  # Added
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def load_checkpoint(agent, checkpoint_path):
    """
    Load training checkpoint (Backward compatible)
    """
    checkpoint = torch.load(
        checkpoint_path, 
        map_location=agent.device,
        weights_only=False
    )
    
    agent.load_state_dict(checkpoint['agent_state'])
    
    # Backward compatibility for older checkpoints
    rewards = checkpoint.get('rewards_history', [])
    rps = checkpoint.get('rps_history', [])
    loss = checkpoint.get('loss_history', [0.0] * len(rewards))
    epsilon = checkpoint.get('epsilon_history', [0.1] * len(rewards))
    
    return (
        checkpoint['episode'],
        rewards,
        rps,
        loss,
        epsilon
    )

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in a directory"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    if not checkpoints:
        return None
    
    final_model = checkpoint_dir / "final_model.pth"
    if final_model.exists():
        return final_model
    
    return max(checkpoints, key=lambda p: p.stat().st_mtime)

def plot_training_results(rewards, rps, losses, epsilons, save_path):
    """
    Plot 4-panel training results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Rewards
    ax = axes[0, 0]
    ax.plot(rewards, label='Raw', alpha=0.3, color='blue')
    if len(rewards) > 10:
        window = min(50, len(rewards) // 5)
        avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), avg, label='Avg', color='blue', linewidth=2)
    ax.set_title('Total Reward per Episode')
    ax.grid(True, alpha=0.3)
    
    # 2. RPS
    ax = axes[0, 1]
    ax.plot(rps, label='Raw', alpha=0.3, color='orange')
    if len(rps) > 10:
        window = min(50, len(rps) // 5)
        avg = np.convolve(rps, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rps)), avg, label='Avg', color='orange', linewidth=2)
    ax.set_title('Reward Per Step (RPS)')
    ax.grid(True, alpha=0.3)
    
    # 3. Loss
    ax = axes[1, 0]
    ax.plot(losses, label='Loss', color='red', alpha=0.6)
    ax.set_yscale('log') # Log scale is usually better for loss
    ax.set_title('Avg Training Loss (Log Scale)')
    ax.grid(True, alpha=0.3)
    
    # 4. Epsilon
    ax = axes[1, 1]
    ax.plot(epsilons, label='Epsilon', color='green', linewidth=2)
    ax.set_title('Exploration Rate (Epsilon)')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

class Logger:
    """Simple logger to write to both console and file"""
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'w') as f:
            f.write(f"Training started at {datetime.now()}\n")
            f.write("="*60 + "\n\n")
    
    def log(self, message, print_console=True):
        if print_console:
            print(message)
        with open(self.log_path, 'a') as f:
            f.write(message + "\n")

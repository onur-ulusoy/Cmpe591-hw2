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
    """
    Create timestamped checkpoint directory
    
    Returns:
        checkpoint_dir: Path to created directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = Path("checkpoints") / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_latest_checkpoint_dir():
    """
    Get the most recent checkpoint directory
    
    Returns:
        latest_dir: Path to latest checkpoint directory, or None if none exist
    """
    checkpoints_root = Path("checkpoints")
    if not checkpoints_root.exists():
        return None
    
    # Get all subdirectories
    subdirs = [d for d in checkpoints_root.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    
    # Sort by modification time and return latest
    latest_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
    return latest_dir


def save_checkpoint(agent, episode, rewards_history, rps_history, 
                   checkpoint_dir, filename="checkpoint.pth"):
    """
    Save training checkpoint
    
    Args:
        agent: DQNAgent instance
        episode: Current episode number
        rewards_history: List of episode rewards
        rps_history: List of reward-per-step values
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
    """
    checkpoint_path = Path(checkpoint_dir) / filename
    
    checkpoint = {
        'episode': episode,
        'agent_state': agent.get_state_dict(),
        'rewards_history': rewards_history,
        'rps_history': rps_history,
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(agent, checkpoint_path):
    """
    Load training checkpoint
    
    Args:
        agent: DQNAgent instance to load into
        checkpoint_path: Path to checkpoint file
        
    Returns:
        episode: Episode number
        rewards_history: List of episode rewards
        rps_history: List of reward-per-step values
    """
    checkpoint = torch.load(checkpoint_path, map_location=agent.device)
    
    agent.load_state_dict(checkpoint['agent_state'])
    
    return (
        checkpoint['episode'],
        checkpoint['rewards_history'],
        checkpoint['rps_history']
    )


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in a directory
    
    Args:
        checkpoint_dir: Directory to search
        
    Returns:
        Path to latest checkpoint, or None if none found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    if not checkpoints:
        return None
    
    # Prefer final_model.pth, otherwise take latest by modification time
    final_model = checkpoint_dir / "final_model.pth"
    if final_model.exists():
        return final_model
    
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def plot_training_results(rewards_history, rps_history, save_path):
    """
    Plot and save training results
    
    Args:
        rewards_history: List of episode rewards
        rps_history: List of reward-per-step values
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Cumulative Reward
    axes[0].plot(rewards_history, label='Reward', alpha=0.6, linewidth=1)
    if len(rewards_history) > 10:
        window = min(20, len(rewards_history) // 5)
        moving_avg = np.convolve(
            rewards_history, 
            np.ones(window)/window, 
            mode='valid'
        )
        axes[0].plot(
            range(window-1, len(rewards_history)), 
            moving_avg, 
            label=f'{window}-episode MA', 
            linewidth=2,
            color='red'
        )
    axes[0].set_title('Cumulative Reward per Episode')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Reward Per Step (RPS)
    axes[1].plot(rps_history, color='orange', label='RPS', alpha=0.6, linewidth=1)
    if len(rps_history) > 10:
        window = min(20, len(rps_history) // 5)
        moving_avg = np.convolve(
            rps_history, 
            np.ones(window)/window, 
            mode='valid'
        )
        axes[1].plot(
            range(window-1, len(rps_history)), 
            moving_avg, 
            label=f'{window}-episode MA', 
            linewidth=2,
            color='darkred'
        )
    axes[1].set_title('Reward Per Step (RPS)')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('RPS')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Success Rate (assuming terminal = success)
    # This is a simplified version - you might want to track this explicitly
    axes[2].plot(rewards_history, color='green', alpha=0.4, linewidth=1)
    if len(rewards_history) > 10:
        window = min(20, len(rewards_history) // 5)
        moving_avg = np.convolve(
            rewards_history, 
            np.ones(window)/window, 
            mode='valid'
        )
        axes[2].plot(
            range(window-1, len(rewards_history)), 
            moving_avg, 
            linewidth=2,
            color='darkgreen',
            label='Smoothed Reward'
        )
    axes[2].set_title('Training Progress')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Reward')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


class Logger:
    """Simple logger to write to both console and file"""
    
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Clear existing log
        with open(self.log_path, 'w') as f:
            f.write(f"Training started at {datetime.now()}\n")
            f.write("="*60 + "\n\n")
    
    def log(self, message, print_console=True):
        """Write message to log file and optionally print to console"""
        if print_console:
            print(message)
        
        with open(self.log_path, 'a') as f:
            f.write(message + "\n")

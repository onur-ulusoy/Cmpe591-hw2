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
    
    # Create plots subdirectory
    (checkpoint_dir / "plots").mkdir(exist_ok=True)
    
    return checkpoint_dir

def get_latest_checkpoint_dir():
    checkpoints_root = Path("checkpoints")
    if not checkpoints_root.exists(): return None
    subdirs = [d for d in checkpoints_root.iterdir() if d.is_dir()]
    if not subdirs: return None
    return max(subdirs, key=lambda d: d.stat().st_mtime)

def save_checkpoint(agent, episode, history_dict, checkpoint_dir, filename="checkpoint.pth"):
    """
    Save training checkpoint with ALL history metrics
    history_dict: Dictionary containing lists of rewards, loss, etc.
    """
    checkpoint_path = Path(checkpoint_dir) / filename
    
    checkpoint = {
        'episode': episode,
        'agent_state': agent.get_state_dict(),
        'history': history_dict
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
    
    # Handle legacy checkpoints by providing defaults
    history = checkpoint.get('history', {})
    
    # If loading old format (direct keys), convert to dict
    if not history:
        history = {
            'rewards': checkpoint.get('rewards_history', []),
            'rps': checkpoint.get('rps_history', []),
            'loss': checkpoint.get('loss_history', []),
            'epsilon': checkpoint.get('epsilon_history', []),
            'q_values': [],
            'grad_norms': [],
            'lr': []
        }
        # Fill missing with zeros to match length
        n = len(history['rewards'])
        if len(history['loss']) < n: history['loss'] = [0.0] * n
        if len(history['epsilon']) < n: history['epsilon'] = [0.1] * n
        history['q_values'] = [0.0] * n
        history['grad_norms'] = [0.0] * n
        history['lr'] = [0.0001] * n # Guess default LR
        
    return checkpoint['episode'], history

def find_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists(): return None
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    if not checkpoints: return None
    
    final_model = checkpoint_dir / "final_model.pth"
    if final_model.exists(): return final_model
    return max(checkpoints, key=lambda p: p.stat().st_mtime)

def save_individual_plot(data, title, ylabel, save_path, color='blue', log_scale=False):
    """Helper to save a single nice looking plot"""
    plt.figure(figsize=(10, 6))
    plt.plot(data, color=color, alpha=0.6, linewidth=1, label='Raw')
    
    # Add moving average if enough data
    if len(data) > 20:
        window = min(50, len(data) // 10)
        if window > 1:
            avg = np.convolve(data, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(data)), avg, color='black', linewidth=2, linestyle='--', label='Avg')
    
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if log_scale:
        plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

def save_all_plots(history, checkpoint_dir):
    """Save all metrics to separate images in plots/ folder"""
    plots_dir = Path(checkpoint_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Rewards
    save_individual_plot(history['rewards'], "Total Reward", "Reward", plots_dir / "reward.png", 'blue')
    
    # 2. RPS
    save_individual_plot(history['rps'], "Reward Per Step (Efficiency)", "RPS", plots_dir / "rps.png", 'orange')
    
    # 3. Loss
    save_individual_plot(history['loss'], "Training Loss (MSE)", "Loss", plots_dir / "loss.png", 'red', log_scale=True)
    
    # 4. Epsilon
    save_individual_plot(history['epsilon'], "Exploration Rate", "Epsilon", plots_dir / "epsilon.png", 'green')
    
    # 5. Q-Values
    save_individual_plot(history['q_values'], "Average Q-Value Prediction", "Q-Value", plots_dir / "q_values.png", 'purple')
    
    # 6. Gradients
    save_individual_plot(history['grad_norms'], "Gradient Norm (Stability)", "Norm", plots_dir / "gradients.png", 'brown', log_scale=True)
    
    # 7. Learning Rate
    save_individual_plot(history['lr'], "Learning Rate", "LR", plots_dir / "lr.png", 'cyan')

class Logger:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'w') as f:
            f.write(f"Training started at {datetime.now()}\n")
            f.write("="*60 + "\n\n")
    
    def log(self, message, print_console=True):
        if print_console: print(message)
        with open(self.log_path, 'a') as f:
            f.write(message + "\n")

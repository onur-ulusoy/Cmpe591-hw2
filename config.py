"""
Configuration and hyperparameters for DQN training
"""
import torch

HYPERPARAMS = {
    # Training
    "n_episodes": 2,
    "n_actions": 8,
    
    # DQN Parameters
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_decay": 0.999,
    "epsilon_decay_iter": 10,
    "min_epsilon": 0.1,
    
    # Network Training
    "lr": 0.0001,
    "batch_size": 32,
    "update_freq": 4,
    "target_update_freq": 100,
    
    # Replay Buffer
    "buffer_length": 10000,
    
    # Checkpointing
    "checkpoint_freq": 1,  # Save every N episodes
    
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Environment settings
ENV_CONFIG = {
    "n_actions": 8,
    "render_mode_train": "offscreen",
    "render_mode_eval": "gui"
}

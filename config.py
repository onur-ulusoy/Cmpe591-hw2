import torch

HYPERPARAMS = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # RL Params
    "gamma": 0.99,
    "lr": 0.0005,              # Slightly lower for stability
    "batch_size": 128,
    "buffer_length": 50000,
    
    # Epsilon Greedy
    "epsilon_start": 1.0,
    "epsilon_decay": 0.995,
    "min_epsilon": 0.05,
    "epsilon_decay_iter": 100,
    
    # Network Updates
    "update_freq": 1,
    "target_update_freq": 1000,
    
    # Training
    "n_episodes": 5000,
    "checkpoint_freq": 50,
}

ENV_CONFIG = {
    "n_actions": 6, # +X, -X, +Y, -Y, +Z, -Z
    "render_mode_train": None
}

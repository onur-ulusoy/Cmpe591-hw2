import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for High-Level State
    
    Input: (Batch, 6) -> [ee_x, ee_y, obj_x, obj_y, goal_x, goal_y]
    Output: (Batch, n_actions)
    """
    
    def __init__(self, n_actions, state_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

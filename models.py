"""
Neural network architectures for DQN
"""
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Convolutional Q-Network for visual input
    
    Input: (Batch, 3, 128, 128) RGB images
    Output: (Batch, n_actions) Q-values
    """
    
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            # (3, 128, 128) -> (32, 64, 64)
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # (32, 64, 64) -> (64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # (64, 32, 32) -> (128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # (128, 16, 16) -> (256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # (256, 8, 8) -> (512, 4, 4)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # (512, 4, 4) -> (512, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # (512) -> (n_actions)
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (Batch, 3, 128, 128) tensor
            
        Returns:
            Q-values: (Batch, n_actions) tensor
        """
        return self.net(x)

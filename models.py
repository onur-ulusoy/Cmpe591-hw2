import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, n_actions, state_dim=6):
        """
        Compact MLP for High-Level State
        Structure: Input(6) -> 128 -> 64 -> Output(n_actions)
        """
        super(QNetwork, self).__init__()
        
        # Layer 1: Expand to 128 features
        self.fc1 = nn.Linear(state_dim, 128)
        self.ln1 = nn.LayerNorm(128)  # Stabilizes training
        
        # Layer 2: Compress to 64 features
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)   # Stabilizes training
        
        # Output Layer: Map to actions
        self.fc3 = nn.Linear(64, n_actions)
        
    def forward(self, x):
        # 1. Safety: Flatten input to (Batch, Features)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        # 2. Forward Pass
        # Input -> Layer 1 -> Norm -> ReLU
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        
        # Layer 1 -> Layer 2 -> Norm -> ReLU
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        
        # Layer 2 -> Output (Raw Q-Values, no activation)
        return self.fc3(x)

"""
DQN Agent implementation with Debugging Stats
"""
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models import QNetwork
from replay_buffer import ReplayBuffer
from config import HYPERPARAMS

class DQNAgent:
    """
    Deep Q-Network Agent with experience replay and target network
    """
    
    def __init__(self, n_actions, device):
        self.n_actions = n_actions
        self.device = device
        
        # Policy network
        self.policy_net = QNetwork(n_actions).to(device)
        self.target_net = QNetwork(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=HYPERPARAMS["lr"]
        )
        
        self.buffer = ReplayBuffer(HYPERPARAMS["buffer_length"])
        self.epsilon = HYPERPARAMS["epsilon_start"]
        self.update_count = 0

    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                state_t = state.unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax().item()

    def get_current_lr(self):
        """Get current learning rate from optimizer"""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0

    def update(self):
        """
        Perform one step of optimization
        Returns: (loss, avg_q_value, grad_norm)
        """
        if len(self.buffer) < HYPERPARAMS["batch_size"]:
            return 0.0, 0.0, 0.0

        # Sample batch
        state, action, reward, next_state, done = self.buffer.sample(
            HYPERPARAMS["batch_size"]
        )
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)

        # Compute Q(s, a)
        current_q = self.policy_net(state).gather(1, action)

        # Compute Target
        with torch.no_grad():
            max_next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
            target_q = reward + (HYPERPARAMS["gamma"] * max_next_q * (1 - done))

        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # --- DEBUG: Calculate Gradient Norm & Clip ---
        # Clipping prevents "exploding gradients" which ruin training
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()

        self.update_count += 1
        
        # Decay epsilon
        if self.update_count % HYPERPARAMS["epsilon_decay_iter"] == 0:
            self.epsilon = max(
                HYPERPARAMS["min_epsilon"], 
                self.epsilon * HYPERPARAMS["epsilon_decay"]
            )

        # Update target network
        if self.update_count % HYPERPARAMS["target_update_freq"] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item(), current_q.mean().item(), grad_norm.item()

    def get_state_dict(self):
        return {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count,
        }

    def load_state_dict(self, state_dict):
        self.policy_net.load_state_dict(state_dict['policy_net_state_dict'])
        self.target_net.load_state_dict(state_dict['target_net_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.epsilon = state_dict['epsilon']
        self.update_count = state_dict['update_count']

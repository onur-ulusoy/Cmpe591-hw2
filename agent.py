"""
DQN Agent implementation
"""
import random
import torch
import torch.nn as nn
import torch.optim as optim

from models import QNetwork
from replay_buffer import ReplayBuffer
from config import HYPERPARAMS


class DQNAgent:
    """
    Deep Q-Network Agent with experience replay and target network
    """
    
    def __init__(self, n_actions, device):
        """
        Initialize DQN agent
        
        Args:
            n_actions: Number of possible actions
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.n_actions = n_actions
        self.device = device
        
        # Policy network (updated every step)
        self.policy_net = QNetwork(n_actions).to(device)
        
        # Target network (updated periodically)
        self.target_net = QNetwork(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=HYPERPARAMS["lr"]
        )
        
        # Replay buffer
        self.buffer = ReplayBuffer(HYPERPARAMS["buffer_length"])
        
        # Exploration
        self.epsilon = HYPERPARAMS["epsilon_start"]
        self.update_count = 0

    def select_action(self, state, eval_mode=False):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state (3, 128, 128) tensor
            eval_mode: If True, use greedy policy (no exploration)
            
        Returns:
            action: Selected action (int)
        """
        if not eval_mode and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploit: best action according to policy
            with torch.no_grad():
                state_t = state.unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax().item()

    def update(self):
        """
        Perform one step of optimization on the policy network
        
        Returns:
            loss: Training loss (float), or 0.0 if buffer too small
        """
        if len(self.buffer) < HYPERPARAMS["batch_size"]:
            return 0.0

        # Sample batch from replay buffer
        state, action, reward, next_state, done = self.buffer.sample(
            HYPERPARAMS["batch_size"]
        )
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)

        # Compute Q(s, a) - Q-value of taken action
        current_q = self.policy_net(state).gather(1, action)

        # Compute target: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            max_next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
            target_q = reward + (HYPERPARAMS["gamma"] * max_next_q * (1 - done))

        # Compute loss and update
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
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
            
        return loss.item()

    def get_state_dict(self):
        """Get state dictionary for saving"""
        return {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count,
        }

    def load_state_dict(self, state_dict):
        """Load state dictionary"""
        self.policy_net.load_state_dict(state_dict['policy_net_state_dict'])
        self.target_net.load_state_dict(state_dict['target_net_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.epsilon = state_dict['epsilon']
        self.update_count = state_dict['update_count']

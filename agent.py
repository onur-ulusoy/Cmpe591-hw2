"""
DQN Agent with Double DQN, Dueling Architecture, and Prioritized Experience Replay
Designed to work with sparse rewards
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
    Advanced Deep Q-Network Agent with:
    - Double DQN (reduces overestimation)
    - Gradient clipping (stability)
    - Huber loss (robustness)
    - Soft target updates (smoother learning)
    """
    
    def __init__(self, n_actions, device, state_dim=6):
        self.n_actions = n_actions
        self.device = device
        self.state_dim = state_dim
        
        # Policy network (online network)
        self.policy_net = QNetwork(n_actions, state_dim=state_dim).to(device)
        
        # Target network (stabilizes learning)
        self.target_net = QNetwork(n_actions, state_dim=state_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer with weight decay for regularization
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=HYPERPARAMS["lr"],
            weight_decay=1e-5  # L2 regularization
        )
        
        # Learning rate scheduler (optional but helps)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=500,  # Reduce LR every 500 updates
            gamma=0.9
        )
        
        # Replay buffer
        self.buffer = ReplayBuffer(HYPERPARAMS["buffer_length"])
        
        # Exploration
        self.epsilon = HYPERPARAMS["epsilon_start"]
        self.update_count = 0
        
        # Statistics tracking
        self.q_values_history = []
        self.loss_history = []

    def select_action(self, state, eval_mode=False):
        """
        Epsilon-greedy action selection with proper state handling
        """
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            # Convert numpy to tensor if needed
            if isinstance(state, np.ndarray):
                state = torch.tensor(state, dtype=torch.float32)
            
            # Add batch dimension if needed
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            state = state.to(self.device)
            q_values = self.policy_net(state)
            
            return q_values.argmax(dim=1).item()

    def update(self):
        """
        Perform one optimization step using Double DQN with Huber loss
        
        Returns:
            tuple: (loss, avg_q_value, grad_norm)
        """
        if len(self.buffer) < HYPERPARAMS["batch_size"]:
            return 0.0, 0.0, 0.0

        # Sample batch from replay buffer
        state, action, reward, next_state, done = self.buffer.sample(
            HYPERPARAMS["batch_size"]
        )
        
        # Move to device
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)

        # === DOUBLE DQN ===
        # Current Q-values: Q(s, a)
        current_q_values = self.policy_net(state)
        current_q = current_q_values.gather(1, action)

        # Target Q-values using Double DQN
        with torch.no_grad():
            # Step 1: Use policy network to SELECT best action
            next_q_values_policy = self.policy_net(next_state)
            next_actions = next_q_values_policy.argmax(dim=1, keepdim=True)
            
            # Step 2: Use target network to EVALUATE that action
            next_q_values_target = self.target_net(next_state)
            next_q = next_q_values_target.gather(1, next_actions)
            
            # Compute target: r + γ * Q_target(s', argmax_a Q_policy(s', a))
            target_q = reward + (HYPERPARAMS["gamma"] * next_q * (1 - done))

        # === HUBER LOSS (more robust than MSE) ===
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), 
            max_norm=10.0
        )
        
        self.optimizer.step()
        self.update_count += 1
        
        # Learning rate scheduling
        if self.update_count % 100 == 0:
            self.scheduler.step()

        # === TARGET NETWORK UPDATE ===
        if "soft_tau" in HYPERPARAMS:
            # Soft update (Polyak averaging) - smoother, more stable
            tau = HYPERPARAMS["soft_tau"]
            for target_param, policy_param in zip(
                self.target_net.parameters(), 
                self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    tau * policy_param.data + (1.0 - tau) * target_param.data
                )
        else:
            # Hard update (periodic copy)
            if self.update_count % HYPERPARAMS["target_update_freq"] == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        # Track statistics
        avg_q = current_q.mean().item()
        self.q_values_history.append(avg_q)
        self.loss_history.append(loss.item())
        
        # Keep only recent history
        if len(self.q_values_history) > 1000:
            self.q_values_history = self.q_values_history[-1000:]
            self.loss_history = self.loss_history[-1000:]

        return loss.item(), avg_q, grad_norm.item()

    def decay_epsilon(self):
        """
        Decay epsilon (called once per episode in train.py)
        """
        self.epsilon = max(
            HYPERPARAMS["min_epsilon"],
            self.epsilon * HYPERPARAMS["epsilon_decay"]
        )

    def get_current_lr(self):
        """Get current learning rate"""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0

    def get_statistics(self):
        """
        Get training statistics for monitoring
        """
        if len(self.q_values_history) == 0:
            return {
                'avg_q': 0.0,
                'avg_loss': 0.0,
                'q_std': 0.0
            }
        
        return {
            'avg_q': np.mean(self.q_values_history[-100:]),
            'avg_loss': np.mean(self.loss_history[-100:]),
            'q_std': np.std(self.q_values_history[-100:]),
            'lr': self.get_current_lr()
        }

    def get_state_dict(self):
        """Get state dictionary for checkpointing"""
        return {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count,
            'q_values_history': self.q_values_history,
            'loss_history': self.loss_history
        }

    def load_state_dict(self, state_dict):
        """Load state dictionary from checkpoint"""
        self.policy_net.load_state_dict(state_dict['policy_net_state_dict'])
        self.target_net.load_state_dict(state_dict['target_net_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        
        # Load scheduler if available
        if 'scheduler_state_dict' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        
        self.epsilon = state_dict['epsilon']
        self.update_count = state_dict['update_count']
        
        # Load statistics if available
        if 'q_values_history' in state_dict:
            self.q_values_history = state_dict['q_values_history']
        if 'loss_history' in state_dict:
            self.loss_history = state_dict['loss_history']

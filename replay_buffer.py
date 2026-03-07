"""
Experience replay buffer for DQN
"""
import random
import collections
import torch


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples
    """
    
    def __init__(self, capacity):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add experience to buffer
        
        Args:
            state: Current state (3, 128, 128) tensor
            action: Action taken (int)
            reward: Reward received (float)
            next_state: Next state (3, 128, 128) tensor
            done: Whether episode ended (bool)
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of batched tensors: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        # Stack into batch tensors
        state = torch.stack(state)
        next_state = torch.stack(next_state)
        action = torch.tensor(action, dtype=torch.int64).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1)
        
        return state, action, reward, next_state, done

    def __len__(self):
        """Return current size of buffer"""
        return len(self.buffer)

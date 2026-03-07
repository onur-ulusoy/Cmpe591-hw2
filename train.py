import time
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os

from homework2 import Hw2Env

# ---------------------------------------------------------
# 1. Hyperparameters
# ---------------------------------------------------------
HYPERPARAMS = {
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_decay": 0.999,
    "epsilon_decay_iter": 10,
    "min_epsilon": 0.1,
    "lr": 0.0001,
    "batch_size": 32,
    "update_freq": 4,
    "target_update_freq": 100,
    "buffer_length": 10000,
    "n_episodes": 2,  # Adjust as needed
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_dir": "checkpoints",
    "checkpoint_freq": 1  # Save every 50 episodes
}

# ---------------------------------------------------------
# 2. The Network Architecture
# ---------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# 3. Replay Buffer
# ---------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        state = torch.stack(state)
        next_state = torch.stack(next_state)
        action = torch.tensor(action, dtype=torch.int64).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1)
        
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# ---------------------------------------------------------
# 4. DQN Agent
# ---------------------------------------------------------
class DQNAgent:
    def __init__(self, n_actions, device):
        self.n_actions = n_actions
        self.device = device
        
        self.policy_net = QNetwork(n_actions).to(device)
        self.target_net = QNetwork(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=HYPERPARAMS["lr"])
        self.buffer = ReplayBuffer(HYPERPARAMS["buffer_length"])
        
        self.epsilon = HYPERPARAMS["epsilon_start"]
        self.update_count = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                state_t = state.unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax().item()

    def update(self):
        if len(self.buffer) < HYPERPARAMS["batch_size"]:
            return 0.0

        state, action, reward, next_state, done = self.buffer.sample(HYPERPARAMS["batch_size"])
        state, next_state = state.to(self.device), next_state.to(self.device)
        action, reward, done = action.to(self.device), reward.to(self.device), done.to(self.device)

        current_q = self.policy_net(state).gather(1, action)

        with torch.no_grad():
            max_next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
            target_q = reward + (HYPERPARAMS["gamma"] * max_next_q * (1 - done))

        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        
        if self.update_count % HYPERPARAMS["epsilon_decay_iter"] == 0:
            self.epsilon = max(HYPERPARAMS["min_epsilon"], 
                               self.epsilon * HYPERPARAMS["epsilon_decay"])

        if self.update_count % HYPERPARAMS["target_update_freq"] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()

# ---------------------------------------------------------
# 5. Save and Load Functions
# ---------------------------------------------------------
def save_checkpoint(agent, episode, rewards_history, rps_history, filename):
    """Save training checkpoint"""
    checkpoint = {
        'episode': episode,
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'update_count': agent.update_count,
        'rewards_history': rewards_history,
        'rps_history': rps_history,
        'hyperparams': HYPERPARAMS
    }
    torch.save(checkpoint, filename)
    print(f"✓ Checkpoint saved: {filename}")


def load_checkpoint(agent, filename):
    """Load training checkpoint"""
    checkpoint = torch.load(filename, map_location=agent.device)
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint['epsilon']
    agent.update_count = checkpoint['update_count']
    
    print(f"✓ Checkpoint loaded: {filename}")
    print(f"  Episode: {checkpoint['episode']}")
    print(f"  Epsilon: {agent.epsilon:.4f}")
    
    return checkpoint['episode'], checkpoint['rewards_history'], checkpoint['rps_history']

# ---------------------------------------------------------
# 6. Main Training Loop
# ---------------------------------------------------------
def train():
    print("="*60)
    print("DQN TRAINING - Robotic Pushing Task")
    print("="*60)
    print(f"Device: {HYPERPARAMS['device']}")
    print(f"Episodes: {HYPERPARAMS['n_episodes']}")
    print(f"Checkpoint directory: {HYPERPARAMS['checkpoint_dir']}")
    print("="*60)
    
    # Create checkpoint directory
    os.makedirs(HYPERPARAMS['checkpoint_dir'], exist_ok=True)
    
    # Initialize Environment (offscreen for faster training)
    env = Hw2Env(n_actions=8, render_mode="offscreen")
    agent = DQNAgent(n_actions=8, device=HYPERPARAMS['device'])
    
    # Metrics
    rewards_history = []
    rps_history = []
    
    global_step = 0
    start_episode = 0
    
    # Check for existing checkpoint to resume training
    latest_checkpoint = os.path.join(HYPERPARAMS['checkpoint_dir'], 'latest_checkpoint.pth')
    if os.path.exists(latest_checkpoint):
        print(f"\nFound existing checkpoint. Resume training? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            start_episode, rewards_history, rps_history = load_checkpoint(agent, latest_checkpoint)
            start_episode += 1
            print(f"Resuming from episode {start_episode}\n")

    training_start_time = time.time()

    for episode in range(start_episode, HYPERPARAMS["n_episodes"]):
        episode_start_time = time.time()
        
        env.reset()
        state = env.state()
        
        done = False
        cumulative_reward = 0.0
        episode_steps = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated
            
            agent.buffer.push(state, action, reward, next_state, is_terminal)
            
            state = next_state
            cumulative_reward += reward
            episode_steps += 1
            global_step += 1
            
            if global_step % HYPERPARAMS["update_freq"] == 0:
                agent.update()

        episode_time = time.time() - episode_start_time
        rps = cumulative_reward / episode_steps if episode_steps > 0 else 0
        rewards_history.append(cumulative_reward)
        rps_history.append(rps)
        
        # Calculate ETA
        elapsed_time = time.time() - training_start_time
        episodes_done = episode - start_episode + 1
        avg_time_per_episode = elapsed_time / episodes_done
        remaining_episodes = HYPERPARAMS["n_episodes"] - episode - 1
        eta_seconds = remaining_episodes * avg_time_per_episode
        eta_minutes = eta_seconds / 60
        
        print(f"Episode {episode+1}/{HYPERPARAMS['n_episodes']} | "
              f"Reward: {cumulative_reward:.2f} | "
              f"RPS: {rps:.4f} | "
              f"Epsilon: {agent.epsilon:.4f} | "
              f"Steps: {episode_steps} | "
              f"Time: {episode_time:.1f}s | "
              f"ETA: {eta_minutes:.1f}min")
        
        # Save checkpoint periodically
        if (episode + 1) % HYPERPARAMS["checkpoint_freq"] == 0:
            checkpoint_path = os.path.join(
                HYPERPARAMS['checkpoint_dir'], 
                f'checkpoint_ep{episode+1}.pth'
            )
            save_checkpoint(agent, episode, rewards_history, rps_history, checkpoint_path)
            
            # Also update latest checkpoint
            save_checkpoint(agent, episode, rewards_history, rps_history, latest_checkpoint)

    # Save final checkpoint
    final_checkpoint = os.path.join(HYPERPARAMS['checkpoint_dir'], 'final_model.pth')
    save_checkpoint(agent, HYPERPARAMS["n_episodes"]-1, rewards_history, rps_history, final_checkpoint)
    
    total_time = time.time() - training_start_time
    print("\n" + "="*60)
    print(f"Training completed in {total_time/60:.1f} minutes")
    print(f"Final model saved: {final_checkpoint}")
    print("="*60)

    # Plotting
    plt.figure(figsize=(14, 5))
    
    # Plot 1: Cumulative Reward
    plt.subplot(1, 3, 1)
    plt.plot(rewards_history, label='Reward', alpha=0.6)
    # Add moving average
    if len(rewards_history) > 10:
        window = 10
        moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards_history)), moving_avg, 
                label=f'{window}-episode MA', linewidth=2)
    plt.title('Cumulative Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Reward Per Step (RPS)
    plt.subplot(1, 3, 2)
    plt.plot(rps_history, color='orange', label='RPS', alpha=0.6)
    if len(rps_history) > 10:
        window = 10
        moving_avg = np.convolve(rps_history, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rps_history)), moving_avg, 
                label=f'{window}-episode MA', linewidth=2, color='red')
    plt.title('Reward Per Step (RPS)')
    plt.xlabel('Episode')
    plt.ylabel('RPS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Epsilon Decay
    plt.subplot(1, 3, 3)
    epsilon_history = [HYPERPARAMS["epsilon_start"] * 
                      (HYPERPARAMS["epsilon_decay"] ** (i // HYPERPARAMS["epsilon_decay_iter"]))
                      for i in range(len(rewards_history))]
    epsilon_history = [max(HYPERPARAMS["min_epsilon"], e) for e in epsilon_history]
    plt.plot(epsilon_history, color='green', label='Epsilon')
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(HYPERPARAMS['checkpoint_dir'], 'training_results.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Training plot saved: {plot_path}")
    
    return agent


if __name__ == "__main__":
    train()

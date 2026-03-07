import torch
import numpy as np
import sys
import os

from homework2 import Hw2Env
from train import QNetwork, DQNAgent, HYPERPARAMS

def load_trained_agent(checkpoint_path, device):
    """Load a trained agent from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Create agent
    agent = DQNAgent(n_actions=8, device=device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    
    # Set to evaluation mode (no exploration)
    agent.epsilon = 0.0
    agent.policy_net.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Trained episodes: {checkpoint['episode'] + 1}")
    print(f"  Training epsilon: {checkpoint['epsilon']:.4f}")
    
    return agent


def play_episodes(agent, n_episodes=5, render_mode="gui", verbose=True):
    """
    Play episodes with the trained agent
    
    Args:
        agent: Trained DQNAgent
        n_episodes: Number of episodes to play
        render_mode: "gui" for visualization, "offscreen" for headless
        verbose: Print detailed step information
    """
    print("\n" + "="*60)
    print(f"PLAYING {n_episodes} EPISODES WITH TRAINED AGENT")
    print("="*60)
    
    # Create environment with GUI
    env = Hw2Env(n_actions=8, render_mode=render_mode)
    
    results = {
        'rewards': [],
        'rps': [],
        'steps': [],
        'successes': 0
    }
    
    for episode in range(n_episodes):
        env.reset()
        state = env.state()
        
        done = False
        cumulative_reward = 0.0
        episode_steps = 0
        
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{n_episodes}")
        print(f"{'='*60}")
        
        while not done:
            # Select best action (greedy policy)
            with torch.no_grad():
                state_t = state.unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_t)
                action = q_values.argmax().item()
                
                if verbose:
                    print(f"Step {episode_steps + 1}:")
                    print(f"  Action: {action}")
                    print(f"  Q-values: {q_values.cpu().numpy()[0]}")
            
            # Step environment
            next_state, reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated
            
            if verbose:
                print(f"  Reward: {reward:.4f}")
                high_level = env.high_level_state()
                print(f"  EE pos: [{high_level[0]:.3f}, {high_level[1]:.3f}]")
                print(f"  Obj pos: [{high_level[2]:.3f}, {high_level[3]:.3f}]")
                print(f"  Goal pos: [{high_level[4]:.3f}, {high_level[5]:.3f}]")
            
            state = next_state
            cumulative_reward += reward
            episode_steps += 1
        
        # Episode summary
        rps = cumulative_reward / episode_steps if episode_steps > 0 else 0
        results['rewards'].append(cumulative_reward)
        results['rps'].append(rps)
        results['steps'].append(episode_steps)
        
        if is_terminal:
            results['successes'] += 1
            status = "✓ SUCCESS - Goal reached!"
        else:
            status = "✗ FAILED - Timeout"
        
        print(f"\n{'-'*60}")
        print(f"Episode Result: {status}")
        print(f"  Total Reward: {cumulative_reward:.4f}")
        print(f"  RPS: {rps:.4f}")
        print(f"  Steps: {episode_steps}")
        print(f"{'-'*60}")
    
    # Print overall summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    print(f"Episodes played: {n_episodes}")
    print(f"Success rate: {results['successes']}/{n_episodes} "
          f"({100*results['successes']/n_episodes:.1f}%)")
    print(f"Average reward: {np.mean(results['rewards']):.4f} ± "
          f"{np.std(results['rewards']):.4f}")
    print(f"Average RPS: {np.mean(results['rps']):.4f} ± "
          f"{np.std(results['rps']):.4f}")
    print(f"Average steps: {np.mean(results['steps']):.1f} ± "
          f"{np.std(results['steps']):.1f}")
    print("="*60)
    
    return results


def main():
    # Configuration
    checkpoint_path = "checkpoints/final_model.pth"  # Default checkpoint
    n_episodes = 5
    verbose = True
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    if len(sys.argv) > 2:
        n_episodes = int(sys.argv[2])
    if len(sys.argv) > 3:
        verbose = sys.argv[3].lower() == 'true'
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pth'):
                    print(f"  - {os.path.join(checkpoint_dir, f)}")
        else:
            print(f"  Checkpoint directory '{checkpoint_dir}' not found!")
            print("  Please train the model first using: python3 train_dqn.py")
        return
    
    # Load agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = load_trained_agent(checkpoint_path, device)
    
    # Play episodes with GUI
    play_episodes(agent, n_episodes=n_episodes, render_mode="gui", verbose=verbose)


if __name__ == "__main__":
    print("="*60)
    print("DQN AGENT PLAYER")
    print("="*60)
    print("Usage: python3 play.py [checkpoint_path] [n_episodes] [verbose]")
    print("Example: python3 play.py checkpoints/final_model.pth 10 True")
    print("="*60 + "\n")
    
    main()

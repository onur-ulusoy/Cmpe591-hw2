"""
Play/evaluate trained DQN agent with visualization
"""
import sys
import numpy as np
from pathlib import Path

from homework2 import Hw2Env
from agent import DQNAgent
from config import HYPERPARAMS, ENV_CONFIG
from utils import (
    get_latest_checkpoint_dir,
    find_latest_checkpoint,
    load_checkpoint
)


def play_episodes(agent, n_episodes=5, verbose=True):
    """
    Play episodes with trained agent
    
    Args:
        agent: Trained DQNAgent
        n_episodes: Number of episodes to play
        verbose: Print detailed step information
        
    Returns:
        results: Dictionary with performance metrics
    """
    print("\n" + "="*60)
    print(f"PLAYING {n_episodes} EPISODES WITH TRAINED AGENT")
    print("="*60)
    
    # Create environment with GUI
    env = Hw2Env(
        n_actions=ENV_CONFIG["n_actions"], 
        render_mode=ENV_CONFIG["render_mode_eval"]
    )
    
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
            # Select best action (eval mode = greedy)
            action = agent.select_action(state, eval_mode=True)
            
            if verbose:
                print(f"Step {episode_steps + 1}: Action={action}")
            
            # Step environment
            next_state, reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated
            
            if verbose:
                print(f"  Reward: {reward:.4f}")
            
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
    
    # Overall summary
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
    """Main entry point"""
    print("="*60)
    print("DQN AGENT PLAYER")
    print("="*60)
    
    # Parse arguments
    checkpoint_dir = None
    n_episodes = 5
    verbose = True
    
    if len(sys.argv) > 1:
        checkpoint_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        n_episodes = int(sys.argv[2])
    if len(sys.argv) > 3:
        verbose = sys.argv[3].lower() == 'true'
    
    # Find checkpoint directory
    if checkpoint_dir is None:
        print("No checkpoint specified, looking for latest...")
        checkpoint_dir = get_latest_checkpoint_dir()
        if checkpoint_dir is None:
            print("Error: No checkpoints found!")
            print("Please train a model first: python3 train.py")
            return
        print(f"Found latest checkpoint: {checkpoint_dir}")
    else:
        if not checkpoint_dir.exists():
            print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
            return
    
    # Find checkpoint file
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        print(f"Error: No checkpoint files found in {checkpoint_dir}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    print("="*60 + "\n")
    
    # Load agent
    agent = DQNAgent(
        n_actions=ENV_CONFIG["n_actions"], 
        device=HYPERPARAMS['device']
    )
    
    episode, _, _ = load_checkpoint(agent, checkpoint_path)
    print(f"✓ Model loaded (trained for {episode + 1} episodes)")
    print(f"✓ Epsilon: {agent.epsilon:.4f} (will use greedy policy)")
    
    # Play episodes
    play_episodes(agent, n_episodes=n_episodes, verbose=verbose)


if __name__ == "__main__":
    main()

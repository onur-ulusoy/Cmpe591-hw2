"""
Play trained DQN agent with GUI visualization
Uses multiprocessing to run offscreen environment in separate process
"""
import sys
import numpy as np
import torch
from pathlib import Path
import time
import multiprocessing as mp
from queue import Empty

from homework2 import Hw2Env
from agent import DQNAgent
from config import HYPERPARAMS, ENV_CONFIG
from utils import (
    get_latest_checkpoint_dir,
    find_latest_checkpoint,
    load_checkpoint
)


def offscreen_worker(checkpoint_path, seed_queue, action_queue):
    """
    Worker process that runs offscreen environment and agent
    Receives seeds, returns actions
    """
    # Load agent
    agent = DQNAgent(
        n_actions=ENV_CONFIG["n_actions"], 
        device=HYPERPARAMS['device']
    )
    load_checkpoint(agent, checkpoint_path)
    
    # Create offscreen environment
    env = Hw2Env(n_actions=ENV_CONFIG["n_actions"], render_mode="offscreen")
    
    while True:
        try:
            # Get seed for new episode
            msg = seed_queue.get(timeout=1.0)
            
            if msg == "STOP":
                break
            
            seed = msg
            
            # Reset environment with seed
            env._create_scene(seed=seed)
            env.reset()
            state = env.state()
            
            # Send ready signal
            action_queue.put("READY")
            
            # Play episode
            done = False
            while not done:
                # Select action
                action = agent.select_action(state, eval_mode=True)
                
                # Send action to main process
                action_queue.put(action)
                
                # Wait for step confirmation
                cmd = seed_queue.get()
                if cmd == "STOP":
                    break
                
                # Step environment
                next_state, reward, is_terminal, is_truncated = env.step(action)
                done = is_terminal or is_truncated
                
                state = next_state
                
                # Send done status
                action_queue.put(("DONE", is_terminal, is_truncated))
            
        except Empty:
            continue
        except Exception as e:
            print(f"Worker error: {e}")
            action_queue.put(("ERROR", str(e)))
            break


def play_episodes_with_agent(checkpoint_path, n_episodes=5, verbose=True, delay=0.05):
    """
    Play episodes with trained agent using GUI
    
    Args:
        checkpoint_path: Path to checkpoint file
        n_episodes: Number of episodes to play
        verbose: Print detailed step information
        delay: Delay between steps (seconds)
    """
    print("\n" + "="*60)
    print(f"PLAYING {n_episodes} EPISODES WITH TRAINED AGENT")
    print("GUI Mode - Watch the visualization window!")
    print("="*60)
    
    # Start worker process
    seed_queue = mp.Queue()
    action_queue = mp.Queue()
    
    worker = mp.Process(
        target=offscreen_worker,
        args=(checkpoint_path, seed_queue, action_queue)
    )
    worker.start()
    
    results = {
        'rewards': [],
        'rps': [],
        'steps': [],
        'successes': 0
    }
    
    try:
        for episode in range(n_episodes):
            # Generate seed
            seed = np.random.randint(0, 1000000)
            
            # Create GUI environment
            env_gui = Hw2Env(n_actions=ENV_CONFIG["n_actions"], render_mode="gui")
            env_gui._create_scene(seed=seed)
            env_gui.reset()
            
            # Send seed to worker
            seed_queue.put(seed)
            
            # Wait for worker ready
            msg = action_queue.get()
            if msg != "READY":
                print(f"Worker error: {msg}")
                break
            
            done = False
            cumulative_reward = 0.0
            episode_steps = 0
            
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"{'='*60}")
            
            while not done:
                # Get action from worker
                action = action_queue.get()
                
                if isinstance(action, tuple) and action[0] == "ERROR":
                    print(f"Worker error: {action[1]}")
                    break
                
                if verbose:
                    print(f"Step {episode_steps + 1}: Action={action}")
                
                # Step GUI environment
                _, reward, is_terminal, is_truncated = env_gui.step(action)
                
                if verbose:
                    print(f"  Reward: {reward:.4f}")
                    try:
                        high_level = env_gui.high_level_state()
                        print(f"  EE pos: [{high_level[0]:.3f}, {high_level[1]:.3f}]")
                        print(f"  Obj pos: [{high_level[2]:.3f}, {high_level[3]:.3f}]")
                        print(f"  Goal pos: [{high_level[4]:.3f}, {high_level[5]:.3f}]")
                    except:
                        pass
                
                cumulative_reward += reward
                episode_steps += 1
                
                # Signal worker to continue
                seed_queue.put("STEP")
                
                # Get done status from worker
                status = action_queue.get()
                if isinstance(status, tuple) and status[0] == "DONE":
                    _, worker_terminal, worker_truncated = status
                    done = worker_terminal or worker_truncated
                
                # Delay for visualization
                time.sleep(delay)
            
            # Clean up GUI environment
            del env_gui
            
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
            
            # Pause between episodes
            time.sleep(1.0)
    
    finally:
        # Stop worker
        seed_queue.put("STOP")
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()
    
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
    print("DQN AGENT PLAYER - GUI MODE WITH TRAINED AGENT")
    print("="*60)
    
    # Parse arguments
    checkpoint_dir = None
    n_episodes = 5
    verbose = True
    delay = 0.05
    
    if len(sys.argv) > 1:
        checkpoint_dir = Path(sys.argv[1])
    
    if len(sys.argv) > 2:
        try:
            n_episodes = int(sys.argv[2])
        except ValueError:
            pass
    
    if len(sys.argv) > 3:
        verbose = sys.argv[3].lower() == 'true'
    
    if len(sys.argv) > 4:
        try:
            delay = float(sys.argv[4])
        except ValueError:
            pass
    
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
    
    print(f"✓ Starting worker process with trained agent")
    print(f"✓ Will use greedy policy (eval mode)")
    print(f"\n🎮 Starting GUI visualization with YOUR trained agent...\n")
    
    # Play episodes
    play_episodes_with_agent(checkpoint_path, n_episodes=n_episodes, verbose=verbose, delay=delay)


if __name__ == "__main__":
    # Required for multiprocessing on some systems
    mp.set_start_method('spawn', force=True)
    
    print("\nUsage: python3 play_gui.py [checkpoint_dir] [n_episodes] [verbose] [delay]")
    print("  checkpoint_dir: Path to checkpoint (default: latest)")
    print("  n_episodes: Number of episodes (default: 5)")
    print("  verbose: True or False (default: True)")
    print("  delay: Seconds between steps (default: 0.05)")
    print("\nExamples:")
    print("  python3 play_gui.py                                    # Use latest")
    print("  python3 play_gui.py checkpoints/2026-03-07_17-59-32    # Specific")
    print("  python3 play_gui.py checkpoints/2026-03-07_17-59-32 10 # 10 episodes")
    print("  python3 play_gui.py checkpoints/2026-03-07_17-59-32 5 True 0.1 # Slower")
    print("="*60 + "\n")
    
    main()

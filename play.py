"""
Play trained DQN agent.
Supports GUI visualization, automatic checkpoint loading, and CLI arguments.
"""
import sys
import time
import torch
import numpy as np
from pathlib import Path
import argparse

from homework2 import Hw2Env
from agent import DQNAgent
from config import HYPERPARAMS, ENV_CONFIG
from utils import (
    get_latest_checkpoint_dir,
    find_latest_checkpoint,
    load_checkpoint
)

def play_episodes(checkpoint_path, n_episodes=5, verbose=True, delay=0.05, render_mode="gui"):
    print(f"\nLoading model from: {checkpoint_path}")
    
    # 1. Initialize Agent
    state_dim = 9 
    agent = DQNAgent(
        n_actions=ENV_CONFIG["n_actions"], 
        device=HYPERPARAMS['device'],
        state_dim=state_dim
    )
    
    # 2. Load Weights
    load_checkpoint(agent, checkpoint_path)
    
    # 3. Set to Evaluation Mode (Greedy)
    agent.epsilon = 0.0
    if hasattr(agent, 'policy_net'):
        agent.policy_net.eval()
    elif hasattr(agent, 'q_net'):
        agent.q_net.eval()
    
    # 4. Initialize Environment
    print(f"Initializing Environment (Mode: {render_mode})...")
    env = Hw2Env(n_actions=ENV_CONFIG["n_actions"], render_mode=render_mode)
    
    # --- SUCCESS TOLERANCE ---
    SUCCESS_DIST = 0.03 
    
    results = {
        'rewards': [],
        'rps': [],
        'steps': [],
        'successes': 0
    }

    for episode in range(n_episodes):
        print(f"\n{'='*40}")
        print(f"Episode {episode + 1} / {n_episodes}")
        print(f"{'='*40}")
        
        # Reset Env
        reset_res = env.reset()
        if isinstance(reset_res, tuple):
            state_np = reset_res[0]
        else:
            state_np = reset_res

        if state_np is None or (isinstance(state_np, np.ndarray) and state_np.size == 0):
             state_np = env.high_level_state()

        state = torch.tensor(state_np, dtype=torch.float32)

        done = False
        step_count = 0
        ep_reward = 0.0
        
        while not done:
            # Select Action
            try:
                action = agent.select_action(state, eval_mode=True)
            except TypeError:
                action = agent.select_action(state)
            
            # Step Environment
            step_res = env.step(action)
            
            if len(step_res) == 4:
                next_state_np, reward, terminal, truncated = step_res
            elif len(step_res) == 5:
                next_state_np, reward, terminal, truncated, _ = step_res
            else:
                raise ValueError(f"Unexpected step return length: {len(step_res)}")

            # --- 🔍 MANUAL SUCCESS CHECK ---
            try:
                obj_pos = env.data.body("obj1").xpos
                goal_pos = env.data.site("goal").xpos
                dist_to_goal = np.linalg.norm(obj_pos - goal_pos)
                
                if dist_to_goal < SUCCESS_DIST:
                    print(f"\n🎯 SUCCESS! Object reached goal (Dist: {dist_to_goal:.4f})")
                    
                    # 1. Force one last render so you see it at the goal
                    if render_mode == "gui" and hasattr(env, 'viewer') and env.viewer is not None:
                        env.viewer.render()
                    
                    # 2. Wait 1 second to celebrate
                    if render_mode == "gui":
                        time.sleep(1.0)
                        
                    terminal = True
                    done = True
                    reward += 10.0 # Bonus for stats
            except Exception:
                pass
            # -------------------------------

            # Standard Render
            if not done and render_mode == "gui" and hasattr(env, 'viewer') and env.viewer is not None:
                env.viewer.render()
            
            # Update State
            state = torch.tensor(next_state_np, dtype=torch.float32)
            
            ep_reward += reward
            step_count += 1
            
            if not done:
                done = terminal or truncated
            
            if verbose and not done:
                print(f"\rStep {step_count:03d} | Act: {action} | Rw: {reward:.4f}", end="")
            
            # Standard delay between steps
            if not done and render_mode == "gui":
                time.sleep(delay)

        # Episode End Stats
        print(f"\nDone. Total Reward: {ep_reward:.4f}")
        
        rps = ep_reward / step_count if step_count > 0 else 0
        results['rewards'].append(ep_reward)
        results['rps'].append(rps)
        results['steps'].append(step_count)

        if terminal:
            print(">>> STATUS: SUCCESS ✅")
            results['successes'] += 1
        elif truncated:
            print(">>> STATUS: TIMEOUT ⏳")
            
        # Small pause before next episode starts
        time.sleep(0.5) 

    # --- Final Summary ---
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    print(f"Episodes played: {n_episodes}")
    print(f"Success rate:    {results['successes']}/{n_episodes} ({100*results['successes']/n_episodes:.1f}%)")
    print(f"Average reward:  {np.mean(results['rewards']):.4f} ± {np.std(results['rewards']):.4f}")
    print(f"Average RPS:     {np.mean(results['rps']):.4f}")
    print(f"Average steps:   {np.mean(results['steps']):.1f}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Play trained DQN Agent")
    
    parser.add_argument("checkpoint_dir", nargs="?", type=str, default=None, 
                        help="Path to checkpoint directory (default: latest)")
    parser.add_argument("--episodes", "-n", type=int, default=5, help="Number of episodes")
    parser.add_argument("--delay", "-d", type=float, default=0.05, help="Delay between steps (seconds)")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI (headless)")
    parser.add_argument("--verbose", "-v", type=bool, default=True, help="Print step details")

    args = parser.parse_args()

    print("="*60)
    print("DQN PLAYER")
    print("="*60)

    # 1. Locate Checkpoint
    checkpoint_dir = None
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        if not checkpoint_dir.exists():
            print(f"❌ Error: Directory not found: {checkpoint_dir}")
            return
    else:
        print("Looking for latest checkpoint...")
        checkpoint_dir = get_latest_checkpoint_dir()
    
    if checkpoint_dir is None:
        print("❌ Error: No checkpoint directory found.")
        print("Run training first: python3 train.py")
        return

    checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        print(f"❌ Error: No .pth files found in {checkpoint_dir}")
        return

    # 2. Determine Render Mode
    render_mode = "offscreen" if args.no_gui else "gui"

    # 3. Run
    play_episodes(
        checkpoint_path, 
        n_episodes=args.episodes, 
        verbose=args.verbose, 
        delay=args.delay,
        render_mode=render_mode
    )

if __name__ == "__main__":
    main()

"""
Play trained DQN agent.
Includes --render_dt for Slow Motion control without freezing the GUI.
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

def smart_sleep(env, duration_sec):
    """
    Waits for 'duration_sec' while keeping the GUI alive.
    It continuously calls render() to process window events,
    preventing the 'Application Not Responding' freeze.
    """
    if duration_sec <= 0:
        return

    # If no viewer, just standard sleep
    if not hasattr(env, 'viewer') or env.viewer is None:
        time.sleep(duration_sec)
        return

    # Loop until time is up, rendering frequently
    end_time = time.time() + duration_sec
    while time.time() < end_time:
        env.viewer.render()
        # Small sleep to prevent 100% CPU usage during wait
        time.sleep(0.01) 

def play_episodes(checkpoint_path, n_episodes=5, verbose=True, render_dt=0.02, render_mode="gui"):
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
    
    # 3. Set to Evaluation Mode
    agent.epsilon = 0.0
    if hasattr(agent, 'policy_net'):
        agent.policy_net.eval()
    elif hasattr(agent, 'q_net'):
        agent.q_net.eval()
    
    # 4. Initialize Environment
    print(f"Initializing Environment (Mode: {render_mode})...")
    env = Hw2Env(n_actions=ENV_CONFIG["n_actions"], render_mode=render_mode)
    
    # Success Tolerance
    SUCCESS_DIST = 0.03 
    
    results = {
        'rewards': [],
        'successes': 0
    }

    for episode in range(n_episodes):
        print(f"\n{'='*40}")
        print(f"Episode {episode + 1} / {n_episodes}")
        print(f"{'='*40}")
        
        # Reset
        reset_res = env.reset()
        state_np = reset_res[0] if isinstance(reset_res, tuple) else reset_res
        if state_np is None or (isinstance(state_np, np.ndarray) and state_np.size == 0):
             state_np = env.high_level_state()

        state = torch.tensor(state_np, dtype=torch.float32)

        done = False
        step_count = 0
        ep_reward = 0.0
        
        while not done:
            # 1. Select Action
            try:
                action = agent.select_action(state, eval_mode=True)
            except TypeError:
                action = agent.select_action(state)
            
            # 2. Step Environment
            step_res = env.step(action)
            
            if len(step_res) == 4:
                next_state_np, reward, terminal, truncated = step_res
            elif len(step_res) == 5:
                next_state_np, reward, terminal, truncated, _ = step_res
            else:
                raise ValueError(f"Unexpected step return length")

            # 3. Render Immediately
            if render_mode == "gui" and hasattr(env, 'viewer') and env.viewer is not None:
                env.viewer.render()

            # 4. Manual Success Check
            try:
                obj_pos = env.data.body("obj1").xpos
                goal_pos = env.data.site("goal").xpos
                dist_to_goal = np.linalg.norm(obj_pos - goal_pos)
                
                if dist_to_goal < SUCCESS_DIST:
                    print(f"\n🎯 SUCCESS! Object reached goal (Dist: {dist_to_goal:.4f})")
                    reward += 10.0
                    terminal = True
                    done = True
                    
                    # Pause on success (1.0s)
                    if render_mode == "gui":
                        smart_sleep(env, 1.0)
            except Exception:
                pass

            # Update State
            state = torch.tensor(next_state_np, dtype=torch.float32)
            ep_reward += reward
            step_count += 1
            
            if not done:
                done = terminal or truncated
            
            if verbose and not done:
                print(f"\rStep {step_count:03d} | Act: {action} | Rw: {reward:.4f}", end="")
            
            # 5. CONTROL SPEED HERE
            # Use smart_sleep so window doesn't freeze even if dt is large
            if not done and render_mode == "gui":
                smart_sleep(env, render_dt)

        # Episode End
        print(f"\nDone. Total Reward: {ep_reward:.4f}")
        results['rewards'].append(ep_reward)

        if terminal:
            print(">>> STATUS: SUCCESS ✅")
            results['successes'] += 1
        elif truncated:
            print(">>> STATUS: TIMEOUT ⏳")
            
        # Pause between episodes
        if render_mode == "gui":
            smart_sleep(env, 0.5)

    # --- Summary ---
    print("\n" + "="*60)
    print(f"Success rate: {results['successes']}/{n_episodes} ({100*results['successes']/n_episodes:.1f}%)")
    print(f"Avg reward:   {np.mean(results['rewards']):.4f}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", nargs="?", type=str, default=None)
    parser.add_argument("--episodes", "-n", type=int, default=5)
    
    # --- SPEED CONTROL ---
    # 0.02 = Fast / Real-time
    # 0.10 = Moderate
    # 0.30 = Slow Motion
    parser.add_argument("--render_dt", "-dt", type=float, default=0.05, 
                        help="Time to wait between steps. Increase for slow motion.")
    
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--verbose", "-v", type=bool, default=True)
    args = parser.parse_args()

    # Locate Checkpoint
    checkpoint_dir = None
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = get_latest_checkpoint_dir()
    
    if not checkpoint_dir or not checkpoint_dir.exists():
        print("❌ Error: No checkpoint found.")
        return

    checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    if not checkpoint_path:
        print("❌ Error: No .pth file found.")
        return

    render_mode = "offscreen" if args.no_gui else "gui"

    play_episodes(
        checkpoint_path, 
        n_episodes=args.episodes, 
        verbose=args.verbose, 
        render_dt=args.render_dt,
        render_mode=render_mode
    )

if __name__ == "__main__":
    main()

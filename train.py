"""
Training script for DQN agent (High-Level State Version)
"""
import time
import sys
import torch
import numpy as np
from pathlib import Path

from homework2 import Hw2Env
from agent import DQNAgent
from config import HYPERPARAMS, ENV_CONFIG
from utils import (
    create_checkpoint_dir,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    plot_training_results,
    Logger
)


def train(resume_from=None):
    """
    Train DQN agent using High-Level State
    """
    # Create checkpoint directory
    if resume_from:
        checkpoint_dir = Path(resume_from)
        print(f"Resuming training from: {checkpoint_dir}")
    else:
        checkpoint_dir = create_checkpoint_dir()
        print(f"Created checkpoint directory: {checkpoint_dir}")
    
    # Initialize logger
    logger = Logger(checkpoint_dir / "training.log")
    
    logger.log("="*60)
    logger.log("DQN TRAINING - Robotic Pushing Task (High-Level State)")
    logger.log("="*60)
    logger.log(f"Device: {HYPERPARAMS['device']}")
    logger.log(f"Episodes: {HYPERPARAMS['n_episodes']}")
    logger.log(f"Checkpoint directory: {checkpoint_dir}")
    logger.log("="*60)
    
    # Initialize environment
    # Note: render_mode can be 'offscreen' or None since we don't use pixels for training
    env = Hw2Env(
        n_actions=ENV_CONFIG["n_actions"], 
        render_mode=ENV_CONFIG["render_mode_train"]
    )
    
    # Initialize Agent
    agent = DQNAgent(
        n_actions=ENV_CONFIG["n_actions"], 
        device=HYPERPARAMS['device']
    )
    
    # Initialize metrics
    rewards_history = []
    rps_history = []
    start_episode = 0
    
    # Resume from checkpoint if specified
    if resume_from:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            logger.log(f"\nLoading checkpoint: {latest_checkpoint}")
            start_episode, rewards_history, rps_history = load_checkpoint(
                agent, latest_checkpoint
            )
            start_episode += 1
            logger.log(f"Resuming from episode {start_episode}\n")
        else:
            logger.log("No checkpoint found, starting from scratch\n")
    
    global_step = 0
    training_start_time = time.time()

    # Training loop
    for episode in range(start_episode, HYPERPARAMS["n_episodes"]):
        episode_start_time = time.time()
        
        env.reset()
        
        # Get High-Level State ---
        state_np = env.high_level_state()
        state = torch.tensor(state_np, dtype=torch.float32)
        
        done = False
        cumulative_reward = 0.0
        episode_steps = 0
        
        # Episode loop
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Step environment
            # NOTE: env.step returns (pixel_state, reward, term, trunc)
            # We ignore the pixel_state (_) and get high_level_state manually
            _, reward, is_terminal, is_truncated = env.step(action)
            
            # Get next high-level state
            next_state_np = env.high_level_state()
            next_state = torch.tensor(next_state_np, dtype=torch.float32)
            
            done = is_terminal or is_truncated
            
            # Push to buffer
            agent.buffer.push(state, action, reward, next_state, is_terminal)
            
            state = next_state
            cumulative_reward += reward
            episode_steps += 1
            global_step += 1
            
            # Update network
            if global_step % HYPERPARAMS["update_freq"] == 0:
                loss = agent.update()

        # Episode metrics
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
        
        # Log progress
        log_msg = (
            f"Episode {episode+1}/{HYPERPARAMS['n_episodes']} | "
            f"Reward: {cumulative_reward:.2f} | "
            f"RPS: {rps:.4f} | "
            f"Epsilon: {agent.epsilon:.4f} | "
            f"Steps: {episode_steps} | "
            f"Time: {episode_time:.1f}s | "
            f"ETA: {eta_minutes:.1f}min"
        )
        logger.log(log_msg)
        
        # Save checkpoint periodically
        if (episode + 1) % HYPERPARAMS["checkpoint_freq"] == 0:
            checkpoint_path = save_checkpoint(
                agent, episode, rewards_history, rps_history,
                checkpoint_dir, f"checkpoint_ep{episode+1}.pth"
            )
            logger.log(f"✓ Checkpoint saved: {checkpoint_path}")
            
            # Also save as latest
            save_checkpoint(
                agent, episode, rewards_history, rps_history,
                checkpoint_dir, "latest_checkpoint.pth"
            )

    # Save final checkpoint
    final_checkpoint = save_checkpoint(
        agent, HYPERPARAMS["n_episodes"]-1, 
        rewards_history, rps_history,
        checkpoint_dir, "final_model.pth"
    )
    
    total_time = time.time() - training_start_time
    logger.log("\n" + "="*60)
    logger.log(f"Training completed in {total_time/60:.1f} minutes")
    logger.log(f"Final model saved: {final_checkpoint}")
    logger.log("="*60)

    # Plot results
    plot_path = checkpoint_dir / "training_results.png"
    plot_training_results(rewards_history, rps_history, plot_path)
    logger.log(f"Training plot saved: {plot_path}")
    
    return checkpoint_dir


def main():
    """Main entry point"""
    resume_from = None
    
    # Check for resume argument
    if len(sys.argv) > 1:
        resume_from = sys.argv[1]
        if not Path(resume_from).exists():
            print(f"Error: Checkpoint directory not found: {resume_from}")
            return
    
    checkpoint_dir = train(resume_from=resume_from)
    print(f"\n✓ Training complete! Results saved to: {checkpoint_dir}")
    print(f"\nTo visualize the trained agent, run:")
    print(f"  python3 play.py {checkpoint_dir}")


if __name__ == "__main__":
    main()

"""
Training script for DQN agent (High-Level State Version)
"""
import time
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

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
    # Create checkpoint directory
    if resume_from:
        checkpoint_dir = Path(resume_from)
        print(f"Resuming training from: {checkpoint_dir}")
    else:
        checkpoint_dir = create_checkpoint_dir()
        print(f"Created checkpoint directory: {checkpoint_dir}")
    
    # Initialize Logger & TensorBoard
    logger = Logger(checkpoint_dir / "training.log")
    writer = SummaryWriter(log_dir=checkpoint_dir / "tensorboard")
    
    logger.log("="*60)
    logger.log("DQN TRAINING - Robotic Pushing Task")
    logger.log("="*60)
    logger.log(f"Device: {HYPERPARAMS['device']}")
    logger.log(f"Checkpoint directory: {checkpoint_dir}")
    logger.log(f"TensorBoard: tensorboard --logdir={checkpoint_dir}/tensorboard")
    logger.log("="*60)
    
    env = Hw2Env(n_actions=ENV_CONFIG["n_actions"], render_mode=ENV_CONFIG["render_mode_train"])
    agent = DQNAgent(n_actions=ENV_CONFIG["n_actions"], device=HYPERPARAMS['device'])
    
    # Metrics
    rewards_history = []
    rps_history = []
    loss_history = []
    epsilon_history = []
    start_episode = 0
    
    # Resume
    if resume_from:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            logger.log(f"\nLoading checkpoint: {latest_checkpoint}")
            start_episode, rewards_history, rps_history, loss_history, epsilon_history = \
                load_checkpoint(agent, latest_checkpoint)
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
        
        state_np = env.high_level_state()
        state = torch.tensor(state_np, dtype=torch.float32)
        
        done = False
        cumulative_reward = 0.0
        episode_steps = 0
        episode_loss = 0.0
        update_counts = 0
        
        while not done:
            action = agent.select_action(state)
            _, reward, is_terminal, is_truncated = env.step(action)
            
            next_state_np = env.high_level_state()
            next_state = torch.tensor(next_state_np, dtype=torch.float32)
            done = is_terminal or is_truncated
            
            agent.buffer.push(state, action, reward, next_state, is_terminal)
            state = next_state
            
            cumulative_reward += reward
            episode_steps += 1
            global_step += 1
            
            # Update network
            if global_step % HYPERPARAMS["update_freq"] == 0:
                loss = agent.update()
                episode_loss += loss
                update_counts += 1

        # Calculate metrics
        avg_loss = episode_loss / update_counts if update_counts > 0 else 0.0
        rps = cumulative_reward / episode_steps if episode_steps > 0 else 0
        
        # Store history
        rewards_history.append(cumulative_reward)
        rps_history.append(rps)
        loss_history.append(avg_loss)
        epsilon_history.append(agent.epsilon)
        
        # TensorBoard Logging
        writer.add_scalar('Train/Reward', cumulative_reward, episode)
        writer.add_scalar('Train/RPS', rps, episode)
        writer.add_scalar('Train/Loss', avg_loss, episode)
        writer.add_scalar('Train/Epsilon', agent.epsilon, episode)
        
        # Console Logging
        episode_time = time.time() - episode_start_time
        elapsed_time = time.time() - training_start_time
        episodes_done = episode - start_episode + 1
        avg_time_per_episode = elapsed_time / episodes_done
        remaining_episodes = HYPERPARAMS["n_episodes"] - episode - 1
        eta_minutes = (remaining_episodes * avg_time_per_episode) / 60
        
        log_msg = (
            f"Ep {episode+1}/{HYPERPARAMS['n_episodes']} | "
            f"Rw: {cumulative_reward:.1f} | "
            f"Loss: {avg_loss:.4f} | "
            f"Eps: {agent.epsilon:.3f} | "
            f"Time: {episode_time:.1f}s | "
            f"ETA: {eta_minutes:.0f}m"
        )
        logger.log(log_msg)
        
        # Save & Plot
        if (episode + 1) % HYPERPARAMS["checkpoint_freq"] == 0:
            # Save Checkpoint
            checkpoint_path = save_checkpoint(
                agent, episode, rewards_history, rps_history, 
                loss_history, epsilon_history,
                checkpoint_dir, f"checkpoint_ep{episode+1}.pth"
            )
            
            # Save Latest
            save_checkpoint(
                agent, episode, rewards_history, rps_history,
                loss_history, epsilon_history,
                checkpoint_dir, "latest_checkpoint.pth"
            )

            # Update Plot
            plot_path = checkpoint_dir / "training_results.png"
            plot_training_results(rewards_history, rps_history, loss_history, epsilon_history, plot_path)
            logger.log(f"  ✓ Saved checkpoint & updated plot: {plot_path}")

    # Final Save
    save_checkpoint(
        agent, HYPERPARAMS["n_episodes"]-1, 
        rewards_history, rps_history, loss_history, epsilon_history,
        checkpoint_dir, "final_model.pth"
    )
    
    plot_path = checkpoint_dir / "training_results.png"
    plot_training_results(rewards_history, rps_history, loss_history, epsilon_history, plot_path)
    
    writer.close()
    logger.log("="*60)
    logger.log("Training Complete.")
    return checkpoint_dir

def main():
    resume_from = None
    if len(sys.argv) > 1:
        resume_from = sys.argv[1]
    
    checkpoint_dir = train(resume_from=resume_from)
    print(f"\nVisualize with: tensorboard --logdir={checkpoint_dir}/tensorboard")

if __name__ == "__main__":
    main()

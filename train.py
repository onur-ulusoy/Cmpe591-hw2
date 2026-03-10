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
    save_all_plots,
    Logger
)

def train(resume_from=None):
    if resume_from:
        checkpoint_dir = Path(resume_from)
        print(f"Resuming training from: {checkpoint_dir}")
        (checkpoint_dir / "plots").mkdir(exist_ok=True) # Ensure plots dir exists
    else:
        checkpoint_dir = create_checkpoint_dir()
        print(f"Created checkpoint directory: {checkpoint_dir}")
    
    logger = Logger(checkpoint_dir / "training.log")
    writer = SummaryWriter(log_dir=checkpoint_dir / "tensorboard")
    
    logger.log("="*60)
    logger.log("DQN TRAINING - FULL DEBUG MODE")
    logger.log("="*60)
    logger.log(f"Device: {HYPERPARAMS['device']}")
    logger.log(f"Checkpoint directory: {checkpoint_dir}")
    logger.log("="*60)
    
    env = Hw2Env(n_actions=ENV_CONFIG["n_actions"], render_mode=ENV_CONFIG["render_mode_train"])
    agent = DQNAgent(n_actions=ENV_CONFIG["n_actions"], device=HYPERPARAMS['device'])
    
    # History Dictionary
    history = {
        'rewards': [], 'rps': [], 'loss': [], 'epsilon': [],
        'q_values': [], 'grad_norms': [], 'lr': []
    }
    start_episode = 0
    
    if resume_from:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            logger.log(f"Loading checkpoint: {latest_checkpoint}")
            start_episode, loaded_history = load_checkpoint(agent, latest_checkpoint)
            # Merge loaded history
            for k, v in loaded_history.items():
                if k in history: history[k] = v
            start_episode += 1
            logger.log(f"Resuming from episode {start_episode}")
    
    global_step = 0
    training_start_time = time.time()

    for episode in range(start_episode, HYPERPARAMS["n_episodes"]):
        episode_start_time = time.time()
        env.reset()
        
        state = torch.tensor(env.high_level_state(), dtype=torch.float32)
        
        done = False
        cumulative_reward = 0.0
        episode_steps = 0
        
        # Episode stats accumulators
        ep_loss = 0.0
        ep_q = 0.0
        ep_grad = 0.0
        update_counts = 0
        
        while not done:
            action = agent.select_action(state)
            _, raw_reward, is_terminal, is_truncated = env.step(action)
            
            # --- FIX: Reward Scaling ---
            # Divide by 10.0 to keep Q-values small (~0.5 to ~2.0)
            reward = raw_reward * 0.1 
            
            next_state = torch.tensor(env.high_level_state(), dtype=torch.float32)
            done = is_terminal or is_truncated
            
            # Push SCALED reward to buffer
            agent.buffer.push(state, action, reward, next_state, is_terminal)
            
            state = next_state
            
            # Keep track of RAW reward for logging (so graphs look normal)
            cumulative_reward += raw_reward 
            episode_steps += 1
            global_step += 1
            
            if global_step % HYPERPARAMS["update_freq"] == 0:
                # Get detailed stats from update
                loss, q_val, grad = agent.update()
                ep_loss += loss
                ep_q += q_val
                ep_grad += grad
                update_counts += 1

        # Averages for this episode
        avg_loss = ep_loss / update_counts if update_counts > 0 else 0.0
        avg_q = ep_q / update_counts if update_counts > 0 else 0.0
        avg_grad = ep_grad / update_counts if update_counts > 0 else 0.0
        rps = cumulative_reward / episode_steps if episode_steps > 0 else 0
        current_lr = agent.get_current_lr()
        
        # Append to history
        history['rewards'].append(cumulative_reward)
        history['rps'].append(rps)
        history['loss'].append(avg_loss)
        history['epsilon'].append(agent.epsilon)
        history['q_values'].append(avg_q)
        history['grad_norms'].append(avg_grad)
        history['lr'].append(current_lr)
        
        # TensorBoard
        writer.add_scalar('Train/Reward', cumulative_reward, episode)
        writer.add_scalar('Train/Loss', avg_loss, episode)
        writer.add_scalar('Train/Epsilon', agent.epsilon, episode)
        writer.add_scalar('Train/Q_Value', avg_q, episode)
        writer.add_scalar('Train/Gradient_Norm', avg_grad, episode)
        writer.add_scalar('Train/LR', current_lr, episode)
        
        # Console Log
        episode_time = time.time() - episode_start_time
        elapsed_time = time.time() - training_start_time
        episodes_done = episode - start_episode + 1
        avg_time_per_episode = elapsed_time / episodes_done
        remaining_episodes = HYPERPARAMS["n_episodes"] - episode - 1
        eta_minutes = (remaining_episodes * avg_time_per_episode) / 60
        
        log_msg = (
            f"Ep {episode+1} | "
            f"Rw: {cumulative_reward:.1f} | "
            f"Loss: {avg_loss:.4f} | "
            f"Q: {avg_q:.2f} | "
            f"Grad: {avg_grad:.2f} | "
            f"Eps: {agent.epsilon:.3f} | "
            f"Time: {episode_time:.1f}s"
        )
        logger.log(log_msg)
        
        # Save & Plot
        if (episode + 1) % HYPERPARAMS["checkpoint_freq"] == 0:
            save_checkpoint(agent, episode, history, checkpoint_dir, f"checkpoint_ep{episode+1}.pth")
            save_checkpoint(agent, episode, history, checkpoint_dir, "latest_checkpoint.pth")
            save_all_plots(history, checkpoint_dir)
            logger.log(f"  ✓ Saved checkpoint & updated plots in {checkpoint_dir}/plots")

    # Final Save
    save_checkpoint(agent, HYPERPARAMS["n_episodes"]-1, history, checkpoint_dir, "final_model.pth")
    save_all_plots(history, checkpoint_dir)
    
    writer.close()
    logger.log("="*60)
    logger.log("Training Complete.")
    return checkpoint_dir

def main():
    resume_from = None
    if len(sys.argv) > 1:
        resume_from = sys.argv[1]
    
    checkpoint_dir = train(resume_from=resume_from)

if __name__ == "__main__":
    main()

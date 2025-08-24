"""
Evaluation script for World Models

This script evaluates trained World Models components and compares
performance with baseline PPO implementation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Dict, List
import time

# Import our modules
from config import get_config
from models.vae import ConvVAE
from models.mdnrnn import MDNRNN
from models.controller import Controller
from utils.environment import EnvironmentWrapper, visualize_rollout
from training.ppo_baseline import PPOAgent

def evaluate_world_models(
    config_or_path: str,
    num_episodes: int = 10,
    render: bool = False,
    save_videos: bool = False
) -> Dict[str, float]:
    """
    Evaluate trained World Models.
    
    Args:
        config_or_path: Path to config or experiment name
        num_episodes: Number of evaluation episodes
        render: Whether to render episodes
        save_videos: Whether to save evaluation videos
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load configuration
    if os.path.exists(config_or_path):
        # Load from file
        config = get_config(experiment_name=config_or_path)  # Simplified for now
    else:
        # Get default config
        config = get_config(experiment_name=config_or_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    print(f"Loading World Models for {config.environment}...")
    
    # VAE
    vae_path = f'./checkpoints/vae_{config.experiment_name}.pth'
    if not os.path.exists(vae_path):
        raise FileNotFoundError(f"VAE model not found: {vae_path}")
        
    vae = ConvVAE(
        latent_size=config.vae.latent_size,
        beta=config.vae.beta
    ).to(device)
    
    vae_checkpoint = torch.load(vae_path, map_location=device)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae.eval()
    
    # MDN-RNN
    mdnrnn_path = f'./checkpoints/mdnrnn_{config.experiment_name}.pth'
    if not os.path.exists(mdnrnn_path):
        raise FileNotFoundError(f"MDN-RNN model not found: {mdnrnn_path}")
        
    mdnrnn = MDNRNN(
        latent_size=config.mdnrnn.latent_size,
        action_size=config.mdnrnn.action_size,
        hidden_size=config.mdnrnn.hidden_size,
        num_mixtures=config.mdnrnn.num_mixtures
    ).to(device)
    
    mdnrnn_checkpoint = torch.load(mdnrnn_path, map_location=device)
    mdnrnn.load_state_dict(mdnrnn_checkpoint['model_state_dict'])
    mdnrnn.eval()
    
    # Controller
    controller_path = f'./checkpoints/controller_{config.experiment_name}.pth'
    if not os.path.exists(controller_path):
        raise FileNotFoundError(f"Controller model not found: {controller_path}")
        
    controller = Controller(
        input_size=config.controller.input_size,
        action_size=config.controller.action_size,
        hidden_sizes=config.controller.hidden_sizes,
        action_type='discrete'
    ).to(device)
    
    controller_checkpoint = torch.load(controller_path, map_location=device)
    controller.load_state_dict(controller_checkpoint['model_state_dict'])
    controller.eval()
    
    print("Models loaded successfully!")
    
    # Create environment
    env = EnvironmentWrapper(config.environment, render_mode='rgb_array')
    
    # Evaluation
    episode_rewards = []
    episode_lengths = []
    all_frames = []
    
    print(f"Evaluating for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Reset environment
        frame, info = env.reset()
        
        # Initialize LSTM hidden state
        hidden = None
        
        episode_reward = 0
        episode_length = 0
        episode_frames = []
        
        max_steps = 1000
        
        for step in range(max_steps):
            if render:
                env.render()
            
            if save_videos or (episode < 3):  # Save first few episodes
                episode_frames.append(frame.copy())
            
            with torch.no_grad():
                # Encode frame to latent
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
                z = vae.get_latent_representation(frame_tensor)
                
                # Get LSTM hidden state (simplified - would need proper state management)
                h = torch.zeros(1, config.mdnrnn.hidden_size, device=device)
                
                # Get action from controller
                action = controller.get_action(z, h, temperature=0.5, deterministic=False)
                
                if isinstance(action, torch.Tensor):
                    action = action.cpu().item()
            
            # Take environment step
            next_frame, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            frame = next_frame
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if episode_frames:
            all_frames.append(episode_frames)
        
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    env.close()
    
    # Save videos if requested
    if save_videos and all_frames:
        os.makedirs('./evaluation_videos', exist_ok=True)
        for i, frames in enumerate(all_frames[:5]):  # Save first 5 episodes
            video_path = f'./evaluation_videos/world_models_{config.environment.replace("/", "_")}_ep{i+1}.mp4'
            visualize_rollout(
                np.array(frames),
                save_path=video_path,
                fps=30
            )
    
    # Compute metrics
    metrics = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
    }
    
    print("\nWorld Models Evaluation Results:")
    print(f"  Mean Reward: {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f}")
    print(f"  Max Reward: {metrics['max_reward']:.3f}")
    print(f"  Mean Episode Length: {metrics['mean_length']:.1f}")
    
    return metrics

def evaluate_ppo_baseline(
    env_name: str,
    model_path: str,
    num_episodes: int = 10,
    render: bool = False
) -> Dict[str, float]:
    """
    Evaluate trained PPO baseline.
    
    Args:
        env_name: Environment name
        model_path: Path to PPO model
        num_episodes: Number of evaluation episodes
        render: Whether to render episodes
        
    Returns:
        Dictionary of evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading PPO baseline for {env_name}...")
    
    # Create PPO agent
    agent = PPOAgent(env_name=env_name, device=device)
    
    # Load model
    agent.load_model(model_path)
    
    print("PPO model loaded successfully!")
    
    # Evaluate
    episode_rewards = []
    episode_lengths = []
    
    print(f"Evaluating for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs, info = agent.env.reset()
        episode_reward = 0
        episode_length = 0
        
        max_steps = 1000
        
        for step in range(max_steps):
            if render:
                agent.env.render()
            
            obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                action, _, _, _ = agent.policy.get_action_and_value(obs_tensor)
                
            action_np = action.cpu().numpy().item()
            obs, reward, terminated, truncated, info = agent.env.step(action_np)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    agent.env.close()
    
    # Compute metrics
    metrics = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
    }
    
    print("\nPPO Baseline Evaluation Results:")
    print(f"  Mean Reward: {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f}")
    print(f"  Max Reward: {metrics['max_reward']:.3f}")
    print(f"  Mean Episode Length: {metrics['mean_length']:.1f}")
    
    return metrics

def compare_methods(
    env_name: str,
    world_models_experiment: str,
    ppo_model_path: str,
    num_episodes: int = 20
):
    """
    Compare World Models and PPO baseline performance.
    
    Args:
        env_name: Environment name
        world_models_experiment: World Models experiment name
        ppo_model_path: Path to PPO model
        num_episodes: Number of evaluation episodes
    """
    print("="*60)
    print("COMPARING WORLD MODELS VS PPO BASELINE")
    print("="*60)
    print(f"Environment: {env_name}")
    print(f"Episodes per method: {num_episodes}")
    print()
    
    # Evaluate World Models
    try:
        wm_metrics = evaluate_world_models(
            world_models_experiment,
            num_episodes=num_episodes,
            save_videos=True
        )
        wm_success = True
    except Exception as e:
        print(f"World Models evaluation failed: {e}")
        wm_metrics = {}
        wm_success = False
    
    print()
    
    # Evaluate PPO
    try:
        ppo_metrics = evaluate_ppo_baseline(
            env_name,
            ppo_model_path,
            num_episodes=num_episodes
        )
        ppo_success = True
    except Exception as e:
        print(f"PPO evaluation failed: {e}")
        ppo_metrics = {}
        ppo_success = False
    
    print()
    print("="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    if wm_success and ppo_success:
        print(f"{'Metric':<20} {'World Models':<15} {'PPO Baseline':<15} {'Difference':<15}")
        print("-" * 70)
        
        for metric in ['mean_reward', 'std_reward', 'mean_length']:
            wm_val = wm_metrics[metric]
            ppo_val = ppo_metrics[metric]
            diff = wm_val - ppo_val
            
            print(f"{metric:<20} {wm_val:<15.3f} {ppo_val:<15.3f} {diff:<15.3f}")
        
        # Determine winner
        if wm_metrics['mean_reward'] > ppo_metrics['mean_reward']:
            winner = "World Models"
            advantage = wm_metrics['mean_reward'] - ppo_metrics['mean_reward']
        else:
            winner = "PPO Baseline"
            advantage = ppo_metrics['mean_reward'] - wm_metrics['mean_reward']
        
        print()
        print(f"Winner: {winner} (advantage: {advantage:.3f} reward)")
        
        # Create comparison plot
        create_comparison_plot(wm_metrics, ppo_metrics, env_name)
        
    elif wm_success:
        print("Only World Models evaluation succeeded")
    elif ppo_success:
        print("Only PPO evaluation succeeded")
    else:
        print("Both evaluations failed")

def create_comparison_plot(
    wm_metrics: Dict[str, float],
    ppo_metrics: Dict[str, float],
    env_name: str
):
    """Create comparison visualization"""
    metrics_to_plot = ['mean_reward', 'max_reward', 'mean_length']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics_to_plot):
        wm_val = wm_metrics[metric]
        ppo_val = ppo_metrics[metric]
        
        bars = axes[i].bar(['World Models', 'PPO'], [wm_val, ppo_val], 
                         color=['skyblue', 'lightcoral'])
        
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].set_ylabel('Value')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')
    
    plt.suptitle(f'Performance Comparison - {env_name}')
    plt.tight_layout()
    
    # Save plot
    os.makedirs('./evaluation_results', exist_ok=True)
    plt.savefig(f'./evaluation_results/comparison_{env_name.replace("/", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate World Models')
    parser.add_argument('--env', type=str, default='ALE/Pong-v5',
                       help='Environment name')
    parser.add_argument('--method', type=str, choices=['world_models', 'ppo', 'compare'], 
                       default='compare', help='Evaluation method')
    parser.add_argument('--wm-experiment', type=str, default=None,
                       help='World Models experiment name')
    parser.add_argument('--ppo-model', type=str, default=None,
                       help='PPO model path')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes')
    parser.add_argument('--save-videos', action='store_true',
                       help='Save evaluation videos')
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    if args.wm_experiment is None:
        args.wm_experiment = f"world_models_{args.env.replace('/', '_')}"
    
    if args.ppo_model is None:
        args.ppo_model = f"./checkpoints/ppo_{args.env.replace('/', '_')}.pth"
    
    print(f"Evaluation settings:")
    print(f"  Environment: {args.env}")
    print(f"  Method: {args.method}")
    print(f"  Episodes: {args.episodes}")
    print(f"  World Models experiment: {args.wm_experiment}")
    print(f"  PPO model: {args.ppo_model}")
    print()
    
    if args.method == 'world_models':
        evaluate_world_models(
            args.wm_experiment,
            num_episodes=args.episodes,
            render=args.render,
            save_videos=args.save_videos
        )
    elif args.method == 'ppo':
        evaluate_ppo_baseline(
            args.env,
            args.ppo_model,
            num_episodes=args.episodes,
            render=args.render
        )
    elif args.method == 'compare':
        compare_methods(
            args.env,
            args.wm_experiment,
            args.ppo_model,
            num_episodes=args.episodes
        )

if __name__ == "__main__":
    main()

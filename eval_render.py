#!/usr/bin/env python3
"""
Visual Environment Evaluation with Real-Time Rendering
"""

import sys
import os
import time
import argparse
import random
from typing import Optional, Tuple, Any, Callable
import warnings

# Suppress gymnasium warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)

def check_cuda_availability() -> None:
    """Check and display CUDA availability and GPU information."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"CUDA Available: {cuda_available}")
            print(f"GPU: {gpu_name}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")
        else:
            print(f"CUDA Available: {cuda_available}")
            print(f"Using CPU for computations")
    except ImportError:
        print("PyTorch not available - running in basic mode")

def pick_env_name(args) -> str:
    """Pick environment name based on arguments and curriculum availability."""
    if args.env:
        return args.env
    
    # Import curriculum resolver
    try:
        sys.path.append(os.path.join(os.getcwd(), 'world_models'))
        from curriculum_trainer_visual import resolve_curriculum
        
        print("No environment specified, selecting from curriculum...")
        curriculum = resolve_curriculum(args.prefer_box2d)
        
        if not curriculum:
            print("ERROR: No environments available from curriculum")
            sys.exit(1)
            
        env_name = curriculum[0][0]  # First environment from curriculum
        print(f"Selected environment: {env_name}")
        return env_name
        
    except ImportError as e:
        print(f"Could not import curriculum system: {e}")
        print("Falling back to CartPole-v1")
        return "CartPole-v1"

def provide_install_hint(env_id: str, error: Exception) -> None:
    """Provide helpful installation hints when environment creation fails."""
    env_id_lower = env_id.lower()
    
    print(f"Failed to create environment '{env_id}'")
    print(f"   Error: {error}")
    
    if env_id_lower.startswith("ale/") or "atari" in env_id_lower:
        print("   Try: pip install \"gymnasium[atari,accept-roms]\" ale-py autorom")
    elif any(box2d_env in env_id_lower for box2d_env in ["lunarlander", "carracing", "bipedal"]):
        print("   Try: pip install swig && pip install \"gymnasium[box2d]\"")
    elif "mujoco" in env_id_lower:
        print("   Try: pip install \"gymnasium[mujoco]\"")
    else:
        print("   Try: pip install gymnasium")
    print()

def make_env(env_name: str) -> Tuple[Any, str]:
    """Create environment with rendering support."""
    import gymnasium as gym
    
    # Try gymnasium native rendering first
    try:
        env = gym.make(env_name, render_mode="human")
        print(f"Created environment with native rendering: {env_name}")
        return env, "human"
        
    except Exception as e:
        print(f"Native rendering failed for {env_name}: {e}")
        provide_install_hint(env_name, e)
        
        # Try rgb_array mode for OpenCV fallback
        try:
            env = gym.make(env_name, render_mode="rgb_array")
            print(f"Created environment with rgb_array rendering: {env_name}")
            return env, "rgb_array"
                
        except Exception as e:
            print(f"Failed to create environment with any render mode")
            provide_install_hint(env_name, e)
            sys.exit(1)

def create_random_policy(env) -> Callable:
    """Create a random policy for the given environment."""
    def random_policy(obs):
        return env.action_space.sample()
    
    print("Using random policy")
    return random_policy

def rollout_episode(env, policy, fps: int, episode_num: int, render_mode: str) -> Tuple[float, int]:
    """Run a single episode with real-time rendering."""
    frame_time = 1.0 / fps if fps > 0 else 0
    
    # Reset environment
    try:
        obs, info = env.reset()
    except TypeError:
        obs = env.reset()
        info = {}
    
    total_reward = 0.0
    steps = 0
    done = False
    truncated = False
    
    print(f"Episode {episode_num}: Starting rollout...")
    
    try:
        while not (done or truncated):
            start_time = time.time()
            
            # Get action from policy
            action = policy(obs)
            
            # Take step in environment
            try:
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, done, truncated, info = step_result
                elif len(step_result) == 4:
                    obs, reward, done, info = step_result
                    truncated = False
                else:
                    obs, reward, done = step_result
                    info = {}
                    truncated = False
            except Exception as e:
                print(f"Step error: {e}")
                break
            
            total_reward += reward
            steps += 1
            
            # Render
            if render_mode == "human":
                try:
                    env.render()
                except Exception as e:
                    print(f"Render error: {e}")
            
            # FPS throttling
            if frame_time > 0:
                elapsed = time.time() - start_time
                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            # Safety check for very long episodes
            if steps > 10000:
                print("Episode too long, stopping")
                break
                
    except KeyboardInterrupt:
        print("\nEpisode interrupted by user")
    
    print(f"Episode {episode_num}: {steps} steps, reward: {total_reward:.2f}")
    return total_reward, steps

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Visual Environment Evaluation with Real-Time Rendering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py -3.12 eval_render_new.py --env CarRacing-v3 --episodes 2 --fps 30 --prefer-box2d true  
  py -3.12 eval_render_new.py --env CartPole-v1 --episodes 3 --fps 60
  py -3.12 eval_render_new.py --prefer-box2d false --episodes 1
        """
    )
    
    # Environment selection
    parser.add_argument('--env', type=str, default=None,
                       help='Environment name (default: auto-select from curriculum)')
    parser.add_argument('--prefer-box2d', choices=['true', 'false'], default='false',
                       help='Prefer Box2D environments when auto-selecting (default: false)')
    
    # Evaluation parameters
    parser.add_argument('--episodes', type=int, default=2,
                       help='Number of episodes to run (default: 2)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS for rendering (default: 30)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    # Policy and device
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint for policy (default: random policy)')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto',
                       help='Device for model inference (default: auto)')
    
    args = parser.parse_args()
    
    # Convert string booleans
    args.prefer_box2d = args.prefer_box2d == 'true'
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        try:
            import numpy as np
            np.random.seed(args.seed)
        except ImportError:
            pass
        try:
            import torch
            torch.manual_seed(args.seed)
        except ImportError:
            pass
    
    print("Visual Environment Evaluation")
    print("=" * 50)
    
    # Display system information
    check_cuda_availability()
    print()
    
    # Pick environment
    try:
        env_name = pick_env_name(args)
    except SystemExit:
        return
    
    print(f"Target Environment: {env_name}")
    print(f"Episodes: {args.episodes}")
    print(f"Target FPS: {args.fps}")
    if args.seed is not None:
        print(f"Random Seed: {args.seed}")
    print()
    
    # Create environment
    try:
        env, render_mode = make_env(env_name)
    except SystemExit:
        return
    
    # Create policy
    policy = create_random_policy(env)
    print()
    
    # Run episodes
    total_rewards = []
    total_steps = []
    
    try:
        for episode in range(1, args.episodes + 1):
            reward, steps = rollout_episode(env, policy, args.fps, episode, render_mode)
            total_rewards.append(reward)
            total_steps.append(steps)
            print()
            
            # Brief pause between episodes
            if episode < args.episodes:
                print("Brief pause before next episode...")
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    
    finally:
        # Clean up
        try:
            env.close()
        except:
            pass
    
    # Final statistics
    if total_rewards:
        print("Final Statistics:")
        print("=" * 30)
        print(f"Episodes completed: {len(total_rewards)}")
        print(f"Average reward: {sum(total_rewards) / len(total_rewards):.2f}")
        print(f"Average steps: {sum(total_steps) / len(total_steps):.1f}")
        print(f"Total reward: {sum(total_rewards):.2f}")
        print(f"Total steps: {sum(total_steps)}")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()

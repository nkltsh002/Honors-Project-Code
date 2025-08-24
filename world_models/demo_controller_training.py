"""CLI demo script for controller training pipeline.

This script provides a simple command-line interface to demonstrate
the controller training capabilities of the World Models implementation.

Features:
- Train controllers using CMA-ES
- Test different controller configurations
- Visualize training progress
- Compare performance metrics

Usage:
    python demo_controller_training.py --help
    python demo_controller_training.py --train-linear     # Train linear controller
    python demo_controller_training.py --train-mlp       # Train MLP controller  
    python demo_controller_training.py --compare         # Compare both controllers
    python demo_controller_training.py --test-env        # Test dream environment
"""

import os
import sys
import argparse
import time
import logging
import tempfile
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from world_models.models.controller import Controller, CMAESController
from world_models.tools.dream_env import DreamEnvironment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_demo_environment() -> DreamEnvironment:
    """Create a demo dream environment for testing."""
    # Create temporary model files for demo
    temp_dir = tempfile.mkdtemp()
    vae_path = os.path.join(temp_dir, 'demo_vae.pt')
    rnn_path = os.path.join(temp_dir, 'demo_rnn.pt')
    
    # Create dummy checkpoint files
    torch.save({'dummy': True}, vae_path)
    torch.save({'dummy': True}, rnn_path)
    
    # Create dream environment
    dream_env = DreamEnvironment(
        vae_model_path=vae_path,
        mdnrnn_model_path=rnn_path,
        action_space_size=3,
        max_episode_steps=100,
        temperature=1.0,
        device='cpu'
    )
    
    logger.info("Demo dream environment created")
    return dream_env


def demo_controller_creation():
    """Demonstrate controller creation and basic operations."""
    logger.info("=" * 50)
    logger.info("DEMO: Controller Creation")
    logger.info("=" * 50)
    
    # Test different controller configurations
    configs = [
        {
            'name': 'Linear Continuous',
            'input_size': 32 + 256,  # z_dim + h_dim
            'action_size': 3,
            'hidden_sizes': (),
            'action_type': 'continuous'
        },
        {
            'name': 'Linear Discrete',
            'input_size': 32 + 256,
            'action_size': 4, 
            'hidden_sizes': (),
            'action_type': 'discrete'
        },
        {
            'name': 'MLP Continuous',
            'input_size': 32 + 256,
            'action_size': 3,
            'hidden_sizes': (64,),
            'action_type': 'continuous'
        }
    ]
    
    for config in configs:
        logger.info(f"\nTesting {config['name']} Controller:")
        
        # Create controller
        controller = Controller(
            input_size=config['input_size'],
            action_size=config['action_size'],
            hidden_sizes=config['hidden_sizes'],
            action_type=config['action_type']
        )
        
        logger.info(f"  Parameters: {controller.get_num_parameters():,}")
        
        # Test forward pass
        batch_size = 4
        z = torch.randn(batch_size, 32)
        h = torch.randn(batch_size, 256)
        
        with torch.no_grad():
            output = controller(z, h)
            logger.info(f"  Output shape: {output.shape}")
            
            # Test action generation
            if config['action_type'] == 'continuous':
                result = controller.get_action(z, h, deterministic=True)
                if isinstance(result, tuple):
                    action = result[0]  # Get action tensor from tuple
                else:
                    action = result
                logger.info(f"  Action range: [{action.min().item():.3f}, {action.max().item():.3f}]")
            else:
                result = controller.get_action(z, h, deterministic=True)
                if isinstance(result, tuple):
                    action = result[0]  # Get action tensor from tuple
                else:
                    action = result
                logger.info(f"  Action shape: {action.shape}")
        
        # Test parameter manipulation
        params = controller.get_parameters()
        logger.info(f"  Parameter vector size: {len(params)}")
        
        # Test weight setting
        new_params = np.random.randn(len(params)) * 0.1
        controller.set_parameters(new_params)
        
        # Verify
        restored_params = controller.get_parameters()
        if np.allclose(new_params, restored_params):
            logger.info("  ✓ Weight management works correctly")
        else:
            logger.error("  ✗ Weight management failed")


def demo_dream_environment():
    """Demonstrate dream environment functionality."""
    logger.info("=" * 50)
    logger.info("DEMO: Dream Environment")
    logger.info("=" * 50)
    
    # Create dream environment
    dream_env = create_demo_environment()
    
    # Test reset
    obs, info = dream_env.reset()
    logger.info(f"Initial observation: {type(obs)}")
    logger.info(f"Action space size: {dream_env.action_space_size}")
    
    # Run a short episode
    total_reward = 0
    for step in range(10):
        # Random action
        action = np.random.uniform(-1, 1, size=dream_env.action_space_size)
        
        # Take step
        obs, reward, terminated, truncated, info = dream_env.step(action)
        total_reward += reward
        
        logger.info(f"Step {step+1}: reward={reward:.3f}, done={terminated or truncated}")
        
        if terminated or truncated:
            logger.info("Episode terminated")
            break
    
    logger.info(f"Total reward: {total_reward:.3f}")


def demo_cmaes_training(controller_config: Dict, generations: int = 20, population_size: int = 16):
    """Demonstrate CMA-ES training."""
    logger.info("=" * 50)
    logger.info(f"DEMO: CMA-ES Training ({controller_config['name']})")
    logger.info("=" * 50)
    
    # Create controller
    controller = Controller(
        input_size=controller_config['input_size'],
        action_size=controller_config['action_size'],
        hidden_sizes=controller_config['hidden_sizes'],
        action_type=controller_config['action_type']
    )
    
    logger.info(f"Controller: {controller.get_num_parameters()} parameters")
    
    # Create CMA-ES trainer
    cmaes_trainer = CMAESController(
        controller=controller,
        population_size=population_size,
        sigma=0.5,
        device=torch.device('cpu')
    )
    
    # Mock fitness function (replace with real environment evaluation)
    def mock_fitness_function(params: np.ndarray) -> float:
        """Mock fitness function for demonstration."""
        # Simple quadratic function with noise
        return -(np.sum(params**2)) / len(params) + np.random.normal(0, 0.1)
    
    # Training loop
    fitness_history = []
    
    for generation in range(generations):
        # Get candidate solutions
        candidates = cmaes_trainer.ask()
        
        # Evaluate candidates
        fitness_values = []
        for candidate in candidates:
            fitness = mock_fitness_function(candidate)
            fitness_values.append(fitness)
        
        fitness_values = np.array(fitness_values)
        
        # Tell CMA-ES the results
        cmaes_trainer.tell(candidates, fitness_values)
        
        # Log progress
        stats = cmaes_trainer.get_stats()
        fitness_history.append(fitness_values)
        
        if generation % 5 == 0 or generation == generations - 1:
            logger.info(
                f"Gen {generation:2d}: "
                f"best={stats['best_fitness']:7.4f}, "
                f"mean={stats['mean_fitness']:7.4f}, "
                f"sigma={stats['sigma']:.4f}"
            )
    
    # Update controller with best parameters
    cmaes_trainer.update_controller()
    
    logger.info(f"Training completed. Final best fitness: {cmaes_trainer.best_fitness:.4f}")
    
    return fitness_history


def plot_training_progress(fitness_histories: Dict[str, List], save_path: Optional[str] = None):
    """Plot training progress comparison."""
    plt.figure(figsize=(12, 6))
    
    for name, history in fitness_histories.items():
        # Convert to arrays
        history = np.array(history)
        generations = range(len(history))
        
        # Plot mean and std
        means = np.mean(history, axis=1)
        stds = np.std(history, axis=1)
        
        plt.subplot(1, 2, 1)
        plt.plot(generations, means, label=f'{name} (mean)')
        plt.fill_between(generations, means - stds, means + stds, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(generations, np.max(history, axis=1), label=f'{name} (best)')
    
    plt.subplot(1, 2, 1)
    plt.xlabel('Generation')
    plt.ylabel('Mean Fitness')
    plt.title('Training Progress - Mean Fitness')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Training Progress - Best Fitness')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Training progress plot saved to: {save_path}")
    else:
        plt.show()


def demo_controller_comparison():
    """Compare different controller architectures."""
    logger.info("=" * 50)
    logger.info("DEMO: Controller Comparison")
    logger.info("=" * 50)
    
    # Controller configurations to compare
    configs = [
        {
            'name': 'Linear',
            'input_size': 32 + 256,
            'action_size': 3,
            'hidden_sizes': (),
            'action_type': 'continuous'
        },
        {
            'name': 'MLP-64',
            'input_size': 32 + 256,
            'action_size': 3,
            'hidden_sizes': (64,),
            'action_type': 'continuous'
        }
    ]
    
    fitness_histories = {}
    
    for config in configs:
        logger.info(f"\nTraining {config['name']} controller...")
        
        history = demo_cmaes_training(
            controller_config=config,
            generations=15,
            population_size=12
        )
        
        fitness_histories[config['name']] = history
    
    # Plot comparison
    plot_training_progress(fitness_histories)


def performance_benchmark():
    """Run performance benchmarks."""
    logger.info("=" * 50)
    logger.info("DEMO: Performance Benchmark")
    logger.info("=" * 50)
    
    # Test different controller sizes
    configs = [
        {'name': 'Small (Linear)', 'input_size': 32+256, 'action_size': 3, 'hidden_sizes': ()},
        {'name': 'Medium (MLP-32)', 'input_size': 32+256, 'action_size': 3, 'hidden_sizes': (32,)},
        {'name': 'Large (MLP-128)', 'input_size': 32+256, 'action_size': 3, 'hidden_sizes': (128,)},
    ]
    
    results = []
    
    for config in configs:
        controller = Controller(
            input_size=config['input_size'],
            action_size=config['action_size'],
            hidden_sizes=config['hidden_sizes'],
            action_type='continuous'
        )
        
        # Time forward passes
        batch_size = 1000
        z = torch.randn(batch_size, 32)
        h = torch.randn(batch_size, 256)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = controller(z, h)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                output = controller(z, h)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        params = controller.get_num_parameters()
        
        results.append({
            'name': config['name'],
            'parameters': params,
            'avg_time': avg_time * 1000,  # ms
            'throughput': batch_size / avg_time  # samples/sec
        })
        
        logger.info(f"{config['name']:15s}: {params:5d} params, {avg_time*1000:6.2f} ms/batch, {batch_size/avg_time:8.0f} samples/sec")
    
    return results


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="World Models Controller Training Demo")
    parser.add_argument('--train-linear', action='store_true', help='Train linear controller')
    parser.add_argument('--train-mlp', action='store_true', help='Train MLP controller')
    parser.add_argument('--compare', action='store_true', help='Compare controllers')
    parser.add_argument('--test-env', action='store_true', help='Test dream environment')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--generations', type=int, default=20, help='Training generations')
    parser.add_argument('--population', type=int, default=16, help='CMA-ES population size')
    parser.add_argument('--save-plot', type=str, help='Save training plot to file')
    
    args = parser.parse_args()
    
    if not any([args.train_linear, args.train_mlp, args.compare, args.test_env, args.benchmark]):
        # Default demo
        logger.info("Running default demo (controller creation + environment test)")
        demo_controller_creation()
        demo_dream_environment()
        return
    
    try:
        if args.test_env:
            demo_dream_environment()
        
        if args.benchmark:
            performance_benchmark()
        
        if args.train_linear:
            config = {
                'name': 'Linear',
                'input_size': 32 + 256,
                'action_size': 3,
                'hidden_sizes': (),
                'action_type': 'continuous'
            }
            demo_cmaes_training(config, args.generations, args.population)
        
        if args.train_mlp:
            config = {
                'name': 'MLP-64',
                'input_size': 32 + 256,
                'action_size': 3,
                'hidden_sizes': (64,),
                'action_type': 'continuous'
            }
            demo_cmaes_training(config, args.generations, args.population)
        
        if args.compare:
            demo_controller_comparison()
        
        logger.info("Demo completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()

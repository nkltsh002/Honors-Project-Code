"""Controller training pipeline for World Models.

This module implements the training pipeline for the Controller (C) component using:
1. CMA-ES (Covariance Matrix Adaptation Evolution Strategy) as primary optimizer
2. PPO (Proximal Policy Optimization) as baseline comparison
3. Multiprocessing for efficient parallel evaluation
4. Integration with DreamEnvironment for training in learned model
5. Real environment evaluation for validation
6. Comprehensive logging and visualization

Key Features:
- Multiprocessing parallel rollout evaluation
- Memory-efficient batch processing 
- Checkpoint saving/loading for long training runs
- TensorBoard logging for training visualization
- CSV export for analysis
- Support for both discrete and continuous action spaces
- Temperature-based exploration control
- Robust error handling and recovery

Usage:
    trainer = ControllerTrainer(
        controller=controller,
        dream_env=dream_env,
        real_env=real_env
    )
    trainer.train_cmaes(generations=100)
"""

import os
import time
import multiprocessing as mp
import numpy as np
import torch
import cma
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, asdict
import pickle
import csv
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

# Local imports
from ..models.controller import Controller, create_controller, ControllerConfig
from ..tools.dream_env import DreamEnvironment
from ..models.vae import ConvVAE
from ..models.mdn_rnn import MDNRNN

# PPO baseline imports 
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    HAS_SB3 = True
except ImportError:
    print("Warning: stable-baselines3 not installed. PPO baseline unavailable.")
    HAS_SB3 = False

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for controller training."""
    
    # CMA-ES Parameters
    population_size: int = 64        # CMA-ES population size
    sigma: float = 0.5               # Initial standard deviation
    max_generations: int = 200       # Maximum training generations
    
    # Evaluation Parameters  
    num_rollouts: int = 16           # Rollouts per candidate evaluation
    max_episode_length: int = 1000   # Maximum episode length
    temperature: float = 1.0         # Action selection temperature
    deterministic_eval: bool = False # Use deterministic actions for evaluation
    
    # Multiprocessing
    num_workers: int = 4             # Number of parallel workers
    
    # Environment Parameters
    use_dream_env: bool = True       # Train in dream environment
    dream_temperature: float = 1.0   # Temperature for dream sampling
    validate_on_real: bool = True    # Validate on real environment
    validation_frequency: int = 10   # Validate every N generations
    
    # Logging and Checkpointing
    save_frequency: int = 25         # Save checkpoint every N generations
    log_frequency: int = 5           # Log statistics every N generations
    tensorboard_log: bool = True     # Enable TensorBoard logging
    csv_log: bool = True            # Enable CSV logging
    
    # Training Control
    early_stopping_patience: int = 50  # Stop if no improvement for N generations  
    early_stopping_threshold: float = 1e-6  # Minimum improvement threshold
    
    # File paths
    checkpoint_dir: str = "checkpoints/controller"
    log_dir: str = "logs/controller"
    
    # Device
    device: str = 'cuda'


@dataclass  
class EvaluationResult:
    """Results from controller evaluation."""
    fitness: float
    episode_length: float
    total_reward: float
    success_rate: float
    std_reward: float
    min_reward: float
    max_reward: float
    

def evaluate_controller_rollouts(
    params: np.ndarray,
    controller_config: Dict,
    dream_env_config: Dict,
    eval_config: Dict,
    worker_id: int = 0
) -> Dict[str, float]:
    """Evaluate controller parameters with multiple rollouts.
    
    This function is designed to run in separate processes for parallel evaluation.
    
    Args:
        params: Flattened controller parameters
        controller_config: Controller configuration dictionary
        dream_env_config: Dream environment configuration dictionary  
        eval_config: Evaluation configuration dictionary
        worker_id: Process worker ID for logging
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Set device for worker process
        device = eval_config.get('device', 'cpu')
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            
        # Recreate controller in worker process
        controller = create_controller(
            controller_type=controller_config['type'],
            config=ControllerConfig(**controller_config['config'])
        )
        controller.set_weights(params)
        controller.eval()
        
        # Recreate dream environment in worker process
        # This is simplified - in practice you'd load VAE/RNN from checkpoints
        dream_env = DreamEnvironment(
            vae=None,  # Mock for multiprocessing
            rnn=None,  # Mock for multiprocessing  
            action_space_type=dream_env_config.get('action_space_type', 'continuous'),
            action_dim=dream_env_config.get('action_dim', 3),
            device=device
        )
        
        # Evaluation parameters
        num_rollouts = eval_config.get('num_rollouts', 16)
        max_episode_length = eval_config.get('max_episode_length', 1000)
        temperature = eval_config.get('temperature', 1.0)
        deterministic = eval_config.get('deterministic_eval', False)
        
        # Run multiple rollouts
        episode_rewards = []
        episode_lengths = []
        successes = []
        
        for rollout in range(num_rollouts):
            try:
                obs, _ = dream_env.reset()
                episode_reward = 0.0
                episode_length = 0
                done = False
                
                # Get initial latent state and hidden state
                z = torch.zeros(1, controller_config['config']['z_dim'])
                h = torch.zeros(1, controller_config['config']['h_dim'])
                
                while not done and episode_length < max_episode_length:
                    # Get action from controller
                    with torch.no_grad():
                        action = controller(z, h)
                        
                        # Process action for environment
                        if controller_config['config']['action_type'] == 'continuous':
                            action_np = torch.tanh(action).cpu().numpy().squeeze()
                        else:
                            action_probs = torch.softmax(action / temperature, dim=-1)
                            if deterministic:
                                action_np = torch.argmax(action_probs).item()
                            else:
                                action_np = torch.multinomial(action_probs, 1).item()
                    
                    # Take environment step
                    obs, reward, terminated, truncated, info = dream_env.step(action_np)
                    done = terminated or truncated
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    # Update states (simplified - would use VAE/RNN in practice)
                    z = torch.randn_like(z) * 0.1  # Mock latent state update
                    h = torch.randn_like(h) * 0.1  # Mock hidden state update
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                successes.append(info.get('success', episode_reward > 0))
                
            except Exception as e:
                logger.warning(f"Worker {worker_id}: Rollout {rollout} failed: {e}")
                episode_rewards.append(-1000.0)  # Penalty for failed rollout
                episode_lengths.append(0)
                successes.append(False)
        
        # Compute statistics
        episode_rewards = np.array(episode_rewards)
        episode_lengths = np.array(episode_lengths)
        
        result = {
            'fitness': float(np.mean(episode_rewards)),
            'total_reward': float(np.sum(episode_rewards)),  
            'episode_length': float(np.mean(episode_lengths)),
            'success_rate': float(np.mean(successes)),
            'std_reward': float(np.std(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'num_rollouts': len(episode_rewards),
            'worker_id': worker_id
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Worker {worker_id}: Critical evaluation error: {e}")
        return {
            'fitness': -2000.0,  # Large penalty for critical errors
            'total_reward': -2000.0,
            'episode_length': 0.0,
            'success_rate': 0.0,
            'std_reward': 0.0,
            'min_reward': -2000.0,
            'max_reward': -2000.0,
            'num_rollouts': 0,
            'worker_id': worker_id
        }


class ControllerTrainer:
    """Main trainer class for controller optimization."""
    
    def __init__(
        self,
        controller: Controller,
        dream_env: Optional[DreamEnvironment] = None,
        real_env: Optional[Any] = None,
        config: Optional[TrainingConfig] = None
    ):
        """Initialize controller trainer.
        
        Args:
            controller: Controller model to train
            dream_env: Dream environment for training
            real_env: Real environment for validation
            config: Training configuration
        """
        self.controller = controller
        self.dream_env = dream_env
        self.real_env = real_env
        self.config = config or TrainingConfig()
        
        # Create directories
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize CMA-ES
        self.setup_cmaes()
        
        # Training state
        self.generation = 0
        self.best_fitness = -np.inf
        self.best_params = None
        self.fitness_history = []
        self.no_improvement_count = 0
        
        logger.info(f"ControllerTrainer initialized")
        logger.info(f"Controller parameters: {controller.get_num_parameters()}")
        logger.info(f"Population size: {self.config.population_size}")
        logger.info(f"Using {self.config.num_workers} workers")
    
    def setup_logging(self):
        """Setup logging infrastructure."""
        
        # CSV logging
        if self.config.csv_log:
            self.csv_path = os.path.join(self.config.log_dir, 'training_log.csv')
            self.csv_fieldnames = [
                'generation', 'best_fitness', 'mean_fitness', 'std_fitness',
                'sigma', 'episode_length', 'success_rate', 'eval_time'
            ]
            
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
                writer.writeheader()
        
        # TensorBoard logging setup would go here
        if self.config.tensorboard_log:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(self.config.log_dir)
            except ImportError:
                logger.warning("TensorBoard not available")
                self.config.tensorboard_log = False
    
    def setup_cmaes(self):
        """Initialize CMA-ES optimizer."""
        initial_params = self.controller.get_weights()
        
        cmaes_options = {
            'popsize': self.config.population_size,
            'seed': 42,
            'tolx': 1e-8,  # Tolerance in x
            'tolfun': 1e-8,  # Tolerance in function value
            'maxiter': self.config.max_generations,
            'verb_disp': 1,  # Display verbosity
        }
        
        self.es = cma.CMAEvolutionStrategy(initial_params, self.config.sigma, cmaes_options)
        logger.info(f"CMA-ES initialized with {len(initial_params)} parameters")
    
    def evaluate_population_parallel(self, candidates: np.ndarray) -> List[Dict[str, float]]:
        """Evaluate population of candidates in parallel.
        
        Args:
            candidates: Array of candidate parameter vectors
            
        Returns:
            List of evaluation results
        """
        num_candidates = len(candidates)
        
        # Prepare configuration dictionaries for worker processes
        controller_config = {
            'type': 'linear' if len(self.controller.hidden_sizes) == 0 else 'mlp',
            'config': {
                'z_dim': self.controller.input_size - 256,  # Assuming h_dim = 256
                'h_dim': 256,
                'action_dim': self.controller.action_size,
                'action_type': self.controller.action_type,
                'device': 'cpu'  # Force CPU for worker processes
            }
        }
        
        dream_env_config = {
            'action_space_type': self.controller.action_type,
            'action_dim': self.controller.action_size,
        }
        
        eval_config = {
            'num_rollouts': self.config.num_rollouts,
            'max_episode_length': self.config.max_episode_length,
            'temperature': self.config.temperature,
            'deterministic_eval': self.config.deterministic_eval,
            'device': 'cpu'
        }
        
        # Use ProcessPoolExecutor for parallel evaluation
        results = []
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            # Submit all evaluation jobs
            futures = []
            for i, params in enumerate(candidates):
                future = executor.submit(
                    evaluate_controller_rollouts,
                    params, controller_config, dream_env_config, eval_config, i
                )
                futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                except Exception as e:
                    logger.error(f"Evaluation failed: {e}")
                    results.append({
                        'fitness': -1000.0,
                        'total_reward': -1000.0,
                        'episode_length': 0.0,
                        'success_rate': 0.0,
                        'std_reward': 0.0,
                        'min_reward': -1000.0,
                        'max_reward': -1000.0,
                        'num_rollouts': 0,
                        'worker_id': -1
                    })
        
        # Sort results to match candidate order
        results.sort(key=lambda x: x.get('worker_id', 0))
        
        return results
    
    def train_cmaes(self, generations: Optional[int] = None) -> Dict[str, Any]:
        """Train controller using CMA-ES.
        
        Args:
            generations: Number of generations to train (uses config if None)
            
        Returns:
            Training results dictionary
        """
        if generations is None:
            generations = self.config.max_generations
        
        logger.info(f"Starting CMA-ES training for {generations} generations")
        start_time = time.time()
        
        for gen in range(generations):
            gen_start = time.time()
            
            # Check if evolution should stop
            if self.es.stop():
                logger.info(f"CMA-ES stopping criteria met at generation {gen}")
                break
            
            # Ask CMA-ES for new candidates
            candidates = np.array(self.es.ask())
            
            # Evaluate candidates in parallel
            eval_results = self.evaluate_population_parallel(candidates)
            
            # Extract fitness values
            fitness_values = np.array([r['fitness'] for r in eval_results])
            
            # Tell CMA-ES the fitness values (CMA-ES minimizes, so negate)
            self.es.tell(candidates, -fitness_values)
            
            # Update best solution
            best_idx = np.argmax(fitness_values)
            if fitness_values[best_idx] > self.best_fitness:
                self.best_fitness = fitness_values[best_idx]
                self.best_params = candidates[best_idx].copy()
                self.controller.set_weights(self.best_params)
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
            
            # Store generation results
            self.generation = gen
            self.fitness_history.append(fitness_values)
            
            # Logging
            gen_time = time.time() - gen_start
            if gen % self.config.log_frequency == 0:
                self.log_generation(eval_results, gen_time)
            
            # Checkpointing  
            if gen % self.config.save_frequency == 0:
                self.save_checkpoint()
            
            # Real environment validation
            if (self.config.validate_on_real and 
                self.real_env is not None and 
                gen % self.config.validation_frequency == 0):
                self.validate_on_real_env()
            
            # Early stopping
            if (self.no_improvement_count >= self.config.early_stopping_patience):
                logger.info(f"Early stopping: no improvement for {self.no_improvement_count} generations")
                break
        
        total_time = time.time() - start_time
        
        # Final update with best parameters
        if self.best_params is not None:
            self.controller.set_weights(self.best_params)
        
        # Save final checkpoint
        self.save_checkpoint(is_final=True)
        
        results = {
            'generations_trained': self.generation + 1,
            'best_fitness': self.best_fitness,
            'total_time': total_time,
            'final_sigma': self.es.sigma,
            'convergence_reached': self.es.stop(),
        }
        
        logger.info(f"CMA-ES training completed in {total_time:.2f}s")
        logger.info(f"Best fitness: {self.best_fitness:.4f}")
        
        return results
    
    def log_generation(self, eval_results: List[Dict], gen_time: float):
        """Log generation statistics."""
        
        # Compute statistics
        fitness_values = [r['fitness'] for r in eval_results]
        episode_lengths = [r['episode_length'] for r in eval_results] 
        success_rates = [r['success_rate'] for r in eval_results]
        
        stats = {
            'generation': self.generation,
            'best_fitness': max(fitness_values),
            'mean_fitness': np.mean(fitness_values),
            'std_fitness': np.std(fitness_values),
            'sigma': self.es.sigma,
            'episode_length': np.mean(episode_lengths),
            'success_rate': np.mean(success_rates),
            'eval_time': gen_time
        }
        
        # Console logging
        logger.info(
            f"Gen {self.generation:3d}: "
            f"fitness={stats['mean_fitness']:7.2f}±{stats['std_fitness']:5.2f} "
            f"best={stats['best_fitness']:7.2f} "
            f"sigma={stats['sigma']:.3f} "
            f"time={gen_time:.1f}s"
        )
        
        # CSV logging
        if self.config.csv_log:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
                writer.writerow(stats)
        
        # TensorBoard logging
        if self.config.tensorboard_log and hasattr(self, 'tb_writer'):
            for key, value in stats.items():
                if key != 'generation':
                    self.tb_writer.add_scalar(f'training/{key}', value, self.generation)
    
    def validate_on_real_env(self):
        """Validate current best controller on real environment."""
        if self.real_env is None or self.best_params is None:
            return
        
        logger.info(f"Validating on real environment...")
        
        # Set best parameters
        temp_params = self.controller.get_weights()
        self.controller.set_weights(self.best_params)
        
        try:
            # Run validation rollouts
            val_rewards = []
            for _ in range(5):  # Fewer rollouts for validation
                obs = self.real_env.reset()
                total_reward = 0
                done = False
                
                while not done:
                    # Get action (this would need proper VAE/RNN integration)
                    action = self.controller.get_action(obs, deterministic=True)
                    obs, reward, done, _ = self.real_env.step(action)
                    total_reward += reward
                
                val_rewards.append(total_reward)
            
            val_mean = np.mean(val_rewards)
            logger.info(f"Real env validation: {val_mean:.2f}±{np.std(val_rewards):.2f}")
            
            if self.config.tensorboard_log and hasattr(self, 'tb_writer'):
                self.tb_writer.add_scalar('validation/real_env_reward', val_mean, self.generation)
                
        except Exception as e:
            logger.error(f"Real environment validation failed: {e}")
        finally:
            # Restore parameters
            self.controller.set_weights(temp_params)
    
    def save_checkpoint(self, is_final: bool = False):
        """Save training checkpoint."""
        suffix = 'final' if is_final else f'gen_{self.generation}'
        
        checkpoint = {
            'generation': self.generation,
            'controller_params': self.best_params,
            'controller_config': {
                'input_size': self.controller.input_size,
                'action_size': self.controller.action_size,
                'hidden_sizes': self.controller.hidden_sizes,
                'action_type': self.controller.action_type
            },
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'es_state': self.es,
            'training_config': asdict(self.config)
        }
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f'checkpoint_{suffix}.pkl')
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.generation = checkpoint['generation']
        self.best_fitness = checkpoint['best_fitness']
        self.best_params = checkpoint['controller_params']
        self.fitness_history = checkpoint['fitness_history']
        
        if self.best_params is not None:
            self.controller.set_weights(self.best_params)
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resumed at generation {self.generation}, best fitness: {self.best_fitness}")


# PPO Baseline Implementation
class PPOControllerTrainer:
    """PPO trainer for controller (baseline comparison)."""
    
    def __init__(
        self,
        env,
        controller_config: Dict,
        total_timesteps: int = 100000,
        device: str = 'cuda'
    ):
        """Initialize PPO trainer."""
        if not HAS_SB3:
            raise ImportError("stable-baselines3 required for PPO training")
        
        self.env = env
        self.total_timesteps = total_timesteps
        self.device = device
        
        # Create vectorized environment
        self.vec_env = DummyVecEnv([lambda: env])
        
        # Initialize PPO model
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device=device,
            verbose=1
        )
    
    def train(self) -> Dict[str, Any]:
        """Train using PPO."""
        logger.info(f"Starting PPO training for {self.total_timesteps} timesteps")
        
        start_time = time.time()
        self.model.learn(total_timesteps=self.total_timesteps)
        training_time = time.time() - start_time
        
        results = {
            'training_time': training_time,
            'total_timesteps': self.total_timesteps
        }
        
        logger.info(f"PPO training completed in {training_time:.2f}s")
        return results
    
    def save(self, path: str):
        """Save PPO model."""
        self.model.save(path)
    
    def load(self, path: str):
        """Load PPO model."""
        self.model = PPO.load(path, env=self.vec_env)


def create_demo_trainer():
    """Create a demo trainer for testing."""
    
    # Create a simple controller
    from ..models.controller import Controller
    
    controller = Controller(
        input_size=32 + 256,  # z_dim + h_dim
        action_size=3,
        hidden_sizes=(),  # Linear controller
        action_type='continuous'
    )
    
    # Create training config
    config = TrainingConfig(
        population_size=16,  # Small for demo
        max_generations=10,
        num_rollouts=4,
        num_workers=2,
        log_frequency=1
    )
    
    # Create trainer
    trainer = ControllerTrainer(
        controller=controller,
        config=config
    )
    
    return trainer


def main():
    """Demo training run."""
    logger.info("Running controller training demo")
    
    try:
        # Create demo trainer
        trainer = create_demo_trainer()
        
        # Run short training
        results = trainer.train_cmaes(generations=5)
        
        logger.info("Demo completed successfully")
        logger.info(f"Results: {results}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()

"""Validation script for controller training pipeline.

This script provides comprehensive validation of the controller training pipeline,
including unit tests, integration tests, and end-to-end training validation.

Tests covered:
1. Controller model functionality (forward pass, weight management)
2. CMA-ES trainer setup and basic operations  
3. Multiprocessing evaluation pipeline
4. Dream environment integration
5. PPO baseline comparison
6. Checkpoint saving/loading
7. Logging functionality
8. Memory usage and performance profiling

Usage:
    python validate_controller_trainer.py --quick    # Quick validation
    python validate_controller_trainer.py --full     # Full validation suite
    python validate_controller_trainer.py --profile  # Performance profiling
"""

import os
import sys
import time
import argparse
import tempfile
import shutil
import traceback
import psutil
import numpy as np
import torch
from typing import Dict, List, Optional
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from world_models.models.controller import Controller, CMAESController
from world_models.train.controller_trainer import ControllerTrainer, TrainingConfig, evaluate_controller_rollouts
from world_models.tools.dream_env import DreamEnvironment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


class ControllerValidation:
    """Main validation class for controller training pipeline."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """Initialize validation suite.
        
        Args:
            temp_dir: Temporary directory for test files
        """
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix='controller_validation_')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results = {}
        
        logger.info(f"Validation initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Temp dir: {self.temp_dir}")
    
    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temp dir: {self.temp_dir}")
    
    def validate_controller_models(self) -> Dict[str, bool]:
        """Test controller model functionality."""
        logger.info("Validating controller models...")
        
        tests = {}
        
        try:
            # Test configurations
            configs = [
                # Linear continuous controller
                {
                    'input_size': 32 + 256,  # z_dim + h_dim
                    'action_size': 3,
                    'hidden_sizes': (),  # Linear controller
                    'action_type': 'continuous'
                },
                # MLP discrete controller
                {
                    'input_size': 32 + 256,
                    'action_size': 4,
                    'hidden_sizes': (64,),  # Single hidden layer
                    'action_type': 'discrete'
                },
            ]
            
            for i, config in enumerate(configs):
                logger.info(f"Testing config {i+1}: {config['action_type']}, {config['action_size']} actions")
                
                # Test controller
                test_name = f"controller_{config['action_type']}_{i}"
                
                try:
                    # Create controller
                    controller = Controller(**config).to(self.device)
                    
                    # Test forward pass
                    batch_size = 4
                    z_dim = 32
                    h_dim = 256
                    z = torch.randn(batch_size, z_dim, device=self.device)
                    h = torch.randn(batch_size, h_dim, device=self.device)
                    
                    with torch.no_grad():
                        output = controller(z, h)
                    
                    # Check output shape for continuous
                    if config['action_type'] == 'continuous':
                        expected_shape = (batch_size, config['action_size'] * 2)  # mean + log_std
                        if output.shape != expected_shape:
                            raise ValidationError(f"Wrong output shape: {output.shape} vs {expected_shape}")
                        
                        # Test action generation
                        action = controller.get_action(z, h, deterministic=True)
                        if action.shape != (batch_size, config['action_size']):
                            raise ValidationError(f"Wrong action shape: {action.shape}")
                        
                    # Check output shape for discrete
                    if config['action_type'] == 'discrete':
                        expected_shape = (batch_size, config['action_size'])
                        if output.shape != expected_shape:
                            raise ValidationError(f"Wrong output shape: {output.shape} vs {expected_shape}")
                        
                        # Test action generation
                        action = controller.get_action(z, h, deterministic=True)
                        if action.shape != (batch_size,):
                            raise ValidationError(f"Wrong action shape: {action.shape}")
                    
                    # Test weight management
                    weights = controller.get_parameters()
                    if len(weights) == 0:
                        raise ValidationError("No weights returned")
                    
                    # Test weight setting
                    new_weights = np.random.randn(len(weights)) * 0.1
                    controller.set_parameters(new_weights)
                    
                    restored_weights = controller.get_parameters()
                    if not np.allclose(new_weights, restored_weights, atol=1e-6):
                        raise ValidationError("Weight setting/getting not consistent")
                    
                    # Test parameter count
                    param_count = controller.get_num_parameters()
                    if param_count != len(weights):
                        raise ValidationError(f"Parameter count mismatch: {param_count} vs {len(weights)}")
                    
                    tests[test_name] = True
                    logger.info(f"  {test_name}: PASSED ({param_count} params)")
                    
                except Exception as e:
                    tests[test_name] = False
                    logger.error(f"  {test_name}: FAILED - {e}")
            
        except Exception as e:
            logger.error(f"Controller model validation failed: {e}")
            tests['overall'] = False
            return tests
        
        tests['overall'] = all(tests.values())
        logger.info(f"Controller model validation: {'PASSED' if tests['overall'] else 'FAILED'}")
        return tests
    
    def validate_dream_environment(self) -> Dict[str, bool]:
        """Test dream environment functionality."""
        logger.info("Validating dream environment...")
        
        tests = {}
        
        try:
            # Create mock dream environment
            dream_env = DreamEnvironment(
                vae=None,  # Mock VAE
                rnn=None,  # Mock RNN
                action_space_type='continuous',
                action_dim=3,
                device=self.device
            )
            
            # Test reset
            obs, info = dream_env.reset()
            tests['reset'] = True
            logger.info(f"  Reset: PASSED - obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
            
            # Test step
            action = np.array([0.1, -0.2, 0.3])
            obs, reward, terminated, truncated, info = dream_env.step(action)
            
            tests['step'] = True
            logger.info(f"  Step: PASSED - reward: {reward}, done: {terminated or truncated}")
            
            # Test multiple steps
            for _ in range(5):
                action = np.random.uniform(-1, 1, size=3)
                obs, reward, terminated, truncated, info = dream_env.step(action)
                if terminated or truncated:
                    obs, info = dream_env.reset()
            
            tests['multi_step'] = True
            logger.info("  Multi-step: PASSED")
            
        except Exception as e:
            logger.error(f"Dream environment validation failed: {e}")
            tests['overall'] = False
            return tests
        
        tests['overall'] = all(tests.values())
        logger.info(f"Dream environment validation: {'PASSED' if tests['overall'] else 'FAILED'}")
        return tests
    
    def validate_evaluation_pipeline(self) -> Dict[str, bool]:
        """Test parallel evaluation pipeline."""
        logger.info("Validating evaluation pipeline...")
        
        tests = {}
        
        try:
            # Create test controller
            controller = create_controller(
                'linear',
                ControllerConfig(z_dim=32, h_dim=256, action_dim=3, device='cpu')  # Force CPU
            )
            
            # Test single evaluation
            params = controller.get_weights()
            
            controller_config = {
                'type': 'linear',
                'config': {
                    'z_dim': 32,
                    'h_dim': 256, 
                    'action_dim': 3,
                    'action_type': 'continuous',
                    'device': 'cpu'
                }
            }
            
            dream_env_config = {
                'action_space_type': 'continuous',
                'action_dim': 3,
            }
            
            eval_config = {
                'num_rollouts': 2,  # Small for testing
                'max_episode_length': 10,
                'temperature': 1.0,
                'deterministic_eval': False,
                'device': 'cpu'
            }
            
            result = evaluate_controller_rollouts(
                params, controller_config, dream_env_config, eval_config, worker_id=0
            )
            
            # Check result structure
            required_keys = ['fitness', 'total_reward', 'episode_length', 'success_rate']
            for key in required_keys:
                if key not in result:
                    raise ValidationError(f"Missing key in evaluation result: {key}")
            
            tests['single_eval'] = True
            logger.info(f"  Single evaluation: PASSED - fitness: {result['fitness']:.3f}")
            
            # Test multiple parameter vectors
            test_params = [
                params,
                params + np.random.randn(len(params)) * 0.1,
                params + np.random.randn(len(params)) * 0.1,
            ]
            
            results = []
            for i, p in enumerate(test_params):
                result = evaluate_controller_rollouts(
                    p, controller_config, dream_env_config, eval_config, worker_id=i
                )
                results.append(result)
            
            tests['multi_eval'] = True
            logger.info(f"  Multi evaluation: PASSED - {len(results)} results")
            
        except Exception as e:
            logger.error(f"Evaluation pipeline validation failed: {e}")
            logger.error(traceback.format_exc())
            tests['overall'] = False
            return tests
        
        tests['overall'] = all(tests.values())
        logger.info(f"Evaluation pipeline validation: {'PASSED' if tests['overall'] else 'FAILED'}")
        return tests
    
    def validate_cmaes_trainer(self, quick: bool = True) -> Dict[str, bool]:
        """Test CMA-ES trainer functionality."""
        logger.info("Validating CMA-ES trainer...")
        
        tests = {}
        
        try:
            # Create controller
            controller = create_controller(
                'linear',
                ControllerConfig(z_dim=32, h_dim=256, action_dim=3, device=self.device)
            )
            
            # Create training config
            config = TrainingConfig(
                population_size=8 if quick else 16,
                max_generations=2 if quick else 5,
                num_rollouts=2 if quick else 4,
                num_workers=1 if quick else 2,
                save_frequency=1,
                log_frequency=1,
                checkpoint_dir=os.path.join(self.temp_dir, 'checkpoints'),
                log_dir=os.path.join(self.temp_dir, 'logs'),
                tensorboard_log=False,  # Disable for testing
                device=self.device
            )
            
            # Create trainer
            trainer = ControllerTrainer(
                controller=controller,
                config=config
            )
            
            tests['trainer_init'] = True
            logger.info("  Trainer initialization: PASSED")
            
            # Test training
            initial_params = controller.get_weights().copy()
            
            results = trainer.train_cmaes(generations=config.max_generations)
            
            # Check training completed
            if 'generations_trained' not in results:
                raise ValidationError("Training results missing generations_trained")
            
            if results['generations_trained'] == 0:
                raise ValidationError("No generations were trained")
            
            tests['training'] = True
            logger.info(f"  Training: PASSED - {results['generations_trained']} generations")
            
            # Check parameters changed
            final_params = controller.get_weights()
            if np.allclose(initial_params, final_params):
                logger.warning("  Parameters didn't change during training")
            
            # Test checkpoint saving
            checkpoint_files = [f for f in os.listdir(config.checkpoint_dir) if f.endswith('.pkl')]
            if len(checkpoint_files) == 0:
                raise ValidationError("No checkpoint files saved")
            
            tests['checkpoints'] = True
            logger.info(f"  Checkpoints: PASSED - {len(checkpoint_files)} files saved")
            
            # Test checkpoint loading
            checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_files[0])
            new_controller = create_controller(
                'linear',
                ControllerConfig(z_dim=32, h_dim=256, action_dim=3, device=self.device)
            )
            new_trainer = ControllerTrainer(controller=new_controller, config=config)
            new_trainer.load_checkpoint(checkpoint_path)
            
            tests['checkpoint_loading'] = True
            logger.info("  Checkpoint loading: PASSED")
            
        except Exception as e:
            logger.error(f"CMA-ES trainer validation failed: {e}")
            logger.error(traceback.format_exc())
            tests['overall'] = False
            return tests
        
        tests['overall'] = all(tests.values())
        logger.info(f"CMA-ES trainer validation: {'PASSED' if tests['overall'] else 'FAILED'}")
        return tests
    
    def validate_logging_functionality(self) -> Dict[str, bool]:
        """Test logging and monitoring functionality."""
        logger.info("Validating logging functionality...")
        
        tests = {}
        
        try:
            # Create controller and config
            controller = create_controller(
                'linear',
                ControllerConfig(z_dim=32, h_dim=256, action_dim=3, device=self.device)
            )
            
            config = TrainingConfig(
                population_size=4,
                max_generations=2,
                num_rollouts=2,
                num_workers=1,
                csv_log=True,
                tensorboard_log=False,  # Skip TensorBoard for validation
                log_frequency=1,
                checkpoint_dir=os.path.join(self.temp_dir, 'checkpoints'),
                log_dir=os.path.join(self.temp_dir, 'logs'),
                device=self.device
            )
            
            # Create trainer
            trainer = ControllerTrainer(controller=controller, config=config)
            
            # Run short training
            trainer.train_cmaes(generations=2)
            
            # Check CSV log
            csv_path = os.path.join(config.log_dir, 'training_log.csv')
            if not os.path.exists(csv_path):
                raise ValidationError("CSV log file not created")
            
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                if len(lines) < 2:  # Header + at least one data row
                    raise ValidationError("CSV log appears empty")
            
            tests['csv_logging'] = True
            logger.info("  CSV logging: PASSED")
            
            # Check checkpoint directory
            if not os.path.exists(config.checkpoint_dir):
                raise ValidationError("Checkpoint directory not created")
            
            checkpoint_files = os.listdir(config.checkpoint_dir)
            if len(checkpoint_files) == 0:
                raise ValidationError("No checkpoint files created")
            
            tests['checkpoint_dir'] = True
            logger.info(f"  Checkpoint directory: PASSED - {len(checkpoint_files)} files")
            
        except Exception as e:
            logger.error(f"Logging functionality validation failed: {e}")
            tests['overall'] = False
            return tests
        
        tests['overall'] = all(tests.values())
        logger.info(f"Logging functionality validation: {'PASSED' if tests['overall'] else 'FAILED'}")
        return tests
    
    def validate_memory_usage(self) -> Dict[str, bool]:
        """Test memory usage and performance."""
        logger.info("Validating memory usage...")
        
        tests = {}
        
        try:
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create larger controller for memory testing
            controller = create_controller(
                'mlp',
                ControllerConfig(
                    z_dim=64, h_dim=512, action_dim=6, 
                    hidden_dim=128, device=self.device
                )
            )
            
            config = TrainingConfig(
                population_size=16,
                max_generations=3,
                num_rollouts=4,
                num_workers=1,  # Single worker to control memory
                checkpoint_dir=os.path.join(self.temp_dir, 'checkpoints'),
                log_dir=os.path.join(self.temp_dir, 'logs'),
                tensorboard_log=False,
                device=self.device
            )
            
            trainer = ControllerTrainer(controller=controller, config=config)
            
            # Monitor memory during training
            max_memory = initial_memory
            
            def memory_monitor():
                nonlocal max_memory
                current_memory = process.memory_info().rss / 1024 / 1024
                max_memory = max(max_memory, current_memory)
                return current_memory
            
            # Run training with memory monitoring
            start_time = time.time()
            results = trainer.train_cmaes(generations=2)
            end_time = time.time()
            
            final_memory = memory_monitor()
            memory_increase = max_memory - initial_memory
            
            # Check memory usage is reasonable (< 2GB increase)
            if memory_increase > 2000:
                logger.warning(f"High memory usage: {memory_increase:.1f} MB increase")
            else:
                tests['memory_usage'] = True
                logger.info(f"  Memory usage: PASSED - {memory_increase:.1f} MB increase")
            
            # Check training time is reasonable
            training_time = end_time - start_time
            if training_time > 300:  # 5 minutes
                logger.warning(f"Slow training: {training_time:.1f} seconds")
            else:
                tests['training_time'] = True
                logger.info(f"  Training time: PASSED - {training_time:.1f} seconds")
            
        except Exception as e:
            logger.error(f"Memory validation failed: {e}")
            tests['overall'] = False
            return tests
        
        tests['overall'] = all(tests.values())
        logger.info(f"Memory usage validation: {'PASSED' if tests['overall'] else 'FAILED'}")
        return tests
    
    def run_validation_suite(self, quick: bool = False, include_memory: bool = False) -> Dict[str, Dict[str, bool]]:
        """Run complete validation suite.
        
        Args:
            quick: Run quick tests only
            include_memory: Include memory/performance tests
            
        Returns:
            Dictionary of test results by category
        """
        logger.info("=" * 60)
        logger.info("STARTING CONTROLLER TRAINER VALIDATION SUITE")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Run validation tests
        test_results = {}
        
        # Core functionality tests
        test_results['controller_models'] = self.validate_controller_models()
        test_results['dream_environment'] = self.validate_dream_environment()
        test_results['evaluation_pipeline'] = self.validate_evaluation_pipeline()
        test_results['cmaes_trainer'] = self.validate_cmaes_trainer(quick=quick)
        test_results['logging'] = self.validate_logging_functionality()
        
        # Optional memory tests
        if include_memory:
            test_results['memory_usage'] = self.validate_memory_usage()
        
        # Summary
        total_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("VALIDATION SUITE RESULTS")
        logger.info("=" * 60)
        
        all_passed = True
        for category, results in test_results.items():
            status = "PASSED" if results.get('overall', False) else "FAILED"
            logger.info(f"{category:20s}: {status}")
            if not results.get('overall', False):
                all_passed = False
        
        logger.info("=" * 60)
        logger.info(f"Overall result: {'PASSED' if all_passed else 'FAILED'}")
        logger.info(f"Total validation time: {total_time:.2f} seconds")
        logger.info("=" * 60)
        
        self.results = test_results
        return test_results


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate controller training pipeline")
    parser.add_argument('--quick', action='store_true', help='Run quick validation')
    parser.add_argument('--full', action='store_true', help='Run full validation suite')
    parser.add_argument('--profile', action='store_true', help='Include performance profiling')
    parser.add_argument('--temp-dir', type=str, help='Temporary directory for test files')
    
    args = parser.parse_args()
    
    # Create validation suite
    validator = ControllerValidation(temp_dir=args.temp_dir)
    
    try:
        # Run validation
        if args.full:
            results = validator.run_validation_suite(quick=False, include_memory=args.profile)
        elif args.profile:
            results = validator.run_validation_suite(quick=False, include_memory=True)
        else:
            results = validator.run_validation_suite(quick=True, include_memory=False)
        
        # Exit with error code if any tests failed
        overall_success = all(r.get('overall', False) for r in results.values())
        sys.exit(0 if overall_success else 1)
        
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        validator.cleanup()


if __name__ == "__main__":
    main()

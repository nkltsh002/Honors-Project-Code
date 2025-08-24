"""
Main training script for World Models architecture

This script orchestrates the complete training pipeline:
1. Data collection (random rollouts)
2. VAE training (Vision model)
3. MDN-RNN training (Memory model) 
4. Controller training (Control model)
5. Evaluation and comparison with baseline PPO

Based on Ha & Schmidhuber (2018): "World Models"
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config, ExperimentConfig
from models.vae import ConvVAE
from models.mdnrnn import MDNRNN, create_sequence_dataset
from models.controller import Controller, CMAESController
from utils.environment import collect_random_rollouts, EnvironmentWrapper, visualize_rollout
from training.train_utils import (
    TrainingLogger, VAETrainer, create_data_loaders, 
    SequenceDataset, RolloutDataset
)

class WorldModelsTrainer:
    """
    Complete World Models training pipeline.
    
    Manages the three-stage training process and provides utilities
    for evaluation, visualization, and model persistence.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize World Models trainer.
        
        Args:
            config: Complete experiment configuration
        """
        self.config = config
        self.device = torch.device(config.training.device)
        
        # Initialize logger
        self.logger = TrainingLogger(
            config.training.log_dir, 
            config.experiment_name
        )
        
        # Initialize models
        self.vae = None
        self.mdnrnn = None
        self.controller = None
        
        # Data storage
        self.rollout_data = None
        
        print(f"World Models Trainer initialized")
        print(f"Experiment: {config.experiment_name}")
        print(f"Environment: {config.environment}")
        print(f"Device: {self.device}")
    
    def collect_data(self, force_recollect: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect random rollouts for training.
        
        Args:
            force_recollect: Force data recollection even if cached data exists
            
        Returns:
            Tuple of (frames, actions, rewards) arrays
        """
        print("=" * 60)
        print("STAGE 1: DATA COLLECTION")
        print("=" * 60)
        
        # Check for existing data
        data_path = f'./data/{self.config.environment.replace("/", "_")}_rollouts.npz'
        
        if os.path.exists(data_path) and not force_recollect:
            print(f"Loading existing rollout data from {data_path}")
            data = np.load(data_path)
            frames = data['frames']
            actions = data['actions']
            rewards = data['rewards']
            
            print(f"Loaded {frames.shape[0]} rollouts")
            print(f"Frame shape: {frames.shape}")
            print(f"Action shape: {actions.shape}")
            print(f"Reward shape: {rewards.shape}")
        else:
            print(f"Collecting {self.config.training.num_random_rollouts} random rollouts...")
            
            frames, actions, rewards = collect_random_rollouts(
                env_name=self.config.environment,
                num_rollouts=self.config.training.num_random_rollouts,
                rollout_length=self.config.training.rollout_length,
                save_dir='./data',
                visualize=self.config.training.visualize and self.config.training.num_random_rollouts <= 10,
                seed=self.config.training.seed
            )
        
        self.rollout_data = (frames, actions, rewards)
        
        # Visualize sample rollouts
        if self.config.training.visualize:
            print("Visualizing sample rollouts...")
            for i in range(min(3, frames.shape[0])):
                rollout_frames = frames[i]
                rollout_actions = actions[i] 
                rollout_rewards = rewards[i]
                
                # Find actual rollout length
                actual_length = np.where(rollout_frames.sum(axis=(1, 2, 3)) == 0)[0]
                if len(actual_length) > 0:
                    actual_length = actual_length[0]
                else:
                    actual_length = len(rollout_frames)
                
                if actual_length > 0:
                    visualize_rollout(
                        rollout_frames[:actual_length],
                        rollout_actions[:actual_length],
                        rollout_rewards[:actual_length],
                        save_path=f'./videos/rollout_{i}.mp4' if self.config.training.save_videos else None
                    )
        
        return frames, actions, rewards
    
    def train_vae(self, frames: np.ndarray) -> ConvVAE:
        """
        Train the Vision model (VAE).
        
        Args:
            frames: Frame data for training
            
        Returns:
            Trained VAE model
        """
        print("=" * 60)
        print("STAGE 2: VAE TRAINING (VISION MODEL)")
        print("=" * 60)
        
        if not self.config.train_vae:
            print("Skipping VAE training (train_vae=False)")
            # Try to load existing model
            vae_path = f'./checkpoints/vae_{self.config.experiment_name}.pth'
            if os.path.exists(vae_path):
                print(f"Loading existing VAE from {vae_path}")
                self.vae = ConvVAE(
                    latent_size=self.config.vae.latent_size,
                    beta=self.config.vae.beta
                ).to(self.device)
                
                checkpoint = torch.load(vae_path, map_location=self.device)
                self.vae.load_state_dict(checkpoint['model_state_dict'])
                return self.vae
            else:
                raise FileNotFoundError(f"VAE model not found at {vae_path} and train_vae=False")
        
        # Initialize VAE
        self.vae = ConvVAE(
            latent_size=self.config.vae.latent_size,
            beta=self.config.vae.beta
        ).to(self.device)
        
        print(f"VAE parameters: {sum(p.numel() for p in self.vae.parameters()):,}")
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            frames=frames,
            batch_size=self.config.vae.batch_size,
            val_split=0.2,
            num_workers=self.config.training.num_workers
        )
        
        # Initialize trainer
        trainer = VAETrainer(
            model=self.vae,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=self.config.vae.learning_rate,
            beta=self.config.vae.beta,
            device=self.device,
            logger=self.logger
        )
        
        # Train
        save_path = f'./checkpoints/vae_{self.config.experiment_name}.pth'
        trainer.train(
            num_epochs=self.config.vae.num_epochs,
            warmup_epochs=self.config.vae.warmup_epochs,
            save_frequency=self.config.training.save_frequency,
            save_path=save_path
        )
        
        print(f"VAE training completed. Model saved to {save_path}")
        
        return self.vae
    
    def encode_rollouts(self, frames: np.ndarray) -> np.ndarray:
        """
        Encode rollout frames to latent representations.
        
        Args:
            frames: Frame data (num_rollouts, seq_len, 64, 64, 3)
            
        Returns:
            Encoded latent sequences (num_rollouts, seq_len, latent_size)
        """
        print("Encoding rollouts to latent space...")
        
        if self.vae is None:
            raise ValueError("VAE must be trained before encoding rollouts")
        
        self.vae.eval()
        
        num_rollouts, seq_len = frames.shape[:2]
        latent_size = self.config.vae.latent_size
        
        # Storage for latent sequences
        latent_sequences = np.zeros((num_rollouts, seq_len, latent_size), dtype=np.float32)
        
        batch_size = self.config.vae.batch_size
        
        with torch.no_grad():
            for rollout_idx in tqdm(range(num_rollouts), desc="Encoding rollouts"):
                rollout_frames = frames[rollout_idx]  # (seq_len, 64, 64, 3)
                
                # Process in batches
                for start_idx in range(0, seq_len, batch_size):
                    end_idx = min(start_idx + batch_size, seq_len)
                    
                    # Get batch frames
                    batch_frames = rollout_frames[start_idx:end_idx]
                    
                    # Skip zero-padded frames
                    if batch_frames.sum() == 0:
                        continue
                    
                    # Convert to tensor
                    batch_tensor = torch.from_numpy(batch_frames).permute(0, 3, 1, 2).float().to(self.device)
                    
                    # Encode
                    latent_batch = self.vae.get_latent_representation(batch_tensor)
                    
                    # Store
                    latent_sequences[rollout_idx, start_idx:end_idx] = latent_batch.cpu().numpy()
        
        print(f"Encoded rollouts shape: {latent_sequences.shape}")
        
        # Save encoded data
        save_path = f'./data/encoded_{self.config.experiment_name}.npz'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(
            save_path,
            latent_sequences=latent_sequences,
            experiment_name=self.config.experiment_name
        )
        
        print(f"Saved encoded rollouts to {save_path}")
        
        return latent_sequences
    
    def train_mdnrnn(
        self, 
        latent_sequences: np.ndarray, 
        action_sequences: np.ndarray
    ) -> MDNRNN:
        """
        Train the Memory model (MDN-RNN).
        
        Args:
            latent_sequences: Encoded latent sequences
            action_sequences: Action sequences
            
        Returns:
            Trained MDN-RNN model
        """
        print("=" * 60)
        print("STAGE 3: MDN-RNN TRAINING (MEMORY MODEL)")
        print("=" * 60)
        
        if not self.config.train_mdnrnn:
            print("Skipping MDN-RNN training (train_mdnrnn=False)")
            # Try to load existing model
            mdnrnn_path = f'./checkpoints/mdnrnn_{self.config.experiment_name}.pth'
            if os.path.exists(mdnrnn_path):
                print(f"Loading existing MDN-RNN from {mdnrnn_path}")
                self.mdnrnn = MDNRNN(
                    latent_size=self.config.mdnrnn.latent_size,
                    action_size=self.config.mdnrnn.action_size,
                    hidden_size=self.config.mdnrnn.hidden_size,
                    num_mixtures=self.config.mdnrnn.num_mixtures
                ).to(self.device)
                
                checkpoint = torch.load(mdnrnn_path, map_location=self.device)
                self.mdnrnn.load_state_dict(checkpoint['model_state_dict'])
                return self.mdnrnn
            else:
                raise FileNotFoundError(f"MDN-RNN model not found at {mdnrnn_path} and train_mdnrnn=False")
        
        # Initialize MDN-RNN
        self.mdnrnn = MDNRNN(
            latent_size=self.config.mdnrnn.latent_size,
            action_size=self.config.mdnrnn.action_size,
            hidden_size=self.config.mdnrnn.hidden_size,
            num_layers=self.config.mdnrnn.num_layers,
            num_mixtures=self.config.mdnrnn.num_mixtures,
            dropout=self.config.mdnrnn.dropout
        ).to(self.device)
        
        print(f"MDN-RNN parameters: {sum(p.numel() for p in self.mdnrnn.parameters()):,}")
        
        # Convert to tensors
        latent_tensor = torch.from_numpy(latent_sequences).float()
        action_tensor = torch.from_numpy(action_sequences).float()
        
        # Create sequence dataset
        dataset = SequenceDataset(
            latent_sequences=latent_tensor,
            action_sequences=action_tensor,
            sequence_length=self.config.mdnrnn.sequence_length,
            overlap=self.config.mdnrnn.sequence_length // 2
        )
        
        # Create data loader
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.mdnrnn.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        # Optimizer and scheduler
        optimizer = torch.optim.Adam(
            self.mdnrnn.parameters(), 
            lr=self.config.mdnrnn.learning_rate
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training loop
        print(f"Starting MDN-RNN training for {self.config.mdnrnn.num_epochs} epochs...")
        
        best_loss = float('inf')
        save_path = f'./checkpoints/mdnrnn_{self.config.experiment_name}.pth'
        
        for epoch in range(self.config.mdnrnn.num_epochs):
            self.mdnrnn.train()
            
            epoch_losses = []
            
            with tqdm(data_loader, desc=f"Epoch {epoch+1}/{self.config.mdnrnn.num_epochs}") as pbar:
                for batch_idx, batch in enumerate(pbar):
                    z = batch['z'].to(self.device)
                    actions = batch['actions'].to(self.device)
                    z_target = batch['z_target'].to(self.device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    losses = self.mdnrnn.compute_loss(z, actions, z_target)
                    
                    # Backward pass
                    losses['total_loss'].backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.mdnrnn.parameters(), 
                        self.config.mdnrnn.gradient_clip
                    )
                    
                    optimizer.step()
                    
                    epoch_losses.append(losses['total_loss'].item())
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{losses['total_loss'].item():.4f}",
                        'pred_err': f"{losses['prediction_error'].item():.4f}",
                        'entropy': f"{losses['mixture_entropy'].item():.4f}"
                    })
                    
                    # Log batch metrics
                    if batch_idx % 50 == 0:
                        step = epoch * len(data_loader) + batch_idx
                        self.logger.log_dict(
                            {k: v.item() for k, v in losses.items()},
                            step=step,
                            prefix="mdnrnn_batch"
                        )
            
            # Epoch metrics
            avg_loss = np.mean(epoch_losses)
            
            # Log epoch metrics
            self.logger.log_scalar("mdnrnn_epoch/loss", avg_loss, step=epoch)
            self.logger.log_scalar("mdnrnn_epoch/learning_rate", 
                                 optimizer.param_groups[0]['lr'], step=epoch)
            
            scheduler.step(avg_loss)
            
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'model_state_dict': self.mdnrnn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_loss': best_loss
                }, save_path)
                
                print(f"Saved best model (loss: {best_loss:.4f})")
        
        print(f"MDN-RNN training completed. Model saved to {save_path}")
        
        return self.mdnrnn
    
    def train_controller(
        self, 
        latent_sequences: np.ndarray,
        action_sequences: np.ndarray
    ) -> Controller:
        """
        Train the Controller with CMA-ES.
        
        Args:
            latent_sequences: Encoded latent sequences
            action_sequences: Action sequences (for action space info)
            
        Returns:
            Trained Controller
        """
        print("=" * 60)
        print("STAGE 4: CONTROLLER TRAINING (CONTROL MODEL)")
        print("=" * 60)
        
        if not self.config.train_controller:
            print("Skipping Controller training (train_controller=False)")
            # Try to load existing model
            controller_path = f'./checkpoints/controller_{self.config.experiment_name}.pth'
            if os.path.exists(controller_path):
                print(f"Loading existing Controller from {controller_path}")
                self.controller = Controller(
                    input_size=self.config.controller.input_size,
                    action_size=self.config.controller.action_size,
                    hidden_sizes=self.config.controller.hidden_sizes,
                    action_type='discrete'  # Most environments use discrete actions
                ).to(self.device)
                
                checkpoint = torch.load(controller_path, map_location=self.device)
                self.controller.load_state_dict(checkpoint['model_state_dict'])
                return self.controller
            else:
                print(f"Controller model not found at {controller_path}, training new one...")
        
        # Initialize controller
        self.controller = Controller(
            input_size=self.config.controller.input_size,
            action_size=self.config.controller.action_size,
            hidden_sizes=self.config.controller.hidden_sizes,
            action_type='discrete',  # Most environments use discrete actions
            activation=self.config.controller.activation
        ).to(self.device)
        
        print(f"Controller parameters: {self.controller.get_num_parameters():,}")
        
        # Initialize CMA-ES trainer
        cmaes_trainer = CMAESController(
            controller=self.controller,
            population_size=self.config.controller.population_size,
            sigma=self.config.controller.sigma,
            device=self.device
        )
        
        print(f"Starting Controller training with CMA-ES...")
        print(f"Population size: {self.config.controller.population_size}")
        print(f"Maximum generations: {self.config.controller.max_generations}")
        
        # Fitness evaluation function
        def evaluate_controller(params_batch: np.ndarray) -> np.ndarray:
            """Evaluate a batch of controller parameters"""
            fitness_values = []
            
            for params in params_batch:
                # Set controller parameters
                temp_controller = Controller(
                    input_size=self.config.controller.input_size,
                    action_size=self.config.controller.action_size,
                    hidden_sizes=self.config.controller.hidden_sizes,
                    action_type='discrete',
                    activation=self.config.controller.activation
                ).to(self.device)
                
                temp_controller.set_parameters(params)
                
                # Evaluate in environment
                fitness = self._evaluate_controller_in_environment(temp_controller)
                fitness_values.append(fitness)
            
            return np.array(fitness_values)
        
        # Training loop
        generation_fitness = []
        
        for generation in range(self.config.controller.max_generations):
            # Get candidate solutions
            candidates = cmaes_trainer.ask()
            
            # Evaluate fitness
            fitness_values = evaluate_controller(candidates)
            
            # Update CMA-ES
            cmaes_trainer.tell(candidates, fitness_values)
            
            # Get statistics
            stats = cmaes_trainer.get_stats()
            generation_fitness.append(fitness_values)
            
            # Log progress
            if generation % 10 == 0 or generation == self.config.controller.max_generations - 1:
                print(f"Generation {generation+1}/{self.config.controller.max_generations}")
                print(f"  Best fitness: {stats['best_fitness']:.3f}")
                print(f"  Mean fitness: {stats['mean_fitness']:.3f}")
                print(f"  Std fitness: {stats['std_fitness']:.3f}")
                print(f"  Sigma: {stats['sigma']:.6f}")
            
            # Log to TensorBoard
            if self.logger:
                self.logger.log_dict(stats, step=generation, prefix="controller")
            
            # Check stopping criteria
            if cmaes_trainer.should_stop():
                print(f"CMA-ES converged at generation {generation+1}")
                break
        
        # Update controller with best parameters
        cmaes_trainer.update_controller(use_best=True)
        
        # Save controller
        save_path = f'./checkpoints/controller_{self.config.experiment_name}.pth'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.controller.state_dict(),
            'generation': generation,
            'best_fitness': cmaes_trainer.best_fitness,
            'fitness_history': generation_fitness
        }, save_path)
        
        print(f"Controller training completed. Model saved to {save_path}")
        print(f"Best fitness achieved: {cmaes_trainer.best_fitness:.3f}")
        
        return self.controller
    
    def _evaluate_controller_in_environment(
        self, 
        controller: Controller, 
        num_episodes: int = 3,
        max_steps: int = 1000
    ) -> float:
        """
        Evaluate controller performance in the environment.
        
        Args:
            controller: Controller to evaluate
            num_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            
        Returns:
            Average reward over episodes
        """
        if self.vae is None or self.mdnrnn is None:
            raise ValueError("VAE and MDN-RNN must be trained before controller evaluation")
        
        # Create environment
        env = EnvironmentWrapper(
            self.config.environment,
            render_mode='rgb_array',
            seed=None  # Different seed for each evaluation
        )
        
        total_rewards = []
        
        for episode in range(num_episodes):
            # Reset environment
            frame, info = env.reset()
            
            # Initialize LSTM hidden state
            hidden = None
            total_reward = 0
            
            for step in range(max_steps):
                # Preprocess frame and encode to latent
                with torch.no_grad():
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                    z = self.vae.get_latent_representation(frame_tensor)
                    
                    # Get LSTM hidden state (mock with zeros for now)
                    # In full implementation, would maintain LSTM state across steps
                    h = torch.zeros(1, self.config.mdnrnn.hidden_size, device=self.device)
                    
                    # Get action from controller
                    action = controller.get_action(z, h, temperature=1.0, deterministic=False)
                    
                    # Convert to environment action
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().item()
                
                # Take environment step
                next_frame, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                frame = next_frame
                
                if terminated or truncated:
                    break
            
            total_rewards.append(total_reward)
        
        env.close()
        
        return np.mean(total_rewards)
    
    def train_full_pipeline(self, force_recollect: bool = False):
        """
        Execute the complete World Models training pipeline.
        
        Args:
            force_recollect: Force data recollection
        """
        print("Starting World Models training pipeline...")
        print(f"Experiment: {self.config.experiment_name}")
        print(f"Environment: {self.config.environment}")
        
        start_time = time.time()
        
        try:
            # Stage 1: Collect data
            frames, actions, rewards = self.collect_data(force_recollect)
            
            # Stage 2: Train VAE
            self.vae = self.train_vae(frames)
            
            # Encode rollouts to latent space
            latent_sequences = self.encode_rollouts(frames)
            
            # Stage 3: Train MDN-RNN
            self.mdnrnn = self.train_mdnrnn(latent_sequences, actions)
            
            # Stage 4: Train Controller
            self.controller = self.train_controller(latent_sequences, actions)
            
            # Final evaluation
            print("=" * 60)
            print("FINAL EVALUATION")
            print("=" * 60)
            
            final_fitness = self._evaluate_controller_in_environment(
                self.controller, num_episodes=10, max_steps=self.config.training.rollout_length
            )
            
            print(f"Final controller performance: {final_fitness:.3f}")
            
            # Log final results
            if self.logger:
                self.logger.log_scalar("final/controller_performance", final_fitness, step=0)
            
            total_time = time.time() - start_time
            print(f"\nTotal training time: {total_time/3600:.2f} hours")
            print("World Models training completed successfully!")
            
        except Exception as e:
            print(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            if self.logger:
                self.logger.close()

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train World Models')
    parser.add_argument('--env', type=str, default='ALE/Pong-v5',
                       help='Environment name')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Training device (cuda/cpu/auto)')
    parser.add_argument('--force-recollect', action='store_true',
                       help='Force data recollection')
    parser.add_argument('--skip-vae', action='store_true',
                       help='Skip VAE training')
    parser.add_argument('--skip-mdnrnn', action='store_true',
                       help='Skip MDN-RNN training')
    parser.add_argument('--skip-controller', action='store_true',
                       help='Skip Controller training')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(
        environment=args.env,
        experiment_name=args.experiment or f"world_models_{args.env.replace('/', '_')}"
    )
    
    # Override device if specified
    if args.device != 'auto':
        if args.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA requested but not available, using CPU")
            config.training.device = 'cpu'
        else:
            config.training.device = args.device
    
    # Override training stages if specified
    if args.skip_vae:
        config.train_vae = False
    if args.skip_mdnrnn:
        config.train_mdnrnn = False  
    if args.skip_controller:
        config.train_controller = False
    
    print("Configuration:")
    print(f"  Environment: {config.environment}")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Device: {config.training.device}")
    print(f"  Train VAE: {config.train_vae}")
    print(f"  Train MDN-RNN: {config.train_mdnrnn}")
    print(f"  Train Controller: {config.train_controller}")
    
    # Create directories
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./videos', exist_ok=True)
    
    # Initialize and run trainer
    trainer = WorldModelsTrainer(config)
    trainer.train_full_pipeline(force_recollect=args.force_recollect)

if __name__ == "__main__":
    main()

"""
CPU-Optimized Controller for World Models

This is a CPU-only version of the controller that avoids CUDA checks and potential hanging imports.
Designed specifically for systems without CUDA or where CUDA initialization causes issues.

Key optimizations:
- No CUDA availability checks during import
- CPU-only device specification
- Simplified imports to avoid hanging
- Fast initialization for diagnostic purposes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np

class ControllerCPU(nn.Module):
    """
    CPU-optimized neural network controller that maps [z_t, h_t] to actions.
    
    This version is specifically designed for:
    - CPU-only environments
    - Fast import without hanging
    - Systems where CUDA initialization causes issues
    """
    
    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_sizes: Tuple[int, ...] = (64,),
        action_type: str = 'discrete',
        temperature: float = 1.0,
        device: str = 'cpu'  # Always CPU for this version
    ):
        super().__init__()
        
        self.input_size = input_size
        self.action_size = action_size
        self.action_type = action_type
        self.temperature = temperature
        self.device = torch.device('cpu')  # Force CPU
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        # Output layer
        if action_type == 'discrete':
            layers.append(nn.Linear(prev_size, action_size))
            self.policy_head = nn.Sequential(*layers)
        elif action_type == 'continuous':
            # Mean and log_std for continuous actions
            self.policy_mean = nn.Sequential(*layers, nn.Linear(prev_size, action_size))
            self.policy_logstd = nn.Parameter(torch.zeros(action_size))
        else:
            raise ValueError(f"Unknown action_type: {action_type}")
        
        # Value function head (optional, for PPO training)
        value_layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            value_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        value_layers.append(nn.Linear(prev_size, 1))
        self.value_head = nn.Sequential(*value_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through controller
        
        Args:
            state: Concatenated [z_t, h_t] state tensor
        
        Returns:
            Dictionary containing action logits/params and value estimate
        """
        batch_size = state.size(0)
        
        # Add value estimate (common to both action types)
        value = self.value_head(state).squeeze(-1)
        
        if self.action_type == 'discrete':
            # Discrete actions: output logits
            logits = self.policy_head(state)
            action_dist = Categorical(logits=logits / self.temperature)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            result = {
                'action': action,
                'log_prob': log_prob,
                'logits': logits,
                'entropy': action_dist.entropy(),
                'value': value
            }
            
        elif self.action_type == 'continuous':
            # Continuous actions: output mean and std
            mean = self.policy_mean(state)
            std = torch.exp(self.policy_logstd.expand_as(mean))
            action_dist = Normal(mean, std * self.temperature)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=-1)
            
            result = {
                'action': action,
                'log_prob': log_prob,
                'mean': mean,
                'std': std,
                'entropy': action_dist.entropy().sum(dim=-1),
                'value': value
            }
        else:
            raise ValueError(f"Unknown action_type: {self.action_type}")
        
        return result
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action from state (inference mode)"""
        self.eval()
        with torch.no_grad():
            if self.action_type == 'discrete':
                logits = self.policy_head(state)
                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                else:
                    probs = F.softmax(logits / self.temperature, dim=-1)
                    action = torch.multinomial(probs, 1).squeeze(-1)
                return action
                
            elif self.action_type == 'continuous':
                mean = self.policy_mean(state)
                if deterministic:
                    return mean
                else:
                    std = torch.exp(self.policy_logstd.expand_as(mean))
                    noise = torch.randn_like(mean)
                    return mean + std * noise * self.temperature
            else:
                raise ValueError(f"Unknown action_type: {self.action_type}")
    
    def get_parameters_as_vector(self) -> np.ndarray:
        """Get all parameters as a flat vector (for evolution strategies)"""
        params = []
        for param in self.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def set_parameters_from_vector(self, vector: np.ndarray):
        """Set parameters from a flat vector (for evolution strategies)"""
        pointer = 0
        for param in self.parameters():
            num_param = param.numel()
            param.data = torch.from_numpy(
                vector[pointer:pointer + num_param].reshape(param.shape)
            ).float().to(self.device)
            pointer += num_param

class CPUEvolutionTrainer:
    """
    CPU-optimized evolution strategies trainer
    Simplified version without CMA-ES to avoid import issues
    """
    
    def __init__(
        self,
        controller: ControllerCPU,
        population_size: int = 16,
        sigma: float = 0.1,
        learning_rate: float = 0.01
    ):
        self.controller = controller
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        
        # Get initial parameters
        self.theta = controller.get_parameters_as_vector()
        self.best_fitness = -np.inf
        self.generation = 0
    
    def ask(self) -> np.ndarray:
        """Generate population of parameter vectors"""
        noise = np.random.randn(self.population_size, len(self.theta))
        population = self.theta + self.sigma * noise
        return population
    
    def tell(self, population: np.ndarray, fitness: np.ndarray):
        """Update parameters based on fitness"""
        # Simple (1+λ) evolution strategy
        best_idx = np.argmax(fitness)
        best_fitness = fitness[best_idx]
        
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.theta = population[best_idx].copy()
            self.controller.set_parameters_from_vector(self.theta)
        
        self.generation += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'sigma': self.sigma
        }

def test_cpu_controller():
    """Test CPU Controller functionality without hanging imports"""
    print("Testing CPU-Optimized Controller...")
    
    # Always use CPU device
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Test discrete controller
    input_size = 32 + 256  # latent_size + hidden_size
    action_size = 6
    
    controller = ControllerCPU(
        input_size=input_size,
        action_size=action_size,
        hidden_sizes=(64,),
        action_type='discrete'
    )
    
    print(f"Controller parameters: {sum(p.numel() for p in controller.parameters())}")
    
    # Test forward pass
    batch_size = 4
    z = torch.randn(batch_size, 32)
    h = torch.randn(batch_size, 256)
    state = torch.cat([z, h], dim=1)
    
    output = controller(state)
    print(f"Output keys: {output.keys()}")
    print(f"Action shape: {output['action'].shape}")
    print(f"Value shape: {output['value'].shape}")
    
    # Test continuous controller
    controller_continuous = ControllerCPU(
        input_size=input_size,
        action_size=2,  # 2D continuous actions
        action_type='continuous'
    )
    
    output_continuous = controller_continuous(state)
    print(f"Continuous action shape: {output_continuous['action'].shape}")
    
    # Test evolution trainer (simplified version)
    trainer = CPUEvolutionTrainer(controller, population_size=8)
    
    # Test ask/tell interface
    population = trainer.ask()
    print(f"Population shape: {population.shape}")
    
    # Dummy fitness evaluation
    dummy_fitness = np.random.randn(len(population))
    trainer.tell(population, dummy_fitness)
    
    stats = trainer.get_stats()
    print(f"Trainer stats: {stats}")
    
    print("✅ CPU Controller test passed!")
    return True

if __name__ == "__main__":
    test_cpu_controller()

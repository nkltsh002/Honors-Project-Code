"""
Controller for World Models

This module implements the Controller (C) component of World Models architecture.
The controller maps the concatenation of latent state z and LSTM hidden state h to actions.

Key features:
- Linear or small MLP architecture for fast inference
- Evolution strategies (CMA-ES) training for robust policy search
- Optional PPO training for comparison
- Temperature-based action selection
- Designed for both discrete and continuous action spaces

Based on Ha & Schmidhuber (2018): "World Models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import cma

class Controller(nn.Module):
    """
    Neural network controller that maps [z_t, h_t] to actions.

    The controller receives:
    - z_t: Current latent state from VAE (captures visual information)
    - h_t: LSTM hidden state from MDN-RNN (captures temporal memory)

    And outputs action probabilities or action values for decision making.

    Architecture choices:
    - Linear: Fast, parameter-efficient, often sufficient for simple tasks
    - MLP: More expressive, better for complex control tasks
    """

    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_sizes: Tuple[int, ...] = (),
        activation: str = 'tanh',
        action_type: str = 'discrete',
        use_bias: bool = True
    ):
        """
        Initialize Controller.

        Args:
            input_size: Size of input [z, h] concatenation
            action_size: Size of action space
            hidden_sizes: Hidden layer sizes (empty for linear controller)
            activation: Activation function ('tanh', 'relu', 'elu')
            action_type: 'discrete' or 'continuous'
            use_bias: Whether to use bias terms
        """
        super().__init__()

        self.input_size = input_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.action_type = action_type

        # Build network layers
        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size, bias=use_bias))

            # Activation function
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'elu':
                layers.append(nn.ELU(inplace=True))
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2, inplace=True))

            prev_size = hidden_size

        # Output layer
        if action_type == 'discrete':
            # For discrete actions: output logits for each action
            layers.append(nn.Linear(prev_size, action_size, bias=use_bias))
        elif action_type == 'continuous':
            # For continuous actions: output mean and log_std
            layers.append(nn.Linear(prev_size, action_size * 2, bias=use_bias))
        else:
            raise ValueError("Unknown action_type: {}".format(action_type))

        self.network = nn.Sequential(*layers)

        # Initialize weights for better training dynamics
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for hidden layers
                if module != self.network[-1]:  # Not output layer
                    nn.init.xavier_normal_(module.weight)
                else:  # Output layer
                    # Smaller initialization for output layer
                    nn.init.xavier_normal_(module.weight, gain=0.1)

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through controller.

        Args:
            z: Latent states (batch_size, latent_size)
            h: LSTM hidden states (batch_size, hidden_size)

        Returns:
            Action logits or parameters
        """
        # Concatenate latent state and hidden state
        x = torch.cat([z, h], dim=-1)

        # Pass through network
        output = self.network(x)

        return output

    def get_action(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        temperature: float = 1.0,
        deterministic: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get action from controller output.

        Args:
            z: Latent state
            h: LSTM hidden state
            temperature: Temperature for action sampling
            deterministic: Whether to use deterministic action selection

        Returns:
            action: Selected action(s)
            log_prob: Log probability of action (for continuous actions)
        """
        output = self.forward(z, h)

        if self.action_type == 'discrete':
            # Apply temperature scaling to logits
            if temperature != 1.0:
                output = output / temperature

            if deterministic:
                # Greedy action selection
                action = torch.argmax(output, dim=-1)
            else:
                # Sample from categorical distribution
                action_dist = Categorical(logits=output)
                action = action_dist.sample()

            return action

        elif self.action_type == 'continuous':
            # Split output into mean and log_std
            mean = output[..., :self.action_size]
            log_std = output[..., self.action_size:]

            # Clamp log_std for numerical stability
            log_std = torch.clamp(log_std, min=-10, max=2)
            std = torch.exp(log_std) * temperature

            if deterministic:
                return mean
            else:
                # Sample from normal distribution
                action_dist = Normal(mean, std)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum(dim=-1)

                return action, log_prob
        else:
            raise ValueError(f"Unknown action_type: {self.action_type}")

    def get_action_probabilities(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Get action probabilities for discrete action spaces.

        Args:
            z: Latent state
            h: LSTM hidden state
            temperature: Temperature for probability scaling

        Returns:
            Action probabilities
        """
        if self.action_type != 'discrete':
            raise ValueError("Action probabilities only available for discrete actions")

        logits = self.forward(z, h)

        if temperature != 1.0:
            logits = logits / temperature

        return F.softmax(logits, dim=-1)

    def get_parameters(self) -> np.ndarray:
        """
        Get flattened parameters for CMA-ES training.

        Returns:
            Flattened parameter array
        """
        params = []
        for param in self.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_parameters(self, params: np.ndarray):
        """
        Set parameters from flattened array for CMA-ES training.

        Args:
            params: Flattened parameter array
        """
        param_idx = 0

        for param in self.parameters():
            param_size = param.numel()
            param_data = params[param_idx:param_idx + param_size]
            param.data = torch.from_numpy(param_data.reshape(param.shape)).float().to(param.device)
            param_idx += param_size

    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

class CMAESController:
    """
    CMA-ES trainer for the Controller.

    Uses Covariance Matrix Adaptation Evolution Strategy to optimize
    controller parameters by maximizing cumulative reward in the environment.
    """

    def __init__(
        self,
        controller: Controller,
        population_size: int = 64,
        sigma: float = 0.5,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize CMA-ES trainer.

        Args:
            controller: Controller network to optimize
            population_size: CMA-ES population size
            sigma: Initial standard deviation
            device: Device for evaluation
        """
        self.controller = controller
        self.device = device
        self.population_size = population_size

        # Initialize CMA-ES
        initial_params = controller.get_parameters()
        self.es = cma.CMAEvolutionStrategy(initial_params, sigma, {
            'popsize': population_size,
            'seed': 42
        })

        # Track training progress
        self.generation = 0
        self.best_fitness = -np.inf
        self.best_params = None
        self.fitness_history = []

    def ask(self) -> np.ndarray:
        """
        Get candidate solutions from CMA-ES.

        Returns:
            Array of candidate parameter vectors
        """
        return np.array(self.es.ask())

    def tell(self, candidates: np.ndarray, fitness_values: np.ndarray):
        """
        Update CMA-ES with fitness evaluations.

        Args:
            candidates: Candidate parameter vectors
            fitness_values: Fitness values (higher is better)
        """
        # CMA-ES minimizes, so negate fitness values
        self.es.tell(candidates.tolist(), (-fitness_values).tolist())

        # Track best solution
        best_idx = np.argmax(fitness_values)
        if fitness_values[best_idx] > self.best_fitness:
            self.best_fitness = fitness_values[best_idx]
            self.best_params = candidates[best_idx].copy()

        self.fitness_history.append(fitness_values)
        self.generation += 1

    def update_controller(self, use_best: bool = True):
        """
        Update controller with current best parameters.

        Args:
            use_best: Whether to use best found params or current mean
        """
        if use_best and self.best_params is not None:
            params = self.best_params
        else:
            params = self.es.result.xbest

        self.controller.set_parameters(params)

    def should_stop(self) -> bool:
        """Check if evolution should stop"""
        stop_result = self.es.stop()
        return bool(stop_result)

    def get_stats(self) -> Dict[str, float]:
        """Get training statistics"""
        if len(self.fitness_history) == 0:
            return {}

        latest_fitness = self.fitness_history[-1]
        return {
            'generation': int(self.generation),
            'best_fitness': float(self.best_fitness),
            'mean_fitness': float(np.mean(latest_fitness)),
            'std_fitness': float(np.std(latest_fitness)),
            'sigma': float(self.es.sigma)
        }

class PPOController:
    """
    PPO trainer for the Controller (alternative to CMA-ES).

    Uses Proximal Policy Optimization for controller training.
    Requires environment interaction and reward signals.
    """

    def __init__(
        self,
        controller: Controller,
        learning_rate: float = 3e-4,
        clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize PPO trainer.

        Args:
            controller: Controller network
            learning_rate: Adam learning rate
            clip_param: PPO clipping parameter
            value_loss_coef: Value function loss coefficient
            entropy_coef: Entropy regularization coefficient
            device: Training device
        """
        self.controller = controller
        self.device = device
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        # Create value network (estimates state values)
        self.value_net = nn.Sequential(
            nn.Linear(controller.input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        ).to(device)

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            controller.parameters(), lr=learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=learning_rate
        )

    def compute_ppo_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss components.

        Args:
            states: Environment states [z, h]
            actions: Actions taken
            old_log_probs: Log probabilities from old policy
            rewards: Environment rewards
            dones: Episode termination flags

        Returns:
            Dictionary with loss components
        """
        # TODO: Implement full PPO loss computation
        # This would require proper advantage estimation and value targets
        # For now, this is a placeholder for the full implementation
        return {
            "policy_loss": torch.tensor(0.0),
            "value_loss": torch.tensor(0.0),
            "entropy_loss": torch.tensor(0.0)
        }

def test_controller():
    """Test Controller functionality"""
    print("Testing Controller...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test discrete controller
    input_size = 32 + 256  # latent_size + hidden_size
    action_size = 6

    controller = Controller(
        input_size=input_size,
        action_size=action_size,
        hidden_sizes=(64,),
        action_type='discrete'
    ).to(device)

    print(f"Discrete Controller has {controller.get_num_parameters():,} parameters")

    # Test forward pass
    batch_size = 8
    z = torch.randn(batch_size, 32, device=device)
    h = torch.randn(batch_size, 256, device=device)

    logits = controller(z, h)
    print(f"Controller output shape: {logits.shape}")

    # Test action selection
    actions = controller.get_action(z, h, temperature=1.0, deterministic=False)
    print(f"Sampled actions: {actions}")

    # Test action probabilities
    probs = controller.get_action_probabilities(z, h, temperature=1.0)
    print(f"Action probabilities shape: {probs.shape}")

    # Test parameter manipulation for CMA-ES
    params = controller.get_parameters()
    print(f"Parameter vector size: {len(params)}")

    # Test continuous controller
    continuous_controller = Controller(
        input_size=input_size,
        action_size=3,
        hidden_sizes=(),
        action_type='continuous'
    ).to(device)

    print(f"Continuous Controller has {continuous_controller.get_num_parameters():,} parameters")

    actions, log_probs = continuous_controller.get_action(z, h[:, :3], deterministic=False)
    print(f"Continuous actions shape: {actions.shape}")
    print(f"Log probabilities shape: {log_probs.shape}")

    # Test CMA-ES trainer
    cmaes_trainer = CMAESController(
        controller=controller,
        population_size=8,  # Small for testing
        sigma=0.5,
        device=device
    )

    # Simulate one CMA-ES iteration
    candidates = cmaes_trainer.ask()
    print(f"CMA-ES candidates shape: {candidates.shape}")

    # Dummy fitness evaluation
    dummy_fitness = np.random.randn(len(candidates))
    cmaes_trainer.tell(candidates, dummy_fitness)

    stats = cmaes_trainer.get_stats()
    print(f"CMA-ES stats: {stats}")

    print("Controller test passed!")

if __name__ == "__main__":
    test_controller()

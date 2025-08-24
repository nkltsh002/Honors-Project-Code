# MDN-RNN Implementation for World Models

This directory contains a comprehensive implementation of the Memory Model (M) component from the World Models architecture, implemented as a Mixture Density Network with LSTM (MDN-RNN).

## Overview

The MDN-RNN serves as the temporal dynamics model in the World Models framework, learning to predict future latent states given current latent observations and actions. It uses a mixture of Gaussians to capture the stochastic nature of environment dynamics.

## Components Implemented

### 1. Core MDN-RNN Model (`models/mdnrnn.py`)

**MDNRNN Class Features:**
- LSTM-based temporal processing with configurable hidden size
- Mixture Density Network head with multiple Gaussian components
- Optional reward and done signal prediction heads
- Support for both training and inference modes
- Proper gradient flow and numerical stability

**Key Architecture Details:**
- Input: Concatenated latent observations (z) and actions (a)
- LSTM core: Single-layer LSTM for sequence processing
- MDN head: Outputs mixture weights, means, and log-std deviations
- Output dimensions: `num_mixtures * (z_dim + 2)` for MDN parameters

**Training Infrastructure:**
- `SequenceDataset`: Handles batching of variable-length sequences
- `mdn_loss_function`: Numerically stable MDN loss with log-sum-exp
- `train_mdnrnn`: Complete training loop with teacher forcing
- Gradient clipping and proper optimization setup

**Sampling and Inference:**
- `sample_from_mdn`: Temperature-controlled sampling from mixture distributions
- `rollout_in_dream`: Generate rollouts using learned dynamics
- Support for both deterministic and stochastic sampling modes

### 2. Dataset Utilities (`tools/dataset_utils.py`)

**FramesToLatentConverter:**
- Converts raw frames to VAE latent representations
- Handles frame preprocessing (resize, normalize)
- Integration with trained ConvVAE models
- Batch processing for efficiency

**TrajectoryDataset:**
- Loads and manages trajectory sequences for training
- Support for variable-length episodes
- Automatic sequence windowing for MDN-RNN training
- Integration with reward and done signals

**Data Processing Pipeline:**
- `preprocess_trajectories`: Convert raw data to latent space
- `create_mdnrnn_dataloader`: One-step DataLoader creation
- Custom collate functions for sequence batching
- Validation utilities for dataset integrity

### 3. Dream Environment (`tools/dream_env.py`)

**DreamEnvironment Class:**
- Gym-like interface for model-based simulation
- Uses trained VAE + MDN-RNN for dynamics prediction
- Operates entirely in latent space for efficiency
- Support for rendering and episode management

**Key Features:**
- Reset with custom initial observations
- Step function with action input and state transition
- Temperature-controlled exploration vs exploitation
- Integration with existing RL frameworks

**DreamAgent:**
- Simple policy implementations (random, constant, neural)
- Compatible with standard RL agent interfaces
- Easy integration for planning and control algorithms

**Planning Utilities:**
- `DreamRollout`: Generate rollouts in dream environment
- `DreamPlanner`: Random shooting and other planning algorithms
- Statistics and analysis tools for rollout evaluation

### 4. Validation and Testing (`validate_mdnrnn_simple.py`)

**Comprehensive Testing Suite:**
- Model architecture validation
- Forward pass testing with various batch sizes
- MDN sampling functionality verification
- Dream environment integration testing
- Component compatibility checks

**Test Coverage:**
- Shape validation for all tensor operations
- Numerical stability checks (NaN/Inf detection)
- Memory usage and performance testing
- Cross-component integration validation

## Mathematical Foundations

### MDN Loss Function

The MDN loss uses a mixture of Gaussians to model the conditional probability distribution:

```
P(z_{t+1} | z_t, a_t) = Σ(i=1 to K) π_i * N(μ_i, σ_i²)
```

Where:
- K = number of mixture components
- π_i = mixture weights (softmax normalized)
- μ_i = component means
- σ_i = component standard deviations (exp of log_std)

The loss function maximizes the log-likelihood:
```
L = -log(Σ(i=1 to K) π_i * N(z_{target} | μ_i, σ_i²))
```

### Temperature Sampling

Temperature-controlled sampling allows balancing exploration vs exploitation:
- Low temperature (T → 0): More deterministic, exploitation
- High temperature (T >> 1): More stochastic, exploration

Temperature is applied to mixture weights before softmax:
```
π_i = softmax(logit_i / T)
```

## Usage Examples

### Basic Model Creation and Training

```python
from models.mdnrnn import MDNRNN, train_mdnrnn
from tools.dataset_utils import create_mdnrnn_dataloader

# Create model
model = MDNRNN(
    z_dim=32,        # VAE latent dimension
    action_dim=3,    # Action space dimension
    rnn_size=256,    # LSTM hidden size
    num_mixtures=5   # Number of mixture components
)

# Load data
dataloader = create_mdnrnn_dataloader(
    data_dir="path/to/trajectories",
    batch_size=32,
    sequence_length=50
)

# Training configuration
config = {
    'lr': 0.001,
    'epochs': 100,
    'device': 'cuda',
    'save_dir': 'checkpoints/'
}

# Train the model
train_mdnrnn(model, dataloader, config)
```

### Dream Environment Usage

```python
from tools.dream_env import DreamEnvironment, DreamAgent, DreamRollout

# Create dream environment
env = DreamEnvironment(
    vae_model_path="checkpoints/vae_best.pth",
    mdnrnn_model_path="checkpoints/mdnrnn_best.pth",
    action_space_size=3,
    temperature=1.0
)

# Create agent
agent = DreamAgent(action_dim=3, policy_type='random')

# Generate rollout
rollout_data = DreamRollout.generate_rollout(
    env=env,
    agent=agent,
    max_steps=1000
)

# Analyze results
stats = DreamRollout.rollout_statistics(rollout_data)
print(f"Episode length: {stats['episode_length']}")
print(f"Total reward: {stats['total_reward']:.3f}")
```

### Model-Based Planning

```python
from tools.dream_env import DreamPlanner

# Create planner
planner = DreamPlanner(env)

# Plan action sequence
initial_obs = torch.randn(32)  # Current latent state
best_action, value = planner.random_shooting(
    initial_obs=initial_obs,
    horizon=20,
    num_candidates=1000
)

print(f"Best action: {best_action}")
print(f"Expected value: {value:.3f}")
```

## File Structure

```
world_models/
├── models/
│   └── mdnrnn.py              # Core MDN-RNN implementation
├── tools/
│   ├── dataset_utils.py       # Data loading and preprocessing
│   └── dream_env.py          # Dream environment wrapper
└── validate_mdnrnn_simple.py # Testing and validation
```

## Key Design Decisions

### 1. Modular Architecture
- Separate concerns: model definition, training, data handling
- Easy to extend and modify individual components
- Clean interfaces between modules

### 2. Numerical Stability
- Log-sum-exp trick in loss computation
- Gradient clipping during training
- Proper initialization of parameters
- Robust handling of edge cases

### 3. Flexible Configuration
- Configurable model dimensions and hyperparameters
- Support for different training regimes
- Optional components (reward/done prediction)
- Easy integration with existing codebases

### 4. Comprehensive Testing
- Unit tests for individual components
- Integration tests for full pipeline
- Validation scripts for quick verification
- Performance and memory usage checks

## Performance Considerations

### Memory Usage
- Efficient sequence batching to minimize memory overhead
- Gradient checkpointing for long sequences
- Lazy loading of large datasets
- Proper cleanup of intermediate tensors

### Computational Efficiency
- Vectorized operations throughout
- CUDA support for GPU acceleration
- Batch processing for improved throughput
- Optimized sampling algorithms

### Training Stability
- Teacher forcing for stable training
- Gradient clipping to prevent exploding gradients
- Learning rate scheduling
- Early stopping and model checkpointing

## Integration with World Models

This MDN-RNN implementation is designed to integrate seamlessly with the complete World Models architecture:

1. **Vision Model (V)**: ConvVAE for learning latent representations
2. **Memory Model (M)**: This MDN-RNN for temporal dynamics
3. **Controller (C)**: Policy network for action selection

The latent space learned by the VAE provides the input observations for the MDN-RNN, which then learns to predict future latent states. The Controller can then be trained using either:
- Real environment experience
- Dream environment rollouts
- A combination of both (hybrid training)

## Future Extensions

Potential improvements and extensions:
1. **Hierarchical MDN-RNN**: Multiple time scales
2. **Attention mechanisms**: Better long-term dependencies
3. **Variational inference**: Improved uncertainty quantification
4. **Multi-modal prediction**: Handle multiple possible futures
5. **Transfer learning**: Adapt to new environments quickly

## Troubleshooting

### Common Issues

1. **NaN losses during training**
   - Check learning rate (try lower values)
   - Verify data normalization
   - Ensure proper gradient clipping

2. **Poor sample quality**
   - Adjust temperature parameter
   - Increase number of mixture components
   - Check data quality and diversity

3. **Memory issues**
   - Reduce batch size or sequence length
   - Use gradient checkpointing
   - Monitor GPU memory usage

4. **Slow training**
   - Use smaller model for debugging
   - Implement data prefetching
   - Profile code for bottlenecks

### Performance Tuning

- Start with smaller models and gradually increase complexity
- Use validation set to monitor overfitting
- Experiment with different mixture component counts
- Consider curriculum learning for complex environments

## Citation and References

This implementation is based on the World Models paper:
```
@article{ha2018world,
  title={World Models},
  author={Ha, David and Schmidhuber, J{\"u}rgen},
  journal={arXiv preprint arXiv:1803.10122},
  year={2018}
}
```

For technical details on Mixture Density Networks:
```
@article{bishop1994mixture,
  title={Mixture density networks},
  author={Bishop, Christopher M},
  year={1994},
  publisher={Aston University}
}
```

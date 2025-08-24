"""
Mixture Density Network + LSTM (MDN-RNN) for World Models

This module implements the Memory Model (M) component of World Models architecture.
The MDN-RNN learns a temporal model of the latent space dynamics:
P(z_{t+1} | a_t, z_t, h_t)

Key features:
- LSTM for temporal memory and dynamics
- Mixture Density Network for multimodal predictions
- Handles variable-length sequences  
- Temperature sampling for exploration vs exploitation
- GPU-optimized implementation

Based on Ha & Schmidhuber (2018): "World Models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, MixtureSameFamily
from typing import Tuple, Dict, Any, Optional
import numpy as np
import math

class MDNRNN(nn.Module):
    """
    Mixture Density Network + LSTM for temporal dynamics modeling.
    
    The MDN-RNN learns to predict the next latent state z_{t+1} given:
    - Current latent state z_t (from VAE)
    - Action a_t
    - Hidden state h_t (from LSTM memory)
    
    The output is a mixture of Gaussians representing the distribution
    over possible next latent states, enabling stochastic prediction.
    """
    
    def __init__(
        self,
        latent_size: int = 32,
        action_size: int = 3,
        hidden_size: int = 256,
        num_layers: int = 1,
        num_mixtures: int = 5,
        dropout: float = 0.1
    ):
        """
        Initialize MDN-RNN.
        
        Args:
            latent_size: Size of VAE latent vectors z
            action_size: Size of action space
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            num_mixtures: Number of Gaussian mixtures in MDN
            dropout: Dropout probability for regularization
        """
        super().__init__()
        
        self.latent_size = latent_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures
        self.dropout = dropout
        
        # Input size: concatenation of latent vector and action
        self.input_size = latent_size + action_size
        
        # LSTM for temporal modeling
        # The LSTM learns to capture temporal dependencies and dynamics
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Mixture Density Network heads
        # These map LSTM output to mixture parameters
        
        # Mixture weights (logits for categorical distribution)
        self.mixture_weights = nn.Linear(hidden_size, num_mixtures)
        
        # Gaussian means (one per mixture component)
        self.mixture_means = nn.Linear(hidden_size, num_mixtures * latent_size)
        
        # Gaussian log-variances (one per mixture component)
        # We predict log-variance for numerical stability
        self.mixture_logvars = nn.Linear(hidden_size, num_mixtures * latent_size)
        
        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights for better training dynamics
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using appropriate schemes for different layer types"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:  # LSTM input weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # LSTM hidden weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:  # LSTM biases
                param.data.fill_(0)
                # Initialize forget gate bias to 1 for better gradient flow
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1)
            elif param.dim() > 1:  # Linear layer weights
                nn.init.xavier_normal_(param.data)
            else:  # Linear layer biases
                param.data.fill_(0)
    
    def forward(
        self, 
        z: torch.Tensor, 
        actions: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Forward pass through MDN-RNN.
        
        Args:
            z: Latent states (batch_size, seq_len, latent_size)
            actions: Actions (batch_size, seq_len, action_size)
            hidden: Initial LSTM hidden state (optional)
            
        Returns:
            Dictionary containing:
                - 'mixture_weights': Mixture component weights
                - 'mixture_means': Mixture component means
                - 'mixture_logvars': Mixture component log-variances
                - 'hidden': Final LSTM hidden state
        """
        batch_size, seq_len = z.size(0), z.size(1)
        
        # Concatenate latent states and actions as LSTM input
        # Input represents (state, action) pairs for dynamics learning
        lstm_input = torch.cat([z, actions], dim=-1)
        
        # Pass through LSTM
        # LSTM learns temporal dependencies and maintains memory
        lstm_output, hidden_state = self.lstm(lstm_input, hidden)
        assert isinstance(hidden_state, tuple) and len(hidden_state) == 2
        
        # Apply dropout for regularization
        lstm_output = self.dropout_layer(lstm_output)
        
        # Reshape for mixture parameter prediction
        lstm_flat = lstm_output.reshape(-1, self.hidden_size)
        
        # Predict mixture parameters
        
        # Mixture weights (probability of each component)
        mixture_weights = self.mixture_weights(lstm_flat)  # (batch*seq, num_mixtures)
        mixture_weights = F.softmax(mixture_weights, dim=-1)
        
        # Mixture means (center of each Gaussian component)
        mixture_means = self.mixture_means(lstm_flat)  # (batch*seq, num_mixtures * latent_size)
        mixture_means = mixture_means.view(-1, self.num_mixtures, self.latent_size)
        
        # Mixture log-variances (spread of each Gaussian component)
        mixture_logvars = self.mixture_logvars(lstm_flat)  # (batch*seq, num_mixtures * latent_size)
        mixture_logvars = mixture_logvars.view(-1, self.num_mixtures, self.latent_size)
        
        # Clamp log-variances to prevent numerical instability
        mixture_logvars = torch.clamp(mixture_logvars, min=-10, max=10)
        
        # Reshape back to sequence format
        mixture_weights = mixture_weights.view(batch_size, seq_len, self.num_mixtures)
        mixture_means = mixture_means.view(batch_size, seq_len, self.num_mixtures, self.latent_size)
        mixture_logvars = mixture_logvars.view(batch_size, seq_len, self.num_mixtures, self.latent_size)
        
        return {
            'mixture_weights': mixture_weights,
            'mixture_means': mixture_means,
            'mixture_logvars': mixture_logvars,
            'hidden': hidden_state,
            'lstm_output': lstm_output  # For controller input
        }
    
    def compute_loss(
        self,
        z: torch.Tensor,
        actions: torch.Tensor, 
        targets: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MDN loss: negative log-likelihood of target under predicted mixture.
        
        The loss encourages the model to assign high probability to the actual
        next latent state under the predicted mixture distribution.
        
        Args:
            z: Current latent states (batch_size, seq_len, latent_size)
            actions: Actions taken (batch_size, seq_len, action_size)
            targets: Target next latent states (batch_size, seq_len, latent_size)
            hidden: Initial LSTM hidden state
            
        Returns:
            Dictionary with loss components and metrics
        """
        # Forward pass
        outputs = self.forward(z, actions, hidden)
        
        mixture_weights = outputs['mixture_weights']  # (B, T, K)
        mixture_means = outputs['mixture_means']      # (B, T, K, D)
        mixture_logvars = outputs['mixture_logvars']  # (B, T, K, D)
        
        batch_size, seq_len = targets.size(0), targets.size(1)
        
        # Expand targets for mixture evaluation
        targets_expanded = targets.unsqueeze(2).expand(-1, -1, self.num_mixtures, -1)  # (B, T, K, D)
        
        # Compute log-probabilities for each mixture component
        # log p(z_{t+1} | component k) for each k
        mixture_stds = torch.exp(0.5 * mixture_logvars)
        
        # Gaussian log-probability: -0.5 * [(x-mu)/sigma]^2 - log(sigma) - 0.5*log(2*pi)
        log_probs = (
            -0.5 * ((targets_expanded - mixture_means) / mixture_stds) ** 2
            - 0.5 * mixture_logvars
            - 0.5 * math.log(2 * math.pi)
        ).sum(dim=-1)  # Sum over latent dimensions: (B, T, K)
        
        # Compute mixture log-probabilities
        # log p(z_{t+1}) = log(sum_k w_k * p(z_{t+1} | component k))
        log_mixture_weights = torch.log(mixture_weights + 1e-8)  # Add small epsilon for stability
        log_mixture_probs = log_mixture_weights + log_probs
        
        # LogSumExp for numerical stability
        max_log_prob = torch.max(log_mixture_probs, dim=-1, keepdim=True)[0]
        mixture_probs = torch.exp(log_mixture_probs - max_log_prob).sum(dim=-1)
        log_prob = max_log_prob.squeeze(-1) + torch.log(mixture_probs + 1e-8)
        
        # Negative log-likelihood loss
        nll_loss = -log_prob.mean()
        
        # Compute additional metrics
        with torch.no_grad():
            # Prediction error (distance to closest mixture component)
            distances = torch.norm(targets_expanded - mixture_means, dim=-1)  # (B, T, K)
            min_distances = torch.min(distances, dim=-1)[0]  # (B, T)
            mean_prediction_error = min_distances.mean()
            
            # Mixture entropy (measure of prediction uncertainty)
            mixture_entropy = -(mixture_weights * log_mixture_weights).sum(dim=-1).mean()
            
            # Dominant mixture component (which component is used most)
            dominant_mixture = torch.argmax(mixture_weights, dim=-1).float().mean()
        
        return {
            'total_loss': nll_loss,
            'nll_loss': nll_loss,
            'prediction_error': mean_prediction_error,
            'mixture_entropy': mixture_entropy,
            'dominant_mixture': dominant_mixture,
            'hidden': outputs['hidden']
        }
    
    def sample(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample next latent states from predicted mixture distribution.
        
        This is used during environment simulation for the controller training.
        Temperature controls the exploration vs exploitation trade-off.
        
        Args:
            z: Current latent states (batch_size, seq_len, latent_size)
            actions: Actions to take (batch_size, seq_len, action_size)
            hidden: LSTM hidden state
            temperature: Sampling temperature (higher = more exploration)
            
        Returns:
            next_z: Sampled next latent states
            hidden_state: Updated LSTM hidden state
        """
        self.eval()  # Set to eval mode for deterministic behavior
        
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(z, actions, hidden)
            
            mixture_weights = outputs['mixture_weights']
            mixture_means = outputs['mixture_means'] 
            mixture_logvars = outputs['mixture_logvars']
            hidden_state = outputs['hidden']
            
            batch_size, seq_len = z.size(0), z.size(1)
            
            # Apply temperature to mixture weights for exploration control
            if temperature != 1.0:
                mixture_weights = F.softmax(
                    torch.log(mixture_weights + 1e-8) / temperature, dim=-1
                )
            
            # Sample mixture component for each timestep
            mixture_dist = Categorical(mixture_weights.view(-1, self.num_mixtures))
            selected_mixtures = mixture_dist.sample()  # (batch*seq,)
            
            # Get parameters for selected mixture components
            batch_indices = torch.arange(batch_size * seq_len, device=z.device)
            selected_means = mixture_means.view(-1, self.num_mixtures, self.latent_size)[
                batch_indices, selected_mixtures
            ]  # (batch*seq, latent_size)
            selected_logvars = mixture_logvars.view(-1, self.num_mixtures, self.latent_size)[
                batch_indices, selected_mixtures
            ]  # (batch*seq, latent_size)
            
            # Sample from selected Gaussian components
            selected_stds = torch.exp(0.5 * selected_logvars) * temperature  # Scale by temperature
            eps = torch.randn_like(selected_means)
            next_z = selected_means + selected_stds * eps
            
            # Reshape back to sequence format
            next_z = next_z.view(batch_size, seq_len, self.latent_size)
            
            return next_z, hidden_state
    
    def predict_mean(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict mean next latent states (deterministic prediction).
        
        Uses the weighted average of mixture components for stable prediction.
        Useful for evaluation and when deterministic behavior is desired.
        
        Args:
            z: Current latent states
            actions: Actions to take
            hidden: LSTM hidden state
            
        Returns:
            predicted_z: Predicted next latent states (weighted mixture mean)
            hidden_state: Updated LSTM hidden state
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(z, actions, hidden)
            
            mixture_weights = outputs['mixture_weights']  # (B, T, K)
            mixture_means = outputs['mixture_means']      # (B, T, K, D)
            hidden_state = outputs['hidden']
            
            # Weighted average of mixture components
            predicted_z = (mixture_weights.unsqueeze(-1) * mixture_means).sum(dim=2)
            
            return predicted_z, hidden_state
    
    def get_hidden_state(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Get LSTM hidden states for controller input.
        
        The hidden state h contains the memory of past experiences and
        is used by the controller along with the current latent state.
        
        Args:
            z: Latent states
            actions: Actions
            hidden: Initial LSTM hidden state
            
        Returns:
            LSTM hidden states for controller
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(z, actions, hidden)
            # Return the cell state (c) which contains long-term memory
            return outputs['lstm_output']

def create_sequence_dataset(
    latent_sequences: torch.Tensor,
    action_sequences: torch.Tensor,
    sequence_length: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create training sequences for MDN-RNN from rollout data.
    
    Converts rollout data into (input, target) pairs for supervised learning.
    Each input is [z_t, a_t] and target is z_{t+1}.
    
    Args:
        latent_sequences: Latent state sequences (num_rollouts, rollout_length, latent_size)
        action_sequences: Action sequences (num_rollouts, rollout_length, action_size)
        sequence_length: Length of training sequences
        
    Returns:
        z_inputs: Input latent sequences
        action_inputs: Input action sequences  
        z_targets: Target latent sequences (shifted by 1 timestep)
    """
    num_rollouts, rollout_length = latent_sequences.shape[:2]
    
    z_inputs = []
    action_inputs = []
    z_targets = []
    
    for rollout_idx in range(num_rollouts):
        z_rollout = latent_sequences[rollout_idx]
        action_rollout = action_sequences[rollout_idx]
        
        # Extract sequences of specified length
        for start_idx in range(0, rollout_length - sequence_length, sequence_length // 2):
            end_idx = start_idx + sequence_length
            
            if end_idx < rollout_length:
                # Input: [z_0, ..., z_{T-1}] and [a_0, ..., a_{T-1}]
                z_seq = z_rollout[start_idx:end_idx]
                action_seq = action_rollout[start_idx:end_idx]
                
                # Target: [z_1, ..., z_T]
                z_target = z_rollout[start_idx+1:end_idx+1]
                
                z_inputs.append(z_seq)
                action_inputs.append(action_seq)
                z_targets.append(z_target)
    
    return (
        torch.stack(z_inputs),
        torch.stack(action_inputs),
        torch.stack(z_targets)
    )

def test_mdnrnn():
    """Test MDN-RNN functionality"""
    print("Testing MDN-RNN...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    mdnrnn = MDNRNN(
        latent_size=32,
        action_size=3,
        hidden_size=256,
        num_mixtures=5
    ).to(device)
    
    print(f"Model has {sum(p.numel() for p in mdnrnn.parameters()):,} parameters")
    
    # Test with dummy sequences
    batch_size, seq_len = 4, 50
    latent_size, action_size = 32, 3
    
    z = torch.randn(batch_size, seq_len, latent_size, device=device)
    actions = torch.randn(batch_size, seq_len, action_size, device=device)
    targets = torch.randn(batch_size, seq_len, latent_size, device=device)
    
    # Forward pass
    outputs = mdnrnn(z, actions)
    print(f"Input shape: {z.shape}")
    print(f"Mixture weights shape: {outputs['mixture_weights'].shape}")
    print(f"Mixture means shape: {outputs['mixture_means'].shape}")
    
    # Loss computation
    losses = mdnrnn.compute_loss(z, actions, targets)
    print(f"NLL loss: {losses['nll_loss'].item():.4f}")
    print(f"Prediction error: {losses['prediction_error'].item():.4f}")
    
    # Sampling
    next_z, hidden = mdnrnn.sample(z[:, :10], actions[:, :10], temperature=1.0)
    print(f"Sampled next states shape: {next_z.shape}")
    
    # Mean prediction
    pred_z, hidden = mdnrnn.predict_mean(z[:, :10], actions[:, :10])
    print(f"Predicted states shape: {pred_z.shape}")
    
    print("MDN-RNN test passed!")

if __name__ == "__main__":
    test_mdnrnn()

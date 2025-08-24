"""
Mixture Density Network + LSTM (MDN-RNN) for World Models

This module implements the Memory Model (M) component of World Models architecture.
The MDN-RNN learns to predict the next latent state z_{t+1} given current state z_t
and action a_t, modeling temporal dynamics in the compressed latent space.

Architecture:
- Input: Concatenated [z_t, a_t] where z_t is latent vector from VAE
- Core: Single-layer LSTM that processes sequential [z, a] inputs  
- MDN Head: Outputs mixture of Gaussians parameters (pi, mu, sigma)
- Optional: Reward prediction and done probability prediction

Key Features:
- Mixture Density Network with configurable number of components
- Temperature-controlled sampling for exploration vs exploitation
- Teacher forcing during training for stable convergence
- Dream environment wrapper for model-based control
- Compatible with PyTorch 2.x and CUDA acceleration

Mathematical Foundation:
- MDN models p(z_{t+1} | z_t, a_t, h_t) as mixture of K Gaussians
- p(z_{t+1}) = Σ π_k * N(z_{t+1} | μ_k, σ_k²)
- Negative log-likelihood loss: -log(Σ π_k * N(z_{t+1} | μ_k, σ_k²))

Usage Example:
    # 1. Load VAE and encode frames to latents
    vae = ConvVAE(latent_size=32)
    vae.load_state_dict(torch.load('vae.pth'))
    
    # 2. Prepare sequence data
    from tools.dataset_utils import create_sequence_dataset
    dataset = create_sequence_dataset(frames, actions, vae)
    
    # 3. Train MDN-RNN
    mdnrnn = MDNRNN(z_dim=32, action_dim=3, rnn_size=256, num_mixtures=5)
    train_mdnrnn(dataset, mdnrnn, config={'epochs': 20, 'lr': 1e-3})
    
    # 4. Use dream environment
    from tools.dream_env import DreamEnv
    dream_env = DreamEnv(mdnrnn, vae, temperature=1.0)
    obs = dream_env.reset()
    obs, reward, done, info = dream_env.step(action)

Based on Ha & Schmidhuber (2018): "World Models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Optional, Any
import numpy as np
import math
import os
from pathlib import Path


# Default hyperparameters for MDN-RNN training and inference
DEFAULTS = {
    'z_dim': 32,
    'action_dim': 3,
    'rnn_size': 256, 
    'num_mixtures': 5,
    'lr': 1e-3,
    'batch_size': 64,
    'seq_len': 100,
    'epochs': 20,
    'teacher_forcing_ratio': 0.9,
    'temperature': 1.0,
    'gradient_clip': 1.0,
    'predict_reward': False,
    'predict_done': False,
    'reward_loss_weight': 1.0,
    'done_loss_weight': 1.0,
    'checkpoint_dir': './checkpoints/mdnrnn',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


class MDNRNN(nn.Module):
    """
    Mixture Density Network + LSTM for temporal dynamics modeling.
    
    This model learns to predict the next latent state z_{t+1} given the current
    state z_t and action a_t, using a mixture of Gaussians to capture uncertainty
    and multimodality in the dynamics.
    
    Args:
        z_dim (int): Dimensionality of latent space (from VAE)
        action_dim (int): Dimensionality of action space
        rnn_size (int): Hidden size of LSTM
        num_mixtures (int): Number of mixture components in MDN
        predict_reward (bool): Whether to predict scalar rewards
        predict_done (bool): Whether to predict episode termination
    """
    
    def __init__(
        self,
        z_dim: int = 32,
        action_dim: int = 3,
        rnn_size: int = 256,
        num_mixtures: int = 5,
        predict_reward: bool = False,
        predict_done: bool = False
    ):
        super().__init__()
        
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.rnn_size = rnn_size
        self.num_mixtures = num_mixtures
        self.predict_reward = predict_reward
        self.predict_done = predict_done
        
        # Input size: concatenated [z_t, a_t]
        self.input_size = z_dim + action_dim
        
        # =====================================================================
        # LSTM CORE: Processes sequential [z, a] inputs
        # =====================================================================
        
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=rnn_size,
            batch_first=True,
            dropout=0.0  # Single layer, no dropout needed
        )
        
        # =====================================================================
        # MIXTURE DENSITY NETWORK HEAD
        # =====================================================================
        
        # MDN outputs: mixture weights (pi), means (mu), log-variances (log_sigma)
        self.mdn_pi = nn.Linear(rnn_size, num_mixtures)  # Mixture weights
        self.mdn_mu = nn.Linear(rnn_size, num_mixtures * z_dim)  # Means
        self.mdn_sigma = nn.Linear(rnn_size, num_mixtures * z_dim)  # Log-stds
        
        # =====================================================================
        # OPTIONAL PREDICTION HEADS
        # =====================================================================
        
        if predict_reward:
            self.reward_head = nn.Linear(rnn_size, 1)  # Scalar reward prediction
            
        if predict_done:
            self.done_head = nn.Linear(rnn_size, 1)  # Binary done prediction
        
        # Initialize weights for stable training
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights using Xavier/Glorot initialization.
        Special initialization for LSTM to prevent vanishing gradients.
        """
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                # Initialize forget gate bias to 1 for better gradient flow
                nn.init.constant_(param, 0)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)
        
        # Initialize MDN heads
        for module in [self.mdn_pi, self.mdn_mu, self.mdn_sigma]:
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
        
        # Initialize optional heads
        if self.predict_reward:
            nn.init.xavier_normal_(self.reward_head.weight)
            nn.init.constant_(self.reward_head.bias, 0)
            
        if self.predict_done:
            nn.init.xavier_normal_(self.done_head.weight)
            nn.init.constant_(self.done_head.bias, 0)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden state (h0, c0) with appropriate shapes.
        
        Args:
            batch_size (int): Batch size for initialization
            device (torch.device): Device to place tensors on
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Initial hidden and cell states
                - h0: (1, batch_size, rnn_size) - initial hidden state
                - c0: (1, batch_size, rnn_size) - initial cell state
        """
        h0 = torch.zeros(1, batch_size, self.rnn_size, device=device)
        c0 = torch.zeros(1, batch_size, self.rnn_size, device=device)
        return h0, c0
    
    def forward(
        self, 
        z: torch.Tensor, 
        actions: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MDN-RNN.
        
        Args:
            z (torch.Tensor): Latent states (batch_size, seq_len, z_dim)
            actions (torch.Tensor): Actions (batch_size, seq_len, action_dim) 
            hidden (Tuple, optional): Initial LSTM hidden state (h0, c0)
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - pi: Mixture weights (batch_size, seq_len, num_mixtures)
                - mu: Mixture means (batch_size, seq_len, num_mixtures, z_dim)
                - sigma: Mixture std devs (batch_size, seq_len, num_mixtures, z_dim)
                - hidden: Final LSTM hidden state (h_n, c_n)
                - reward (optional): Predicted rewards (batch_size, seq_len, 1)
                - done (optional): Predicted done probs (batch_size, seq_len, 1)
        """
        batch_size, seq_len = z.size(0), z.size(1)
        
        # Concatenate latent states and actions: [z_t, a_t]
        lstm_input = torch.cat([z, actions], dim=-1)  # (batch_size, seq_len, z_dim + action_dim)
        
        # Pass through LSTM
        if hidden is None:
            hidden = self.init_hidden(batch_size, z.device)
        
        lstm_out, hidden_final = self.lstm(lstm_input, hidden)  # (batch_size, seq_len, rnn_size)
        
        # =====================================================================
        # MIXTURE DENSITY NETWORK OUTPUTS
        # =====================================================================
        
        # Mixture weights (softmax normalized)
        pi_logits = self.mdn_pi(lstm_out)  # (batch_size, seq_len, num_mixtures)
        pi = F.softmax(pi_logits, dim=-1)  # Ensure weights sum to 1
        
        # Mixture means (no activation, can be any real value)
        mu = self.mdn_mu(lstm_out)  # (batch_size, seq_len, num_mixtures * z_dim)
        mu = mu.view(batch_size, seq_len, self.num_mixtures, self.z_dim)
        
        # Mixture standard deviations (exp ensures positive values)
        log_sigma = self.mdn_sigma(lstm_out)  # (batch_size, seq_len, num_mixtures * z_dim)
        log_sigma = log_sigma.view(batch_size, seq_len, self.num_mixtures, self.z_dim)
        # Clamp log_sigma to prevent numerical instability
        log_sigma = torch.clamp(log_sigma, min=-20, max=20)
        sigma = torch.exp(log_sigma)
        
        # Prepare output dictionary
        outputs = {
            'pi': pi,
            'mu': mu, 
            'sigma': sigma,
            'hidden': hidden_final
        }
        
        # =====================================================================
        # OPTIONAL PREDICTION OUTPUTS
        # =====================================================================
        
        if self.predict_reward:
            reward = self.reward_head(lstm_out)  # (batch_size, seq_len, 1)
            outputs['reward'] = reward
            
        if self.predict_done:
            done_logits = self.done_head(lstm_out)  # (batch_size, seq_len, 1) 
            done_prob = torch.sigmoid(done_logits)  # Convert to probability
            outputs['done'] = done_prob
        
        return outputs
    
    def get_device(self) -> torch.device:
        """Get the device this model is on."""
        return next(self.parameters()).device


class SequenceDataset(Dataset):
    """
    Dataset for sequential data used in MDN-RNN training.
    
    Handles batching of variable-length sequences and provides
    proper padding and masking for efficient training.
    """
    
    def __init__(
        self,
        latents: np.ndarray,
        actions: np.ndarray,
        rewards: Optional[np.ndarray] = None,
        dones: Optional[np.ndarray] = None,
        seq_len: int = 100
    ):
        """
        Initialize sequence dataset.
        
        Args:
            latents (np.ndarray): Latent states (num_episodes, episode_len, z_dim)
            actions (np.ndarray): Actions (num_episodes, episode_len, action_dim)
            rewards (np.ndarray, optional): Rewards (num_episodes, episode_len)
            dones (np.ndarray, optional): Done flags (num_episodes, episode_len)
            seq_len (int): Maximum sequence length for training
        """
        self.latents = latents
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.seq_len = seq_len
        
        # Create sequence indices
        self.sequences = self._create_sequences()
    
    def _create_sequences(self) -> List[Tuple[int, int, int]]:
        """
        Create list of valid sequence indices.
        
        Returns:
            List of tuples (episode_idx, start_idx, actual_length)
        """
        sequences = []
        
        for ep_idx in range(len(self.latents)):
            episode_len = len(self.latents[ep_idx])
            
            # Create overlapping sequences from this episode
            for start_idx in range(episode_len - 1):  # -1 because we need next state
                end_idx = min(start_idx + self.seq_len, episode_len - 1)
                actual_len = end_idx - start_idx
                
                if actual_len > 1:  # Need at least 2 timesteps
                    sequences.append((ep_idx, start_idx, actual_len))
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sequence sample for training.
        
        Returns:
            Dictionary containing input sequences and targets
        """
        ep_idx, start_idx, seq_len = self.sequences[idx]
        end_idx = start_idx + seq_len
        
        # Get sequence data
        z_seq = self.latents[ep_idx][start_idx:end_idx]  # Input latents
        action_seq = self.actions[ep_idx][start_idx:end_idx]  # Input actions
        z_next_seq = self.latents[ep_idx][start_idx+1:end_idx+1]  # Target next latents
        
        sample = {
            'z': torch.FloatTensor(z_seq[:-1]),  # Input: z_0 to z_{T-1}
            'actions': torch.FloatTensor(action_seq[:-1]),  # Input: a_0 to a_{T-1}  
            'z_next': torch.FloatTensor(z_next_seq[:-1]),  # Target: z_1 to z_T
            'seq_len': seq_len - 1
        }
        
        # Add optional targets
        if self.rewards is not None:
            reward_seq = self.rewards[ep_idx][start_idx+1:end_idx+1]  # Rewards r_1 to r_T
            sample['rewards'] = torch.FloatTensor(reward_seq[:-1]).unsqueeze(-1)
            
        if self.dones is not None:
            done_seq = self.dones[ep_idx][start_idx+1:end_idx+1]  # Dones d_1 to d_T
            sample['dones'] = torch.FloatTensor(done_seq[:-1]).unsqueeze(-1)
        
        return sample


def mdn_loss_function(
    pi: torch.Tensor,
    mu: torch.Tensor, 
    sigma: torch.Tensor,
    target: torch.Tensor,
    reduce: bool = True
) -> torch.Tensor:
    """
    Compute negative log-likelihood loss for Mixture Density Network.
    
    Implements stable computation of -log(Σ π_k * N(target | μ_k, σ_k²))
    using log-sum-exp trick to prevent numerical underflow.
    
    Args:
        pi (torch.Tensor): Mixture weights (batch_size, seq_len, num_mixtures)
        mu (torch.Tensor): Mixture means (batch_size, seq_len, num_mixtures, z_dim)
        sigma (torch.Tensor): Mixture std devs (batch_size, seq_len, num_mixtures, z_dim)
        target (torch.Tensor): Target values (batch_size, seq_len, z_dim)
        reduce (bool): Whether to average over batch and sequence
        
    Returns:
        torch.Tensor: Negative log-likelihood loss
    """
    batch_size, seq_len, num_mixtures, z_dim = mu.shape
    
    # Expand target to match mixture dimensions
    target_expanded = target.unsqueeze(-2)  # (batch_size, seq_len, 1, z_dim)
    target_expanded = target_expanded.expand(-1, -1, num_mixtures, -1)
    
    # Compute squared Mahalanobis distance for each mixture component
    # For diagonal covariance: (x - μ)ᵀ Σ⁻¹ (x - μ) = Σ ((x_i - μ_i) / σ_i)²
    diff = target_expanded - mu  # (batch_size, seq_len, num_mixtures, z_dim)
    scaled_diff = diff / (sigma + 1e-8)  # Add epsilon for numerical stability
    mahalanobis_sq = torch.sum(scaled_diff ** 2, dim=-1)  # (batch_size, seq_len, num_mixtures)
    
    # Compute log probabilities for each mixture component
    # log N(x | μ, σ²) = -0.5 * (d*log(2π) + Σ log(σ²) + Σ ((x-μ)/σ)²)
    log_2pi = math.log(2 * math.pi)
    log_det = torch.sum(torch.log(sigma + 1e-8), dim=-1)  # Sum of log(σ) for diagonal cov
    log_probs = -0.5 * (z_dim * log_2pi + 2 * log_det + mahalanobis_sq)
    
    # Add mixture weights in log space: log(π_k) + log(N(x | μ_k, σ_k²))
    log_weighted_probs = torch.log(pi + 1e-8) + log_probs
    
    # Use log-sum-exp trick for numerical stability
    # log(Σ π_k * N(x | μ_k, σ_k²)) = log-sum-exp(log(π_k) + log(N(...)))
    max_log_prob = torch.max(log_weighted_probs, dim=-1, keepdim=True)[0]
    log_sum_exp = max_log_prob + torch.log(
        torch.sum(torch.exp(log_weighted_probs - max_log_prob), dim=-1, keepdim=True)
    )
    log_sum_exp = log_sum_exp.squeeze(-1)  # (batch_size, seq_len)
    
    # Negative log-likelihood loss
    nll_loss = -log_sum_exp
    
    if reduce:
        return torch.mean(nll_loss)
    else:
        return nll_loss


def sample_from_mdn(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor, 
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Sample from Mixture Density Network output.
    
    First samples which mixture component to use based on π weights,
    then samples from the chosen Gaussian component.
    
    Args:
        pi (torch.Tensor): Mixture weights (..., num_mixtures)
        mu (torch.Tensor): Mixture means (..., num_mixtures, z_dim)
        sigma (torch.Tensor): Mixture std devs (..., num_mixtures, z_dim)
        temperature (float): Temperature for sampling (1.0 = no scaling)
        
    Returns:
        torch.Tensor: Sampled values (..., z_dim)
    """
    # Apply temperature to mixture weights (higher temp = more uniform)
    if temperature != 1.0:
        pi = pi ** (1.0 / temperature)
        pi = pi / torch.sum(pi, dim=-1, keepdim=True)
    
    # Sample mixture component indices
    mixture_indices = torch.multinomial(pi.view(-1, pi.size(-1)), 1)
    mixture_indices = mixture_indices.view(pi.shape[:-1])  # Restore original shape
    
    # Gather means and sigmas for selected mixture components
    # Create indices for advanced indexing
    batch_indices = torch.arange(mu.size(0), device=mu.device)
    if len(mu.shape) == 4:  # (batch_size, seq_len, num_mixtures, z_dim)
        seq_indices = torch.arange(mu.size(1), device=mu.device)
        batch_grid, seq_grid = torch.meshgrid(batch_indices, seq_indices, indexing='ij')
        selected_mu = mu[batch_grid, seq_grid, mixture_indices]
        selected_sigma = sigma[batch_grid, seq_grid, mixture_indices] 
    else:  # (batch_size, num_mixtures, z_dim)
        selected_mu = mu[batch_indices, mixture_indices.squeeze(-1)]
        selected_sigma = sigma[batch_indices, mixture_indices.squeeze(-1)]
    
    # Sample from selected Gaussian components
    if temperature != 1.0:
        selected_sigma = selected_sigma * temperature
    
    epsilon = torch.randn_like(selected_mu)
    samples = selected_mu + selected_sigma * epsilon
    
    return samples


def train_mdnrnn(
    dataset: SequenceDataset,
    model: MDNRNN,
    config: Dict[str, Any]
) -> Dict[str, List[float]]:
    """
    Train MDN-RNN model on sequential data.
    
    Uses teacher forcing during training where the model receives ground truth
    previous states rather than its own predictions. This stabilizes training
    but may lead to exposure bias.
    
    Args:
        dataset (SequenceDataset): Training data containing sequences
        model (MDNRNN): Model to train
        config (Dict): Training configuration with hyperparameters
        
    Returns:
        Dict[str, List[float]]: Training history with losses per epoch
    """
    # Extract config with defaults
    cfg = {**DEFAULTS, **config}
    
    device = torch.device(cfg['device'])
    model = model.to(device)
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=0,  # Use 0 for Windows compatibility
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path(cfg['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training history
    history = {
        'total_loss': [],
        'mdn_loss': [],
        'reward_loss': [], 
        'done_loss': []
    }
    
    print("Starting MDN-RNN training...")
    print("Device: {}".format(device))
    print("Model parameters: {}".format(sum(p.numel() for p in model.parameters())))
    print("Batch size: {}, Epochs: {}".format(cfg['batch_size'], cfg['epochs']))
    
    for epoch in range(cfg['epochs']):
        model.train()
        epoch_losses = {'total': 0.0, 'mdn': 0.0, 'reward': 0.0, 'done': 0.0}
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            z = batch['z'].to(device)  # (batch_size, seq_len, z_dim)
            actions = batch['actions'].to(device)  # (batch_size, seq_len, action_dim)
            z_next = batch['z_next'].to(device)  # (batch_size, seq_len, z_dim)
            
            batch_size, seq_len = z.size(0), z.size(1)
            
            # Forward pass with teacher forcing
            outputs = model(z, actions)
            
            # =====================================================================
            # COMPUTE LOSSES
            # =====================================================================
            
            # MDN loss (negative log-likelihood)
            mdn_loss = mdn_loss_function(
                outputs['pi'], outputs['mu'], outputs['sigma'], z_next
            )
            
            total_loss = mdn_loss
            
            # Optional reward prediction loss
            reward_loss = torch.tensor(0.0, device=device)
            if model.predict_reward and 'rewards' in batch:
                target_rewards = batch['rewards'].to(device)
                pred_rewards = outputs['reward']
                reward_loss = F.mse_loss(pred_rewards, target_rewards)
                total_loss += cfg['reward_loss_weight'] * reward_loss
            
            # Optional done prediction loss  
            done_loss = torch.tensor(0.0, device=device)
            if model.predict_done and 'dones' in batch:
                target_dones = batch['dones'].to(device) 
                pred_dones = outputs['done']
                done_loss = F.binary_cross_entropy(pred_dones, target_dones)
                total_loss += cfg['done_loss_weight'] * done_loss
            
            # =====================================================================
            # BACKWARD PASS AND OPTIMIZATION
            # =====================================================================
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
            
            optimizer.step()
            
            # Track losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['mdn'] += mdn_loss.item()
            epoch_losses['reward'] += reward_loss.item()
            epoch_losses['done'] += done_loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 50 == 0:
                print("Epoch {}/{}, Batch {}/{}, Total Loss: {:.6f}, MDN Loss: {:.6f}".format(
                    epoch + 1, cfg['epochs'], batch_idx + 1, len(dataloader),
                    total_loss.item(), mdn_loss.item()
                ))
        
        # Average losses over epoch
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Update learning rate based on total loss
        scheduler.step(epoch_losses['total'])
        
        # Store history
        history['total_loss'].append(epoch_losses['total'])
        history['mdn_loss'].append(epoch_losses['mdn'])
        history['reward_loss'].append(epoch_losses['reward'])
        history['done_loss'].append(epoch_losses['done'])
        
        print("Epoch {}/{} - Total Loss: {:.6f}, MDN Loss: {:.6f}".format(
            epoch + 1, cfg['epochs'], epoch_losses['total'], epoch_losses['mdn']
        ))
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = checkpoint_dir / "mdnrnn_epoch_{}.pth".format(epoch + 1)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_losses['total'],
                'config': cfg
            }, checkpoint_path)
            print("Saved checkpoint: {}".format(checkpoint_path))
    
    # Save final model
    final_path = checkpoint_dir / "mdnrnn_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': cfg,
        'history': history
    }, final_path)
    print("Saved final model: {}".format(final_path))
    
    return history


def rollout_in_dream(
    controller,
    mdnrnn: MDNRNN,
    vae,
    initial_z: torch.Tensor,
    max_steps: int = 100,
    temperature: float = 1.0,
    device: Optional[torch.device] = None
) -> Dict[str, List]:
    """
    Perform rollout in the learned dream environment.
    
    Uses the MDN-RNN to simulate dynamics and VAE decoder to generate
    visual observations. The controller acts in this simulated environment.
    
    Args:
        controller: Policy that takes observations and returns actions
        mdnrnn (MDNRNN): Trained dynamics model
        vae: Trained VAE for decoding latents to images
        initial_z (torch.Tensor): Initial latent state (z_dim,)
        max_steps (int): Maximum rollout length
        temperature (float): Sampling temperature for MDN
        device (torch.device, optional): Device to run on
        
    Returns:
        Dict containing rollout data (frames, latents, actions, rewards, dones)
    """
    if device is None:
        device = mdnrnn.get_device()
    
    mdnrnn.eval()
    vae.eval()
    
    # Initialize rollout data
    frames = []
    latents = []
    actions_taken = []
    rewards = []
    dones = []
    
    # Initialize state
    current_z = initial_z.unsqueeze(0).to(device)  # (1, z_dim)
    hidden = mdnrnn.init_hidden(1, device)
    done = False
    
    with torch.no_grad():
        for step in range(max_steps):
            if done:
                break
            
            # Decode current latent to visual observation
            frame = vae.decode(current_z)  # (1, 3, 64, 64)
            frames.append(frame.cpu().numpy().squeeze())
            latents.append(current_z.cpu().numpy().squeeze())
            
            # Get action from controller (convert frame to numpy if needed)
            frame_np = frame.cpu().numpy().squeeze().transpose(1, 2, 0)  # (64, 64, 3)
            action = controller(frame_np)  # Controller should return action array
            
            if isinstance(action, np.ndarray):
                action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
            else:
                action_tensor = torch.tensor([action], dtype=torch.float32).unsqueeze(0).to(device)
            
            actions_taken.append(action_tensor.cpu().numpy().squeeze())
            
            # Predict next state using MDN-RNN
            z_input = current_z.unsqueeze(1)  # (1, 1, z_dim)
            action_input = action_tensor.unsqueeze(1)  # (1, 1, action_dim)
            
            outputs = mdnrnn(z_input, action_input, hidden)
            
            # Sample next latent state
            next_z = sample_from_mdn(
                outputs['pi'].squeeze(1),  # (1, num_mixtures)
                outputs['mu'].squeeze(1),  # (1, num_mixtures, z_dim)
                outputs['sigma'].squeeze(1),  # (1, num_mixtures, z_dim)
                temperature=temperature
            )
            
            # Update state
            current_z = next_z
            hidden = outputs['hidden']
            
            # Get predicted reward and done (if available)
            if mdnrnn.predict_reward:
                reward = outputs['reward'].squeeze().item()
                rewards.append(reward)
            else:
                rewards.append(0.0)  # Default reward
            
            if mdnrnn.predict_done:
                done_prob = outputs['done'].squeeze().item()
                done = done_prob > 0.5  # Threshold for termination
                dones.append(done)
            else:
                dones.append(False)  # Default no termination
    
    return {
        'frames': frames,
        'latents': latents, 
        'actions': actions_taken,
        'rewards': rewards,
        'dones': dones
    }


# Test and validation utilities
def validate_shapes_and_gradients(model: MDNRNN, batch_size: int = 4, seq_len: int = 10):
    """
    Validate that model produces correct output shapes and gradients flow properly.
    """
    print("Validating MDN-RNN shapes and gradients...")
    
    device = model.get_device()
    
    # Create dummy input
    z = torch.randn(batch_size, seq_len, model.z_dim, device=device)
    actions = torch.randn(batch_size, seq_len, model.action_dim, device=device)
    z_next = torch.randn(batch_size, seq_len, model.z_dim, device=device)
    
    # Forward pass
    outputs = model(z, actions)
    
    # Check output shapes
    assert outputs['pi'].shape == (batch_size, seq_len, model.num_mixtures)
    assert outputs['mu'].shape == (batch_size, seq_len, model.num_mixtures, model.z_dim)
    assert outputs['sigma'].shape == (batch_size, seq_len, model.num_mixtures, model.z_dim)
    
    # Check optional outputs
    if model.predict_reward:
        assert outputs['reward'].shape == (batch_size, seq_len, 1)
    if model.predict_done:
        assert outputs['done'].shape == (batch_size, seq_len, 1)
    
    # Test loss computation
    loss = mdn_loss_function(outputs['pi'], outputs['mu'], outputs['sigma'], z_next)
    
    # Check gradients
    loss.backward()
    
    # Verify gradients exist
    for name, param in model.named_parameters():
        assert param.grad is not None, "No gradient for {}".format(name)
    
    print("✅ All shape and gradient checks passed!")
    print("  Pi shape: {}".format(outputs['pi'].shape))
    print("  Mu shape: {}".format(outputs['mu'].shape))  
    print("  Sigma shape: {}".format(outputs['sigma'].shape))
    print("  Loss value: {:.6f}".format(loss.item()))


if __name__ == "__main__":
    # Quick test of MDN-RNN implementation
    print("Testing MDN-RNN implementation...")
    
    # Create model
    model = MDNRNN(
        z_dim=32,
        action_dim=3,
        rnn_size=256,
        num_mixtures=5,
        predict_reward=True,
        predict_done=True
    )
    
    print("Model created with {} parameters".format(sum(p.numel() for p in model.parameters())))
    
    # Validate shapes and gradients
    validate_shapes_and_gradients(model)
    
    # Test sampling
    batch_size, seq_len = 2, 5
    device = torch.device('cpu')
    model = model.to(device)
    
    z = torch.randn(batch_size, seq_len, 32, device=device)
    actions = torch.randn(batch_size, seq_len, 3, device=device)
    
    outputs = model(z, actions)
    samples = sample_from_mdn(
        outputs['pi'], outputs['mu'], outputs['sigma'], temperature=1.0
    )
    
    print("Sample shape: {}".format(samples.shape))
    print("✅ MDN-RNN implementation test completed successfully!")

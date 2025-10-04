
"""
Custom PPO Policy with Action Masking for Chess
Optimized for 2x T4 GPUs
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import CategoricalDistribution
from typing import Dict, List, Tuple, Type, Union, Optional, Any
import gym


class ChessFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for chess observations
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        # Embedding layers for board state
        self.piece_embedding = nn.Embedding(13, 64)
        self.position_embedding = nn.Embedding(64, 64)
        
        # Metadata embeddings
        self.turn_embedding = nn.Embedding(2, 32)
        self.castling_embedding = nn.Embedding(16, 32)
        self.en_passant_embedding = nn.Embedding(65, 32)
        
        # Projection to features
        total_dim = 64 * 64 + 32 + 32 + 32
        self.projection = nn.Sequential(
            nn.Linear(total_dim, features_dim * 2),
            nn.ReLU(),
            nn.Linear(features_dim * 2, features_dim),
            nn.LayerNorm(features_dim)
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: [batch, 67]
        board = observations[:, :64].long()
        metadata = observations[:, 64:67].long()
        
        # Embed board
        piece_emb = self.piece_embedding(board)  # [batch, 64, 64]
        pos_ids = torch.arange(64, device=observations.device).unsqueeze(0).expand(observations.shape[0], -1)
        pos_emb = self.position_embedding(pos_ids)  # [batch, 64, 64]
        
        board_features = (piece_emb + pos_emb).flatten(1)  # [batch, 64*64]
        
        # Embed metadata
        turn_emb = self.turn_embedding(metadata[:, 0])
        castling_emb = self.castling_embedding(metadata[:, 1])
        en_passant_emb = self.en_passant_embedding(metadata[:, 2])
        
        # Concatenate and project
        all_features = torch.cat([
            board_features, turn_emb, castling_emb, en_passant_emb
        ], dim=1)
        
        return self.projection(all_features)


class MaskedCategoricalDistribution(CategoricalDistribution):
    """
    Categorical distribution with action masking support
    """
    
    def __init__(self, action_dim: int):
        super().__init__(action_dim)
        self.action_mask = None
    
    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """Create the layers for the distribution"""
        return nn.Linear(latent_dim, self.action_dim)
    
    def proba_distribution(self, action_logits: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> "MaskedCategoricalDistribution":
        """Set parameters of the distribution"""
        if action_mask is not None:
            # Mask invalid actions with large negative value
            action_logits = torch.where(
                action_mask.bool(),
                action_logits,
                torch.tensor(-1e8, dtype=action_logits.dtype, device=action_logits.device)
            )
        
        self.distribution = torch.distributions.Categorical(logits=action_logits)
        return self
    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probability of actions"""
        return self.distribution.log_prob(actions)
    
    def entropy(self) -> torch.Tensor:
        """Get entropy"""
        return self.distribution.entropy()
    
    def sample(self) -> torch.Tensor:
        """Sample action"""
        return self.distribution.sample()
    
    def mode(self) -> torch.Tensor:
        """Get most probable action"""
        return torch.argmax(self.distribution.probs, dim=1)
    
    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """Sample or get deterministic actions"""
        if deterministic:
            return self.mode()
        return self.sample()


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """
    PPO Policy with action masking for illegal moves
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _build_mlp_extractor(self) -> None:
        """Build the MLP extractor"""
        super()._build_mlp_extractor()
    
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> MaskedCategoricalDistribution:
        """Get action distribution with masking capability"""
        action_logits = self.action_net(latent_pi)
        return MaskedCategoricalDistribution(self.action_space.n).proba_distribution(action_logits)
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False, action_masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with action masking
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Get action logits
        action_logits = self.action_net(latent_pi)
        
        # Create distribution with masking
        distribution = MaskedCategoricalDistribution(self.action_space.n).proba_distribution(
            action_logits, action_masks
        )
        
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        
        return actions, values, log_prob
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, action_masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions with masking
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Get action logits
        action_logits = self.action_net(latent_pi)
        
        # Create distribution with masking
        distribution = MaskedCategoricalDistribution(self.action_space.n).proba_distribution(
            action_logits, action_masks
        )
        
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        
        return values, log_prob, entropy
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict values"""
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

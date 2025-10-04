
"""
Custom PPO Policy with Action Masking for Chess
Optimized for 2x T4 GPUs
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, List, Tuple, Type, Union
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


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """
    PPO Policy with action masking for illegal moves
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_masks = None
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with action masking
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Get action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        # Apply action mask if available
        if self.action_masks is not None:
            # Mask illegal actions
            logits = distribution.distribution.logits
            masked_logits = logits + (self.action_masks - 1) * 1e9
            distribution.distribution.logits = masked_logits
        
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        
        return actions, values, log_prob
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions with masking
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        if self.action_masks is not None:
            logits = distribution.distribution.logits
            masked_logits = logits + (self.action_masks - 1) * 1e9
            distribution.distribution.logits = masked_logits
        
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        
        return values, log_prob, entropy

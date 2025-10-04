
"""
Action Masking Wrapper for Chess Environment
Properly handles illegal move masking for PPO
"""

import numpy as np
import gym
from typing import Dict, Any, Tuple


class ActionMasker(gym.Wrapper):
    """
    Wrapper to handle action masking for discrete action spaces.
    This is required for stable_baselines3 to properly mask illegal actions.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.current_mask = None
    
    def reset(self, **kwargs) -> np.ndarray:
        """Reset and get initial action mask"""
        obs = self.env.reset(**kwargs)
        self.current_mask = self.env.get_action_mask()
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step and update action mask"""
        obs, reward, done, info = self.env.step(action)
        
        # Get action mask from info
        if 'action_mask' in info:
            self.current_mask = info['action_mask']
        else:
            self.current_mask = self.env.get_action_mask()
        
        # Add mask to info for policy
        info['action_mask'] = self.current_mask
        
        return obs, reward, done, info
    
    def get_action_mask(self) -> np.ndarray:
        """Get current action mask"""
        if self.current_mask is None:
            self.current_mask = self.env.get_action_mask()
        return self.current_mask

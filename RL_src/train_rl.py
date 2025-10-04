
#!/usr/bin/env python3
"""
Reinforcement Learning Training Script for Chess Self-Play
Using PPO with stable_baselines3
Optimized for 2x NVIDIA T4 GPUs (16GB each)
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor

from RL_src.chess_env import ChessEnv
from RL_src.ppo_policy import ChessFeatureExtractor, MaskedActorCriticPolicy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rl_training.log')
    ]
)
logger = logging.getLogger(__name__)


def configure_t4_gpus():
    """
    Configure PyTorch for optimal performance on 2x T4 GPUs
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available!")
        return
    
    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} GPU(s)")
    
    if num_gpus >= 2:
        logger.info("Configuring for 2x T4 GPUs")
        
        # Set device placement strategy
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        
        # T4 specific optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Enable TF32 for T4 (Turing architecture supports it)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set optimal memory allocation
        torch.cuda.set_per_process_memory_fraction(0.95, device=0)
        torch.cuda.set_per_process_memory_fraction(0.95, device=1)
        
        # Memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        logger.info("T4 GPU optimizations applied")
        
        # Log GPU info
        for i in range(min(2, num_gpus)):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}, Memory: {props.total_memory / 1e9:.2f} GB")
    else:
        logger.warning(f"Expected 2 GPUs, found {num_gpus}")


def make_env(supervised_model_path: str, rank: int, use_supervised: bool = True):
    """
    Create environment factory
    """
    def _init():
        env = ChessEnv(
            supervised_model_path=supervised_model_path,
            use_supervised_policy=use_supervised
        )
        env = Monitor(env)
        return env
    return _init


class RLTrainer:
    """
    RL Trainer for Chess Self-Play with PPO
    """
    
    def __init__(self, config: Dict[str, Any], supervised_model_path: str):
        self.config = config
        self.supervised_model_path = supervised_model_path
        
        # Configure GPUs
        configure_t4_gpus()
        
        # Set device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create directories
        self.checkpoint_dir = Path("RL_checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.log_dir = Path("RL_logs")
        self.log_dir.mkdir(exist_ok=True)
    
    def create_environments(self, n_envs: int = 8):
        """
        Create vectorized environments
        For 2x T4, we can run multiple environments in parallel
        """
        logger.info(f"Creating {n_envs} parallel environments")
        
        # Use SubprocVecEnv for parallel processing
        env_fns = [
            make_env(self.supervised_model_path, i, use_supervised=True)
            for i in range(n_envs)
        ]
        
        return SubprocVecEnv(env_fns)
    
    def create_ppo_model(self, env):
        """
        Create PPO model with custom policy
        Optimized for T4 GPUs
        """
        rl_config = self.config.get('rl', {})
        
        # Policy kwargs
        policy_kwargs = dict(
            features_extractor_class=ChessFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=[dict(pi=[512, 512, 512], vf=[512, 512, 512])],
            activation_fn=nn.ReLU,
        )
        
        # PPO hyperparameters optimized for T4
        model = PPO(
            MaskedActorCriticPolicy,
            env,
            learning_rate=rl_config.get('learning_rate', 3e-4),
            n_steps=rl_config.get('n_steps', 2048),
            batch_size=rl_config.get('batch_size', 512),  # T4 can handle this
            n_epochs=rl_config.get('n_epochs', 10),
            gamma=rl_config.get('gamma', 0.99),
            gae_lambda=rl_config.get('gae_lambda', 0.95),
            clip_range=rl_config.get('clip_range', 0.2),
            clip_range_vf=rl_config.get('clip_range_vf', None),
            ent_coef=rl_config.get('ent_coef', 0.01),
            vf_coef=rl_config.get('vf_coef', 0.5),
            max_grad_norm=rl_config.get('max_grad_norm', 0.5),
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=rl_config.get('target_kl', 0.01),
            tensorboard_log=str(self.log_dir),
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=self.device,
        )
        
        logger.info("PPO model created")
        return model
    
    def train(self, total_timesteps: int = 10_000_000):
        """
        Train PPO agent
        """
        logger.info("=" * 80)
        logger.info("Starting RL Training with PPO")
        logger.info("=" * 80)
        
        # Create environments
        n_envs = self.config.get('rl', {}).get('n_parallel_envs', 8)
        env = self.create_environments(n_envs=n_envs)
        
        # Create model
        model = self.create_ppo_model(env)
        
        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.get('rl', {}).get('checkpoint_freq', 50000),
            save_path=str(self.checkpoint_dir),
            name_prefix='ppo_chess'
        )
        
        callbacks = CallbackList([checkpoint_callback])
        
        # Train
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                tb_log_name="PPO_Chess",
                reset_num_timesteps=True,
                progress_bar=True
            )
            
            # Save final model
            final_model_path = self.checkpoint_dir / 'ppo_chess_final.zip'
            model.save(str(final_model_path))
            logger.info(f"Final model saved to {final_model_path}")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            model.save(str(self.checkpoint_dir / 'ppo_chess_interrupted.zip'))
        
        finally:
            env.close()
        
        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Chess RL Training with PPO')
    parser.add_argument(
        '--config',
        type=str,
        default='settings.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--supervised-model',
        type=str,
        required=True,
        help='Path to supervised learning checkpoint (.pt file)'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=10_000_000,
        help='Total training timesteps'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    
    # Validate supervised model
    if not Path(args.supervised_model).exists():
        logger.error(f"Supervised model not found: {args.supervised_model}")
        sys.exit(1)
    
    logger.info(f"Using supervised model: {args.supervised_model}")
    
    # Create trainer
    trainer = RLTrainer(config, args.supervised_model)
    
    # Start training
    trainer.train(total_timesteps=args.timesteps)


if __name__ == '__main__':
    main()

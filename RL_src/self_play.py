
#!/usr/bin/env python3
"""
Self-Play Script for Chess RL Agent
Generate training data through self-play
"""

import argparse
import logging
from pathlib import Path
import chess
import numpy as np
from stable_baselines3 import PPO
import sys
sys.path.append('/kaggle/input/okjjjkml/RLL-main')
from RL_src.chess_env import ChessEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def self_play_game(model1: PPO, model2: PPO, env: ChessEnv, verbose: bool = False):
    """
    Play one game between two models
    """
    obs = env.reset()
    done = False
    move_count = 0
    game_pgn = []
    
    current_model = model1
    
    while not done:
        action, _states = current_model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        
        if verbose:
            print(f"Move {move_count + 1}")
            env.render()
            print()
        
        move_count += 1
        
        # Switch models (for self-play)
        current_model = model2 if current_model == model1 else model1
    
    result = None
    if info.get('checkmate'):
        result = "1-0" if env.board.turn == chess.BLACK else "0-1"
    elif info.get('draw') or info.get('stalemate'):
        result = "1/2-1/2"
    
    return result, move_count, info


def run_self_play(model_path: str, num_games: int = 100, verbose: bool = False):
    """
    Run multiple self-play games
    """
    logger.info(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    env = ChessEnv(supervised_model_path=None, use_supervised_policy=False)
    
    results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0, 'invalid': 0}
    total_moves = []
    
    for game_num in range(num_games):
        logger.info(f"Playing game {game_num + 1}/{num_games}")
        result, moves, info = self_play_game(model, model, env, verbose=verbose)
        
        if result:
            results[result] += 1
        else:
            results['invalid'] += 1
        
        total_moves.append(moves)
        
        if (game_num + 1) % 10 == 0:
            logger.info(f"Results after {game_num + 1} games:")
            logger.info(f"  Wins (White): {results['1-0']}")
            logger.info(f"  Wins (Black): {results['0-1']}")
            logger.info(f"  Draws: {results['1/2-1/2']}")
            logger.info(f"  Invalid: {results['invalid']}")
            logger.info(f"  Avg moves: {np.mean(total_moves):.1f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Final Results:")
    logger.info(f"  Total games: {num_games}")
    logger.info(f"  Wins (White): {results['1-0']} ({100*results['1-0']/num_games:.1f}%)")
    logger.info(f"  Wins (Black): {results['0-1']} ({100*results['0-1']/num_games:.1f}%)")
    logger.info(f"  Draws: {results['1/2-1/2']} ({100*results['1/2-1/2']/num_games:.1f}%)")
    logger.info(f"  Invalid: {results['invalid']} ({100*results['invalid']/num_games:.1f}%)")
    logger.info(f"  Average moves per game: {np.mean(total_moves):.1f}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Self-play for Chess RL')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained PPO model (.zip file)'
    )
    parser.add_argument(
        '--games',
        type=int,
        default=100,
        help='Number of self-play games'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print board after each move'
    )
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        return
    
    run_self_play(args.model, args.games, args.verbose)


if __name__ == '__main__':
    main()

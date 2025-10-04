
"""
Custom Chess Gym Environment for RL
Compatible with stable_baselines3
"""

import numpy as np
import chess
import torch
import gym
from gym import spaces
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ChessEnv(gym.Env):
    """
    Custom Chess Environment for Reinforcement Learning
    Compatible with stable_baselines3
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, supervised_model_path: Optional[str] = None, use_supervised_policy: bool = True):
        super(ChessEnv, self).__init__()
        
        self.board = chess.Board()
        self.use_supervised_policy = use_supervised_policy
        self.supervised_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load supervised model if provided
        if supervised_model_path and use_supervised_policy:
            self._load_supervised_model(supervised_model_path)
        
        # Action space: all possible moves (4672 moves in UCI format)
        self.action_space = spaces.Discrete(4672)
        
        # Observation space: board state + metadata
        # 64 squares (piece encoding) + 3 metadata (turn, castling, en passant)
        self.observation_space = spaces.Box(
            low=0, high=12, shape=(67,), dtype=np.int32
        )
        
        # Move mapping
        self.move_to_idx = {}
        self.idx_to_move = {}
        self._build_move_vocabulary()
        
        # Episode tracking
        self.move_count = 0
        self.max_moves = 300
        
    def _build_move_vocabulary(self):
        """Build vocabulary of all possible chess moves"""
        idx = 0
        for from_square in chess.SQUARES:
            for to_square in chess.SQUARES:
                if from_square != to_square:
                    move_uci = chess.square_name(from_square) + chess.square_name(to_square)
                    self.move_to_idx[move_uci] = idx
                    self.idx_to_move[idx] = move_uci
                    idx += 1
                    
                    if chess.square_rank(from_square) in [1, 6]:
                        for promotion in ['q', 'r', 'b', 'n']:
                            move_uci_promo = move_uci + promotion
                            self.move_to_idx[move_uci_promo] = idx
                            self.idx_to_move[idx] = move_uci_promo
                            idx += 1
    
    def _load_supervised_model(self, model_path: str):
        """Load supervised learning model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Reconstruct model architecture (same as training)
            from RL_src.supervised_model import ChessTransformer
            
            self.supervised_model = ChessTransformer(
                dim=768,
                depth=18,
                num_heads=12,
                dropout=0.0,  # No dropout for inference
                num_moves=4672,
                use_flash_attention=False,
                use_gradient_checkpointing=False
            ).to(self.device)
            
            self.supervised_model.load_state_dict(checkpoint['model_state_dict'])
            self.supervised_model.eval()
            logger.info(f"Loaded supervised model from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load supervised model: {e}")
            self.supervised_model = None
    
    def _board_to_observation(self) -> np.ndarray:
        """Convert board state to observation array"""
        obs = np.zeros(67, dtype=np.int32)
        
        # Encode pieces (0=empty, 1-6=white pieces, 7-12=black pieces)
        piece_map = {
            chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
            chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                piece_idx = piece_map[piece.piece_type]
                if piece.color == chess.BLACK:
                    piece_idx += 6
                obs[square] = piece_idx
        
        # Metadata
        obs[64] = 0 if self.board.turn == chess.WHITE else 1
        
        # Castling rights
        castling = 0
        if self.board.has_kingside_castling_rights(chess.WHITE):
            castling |= 1
        if self.board.has_queenside_castling_rights(chess.WHITE):
            castling |= 2
        if self.board.has_kingside_castling_rights(chess.BLACK):
            castling |= 4
        if self.board.has_queenside_castling_rights(chess.BLACK):
            castling |= 8
        obs[65] = castling
        
        # En passant
        obs[66] = self.board.ep_square if self.board.ep_square else 64
        
        return obs
    
    def _get_legal_moves_mask(self) -> np.ndarray:
        """Get mask of legal moves"""
        mask = np.zeros(4672, dtype=np.float32)
        for move in self.board.legal_moves:
            move_idx = self.move_to_idx.get(move.uci(), -1)
            if move_idx >= 0:
                mask[move_idx] = 1.0
        return mask
    
    def _get_supervised_opponent_move(self) -> Optional[chess.Move]:
        """Get move from supervised model"""
        if self.supervised_model is None:
            return None
        
        obs = self._board_to_observation()
        board_tensor = torch.from_numpy(obs[:64]).long().unsqueeze(0).to(self.device)
        metadata_tensor = torch.from_numpy(obs[64:67]).long().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, _ = self.supervised_model(board_tensor, metadata_tensor)
            
            # Mask illegal moves
            legal_mask = self._get_legal_moves_mask()
            legal_mask_tensor = torch.from_numpy(legal_mask).to(self.device)
            policy_logits = policy_logits.squeeze(0)
            policy_logits = policy_logits + (legal_mask_tensor - 1) * 1e9
            
            move_idx = policy_logits.argmax().item()
            move_uci = self.idx_to_move.get(move_idx)
            
            if move_uci:
                try:
                    return chess.Move.from_uci(move_uci)
                except:
                    pass
        
        return None
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.board = chess.Board()
        self.move_count = 0
        return self._board_to_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step"""
        # Convert action to move
        move_uci = self.idx_to_move.get(action)
        reward = 0.0
        done = False
        info = {}
        
        if not move_uci:
            # Invalid action index
            reward = -10.0
            done = True
            info['invalid_action'] = True
            return self._board_to_observation(), reward, done, info
        
        try:
            move = chess.Move.from_uci(move_uci)
            
            if move not in self.board.legal_moves:
                # Illegal move
                reward = -5.0
                done = True
                info['illegal_move'] = True
                return self._board_to_observation(), reward, done, info
            
            # Execute move
            self.board.push(move)
            self.move_count += 1
            
            # Check game end
            if self.board.is_checkmate():
                reward = 100.0  # Win
                done = True
                info['checkmate'] = True
                return self._board_to_observation(), reward, done, info
            
            if self.board.is_stalemate() or self.board.is_insufficient_material():
                reward = 0.0  # Draw
                done = True
                info['draw'] = True
                return self._board_to_observation(), reward, done, info
            
            if self.move_count >= self.max_moves:
                reward = 0.0
                done = True
                info['max_moves'] = True
                return self._board_to_observation(), reward, done, info
            
            # Opponent move (supervised model or random)
            if self.use_supervised_policy:
                opponent_move = self._get_supervised_opponent_move()
                if opponent_move and opponent_move in self.board.legal_moves:
                    self.board.push(opponent_move)
                else:
                    # Fallback to random
                    legal_moves = list(self.board.legal_moves)
                    if legal_moves:
                        self.board.push(np.random.choice(legal_moves))
            else:
                # Random opponent
                legal_moves = list(self.board.legal_moves)
                if legal_moves:
                    self.board.push(np.random.choice(legal_moves))
            
            self.move_count += 1
            
            # Check opponent win
            if self.board.is_checkmate():
                reward = -100.0  # Loss
                done = True
                info['opponent_checkmate'] = True
                return self._board_to_observation(), reward, done, info
            
            if self.board.is_stalemate() or self.board.is_insufficient_material():
                reward = 0.0
                done = True
                info['draw'] = True
                return self._board_to_observation(), reward, done, info
            
            # Small positive reward for valid move
            reward = 0.1
            
            # Bonus for check
            if self.board.is_check():
                reward += 1.0
            
        except Exception as e:
            logger.error(f"Error in step: {e}")
            reward = -10.0
            done = True
            info['error'] = str(e)
        
        return self._board_to_observation(), reward, done, info
    
    def render(self, mode='human'):
        """Render the board"""
        if mode == 'human':
            print(self.board)
        return str(self.board)

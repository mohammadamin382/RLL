
"""
Chess Transformer Model Architecture (Copy for RL)
Must match the supervised learning model exactly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, use_flash: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rope: Optional[RotaryPositionalEmbedding] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if rope is not None:
            q, k = rope.apply_rotary_pos_emb(q, k)
        
        if self.use_flash:
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            attn_output = attn @ v
        
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        hidden_dim = (hidden_dim + 255) // 256 * 256
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, use_flash: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout, use_flash)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, dropout=dropout)

    def forward(self, x: torch.Tensor, rope: Optional[RotaryPositionalEmbedding] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope, mask)
        x = x + self.mlp(self.norm2(x))
        return x


class ChessBoardEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.piece_embedding = nn.Embedding(13, embedding_dim)
        self.square_embedding = nn.Embedding(64, embedding_dim)
        self.turn_embedding = nn.Embedding(2, embedding_dim)
        self.castling_embedding = nn.Embedding(16, embedding_dim)
        self.en_passant_embedding = nn.Embedding(65, embedding_dim)
        self.norm = RMSNorm(embedding_dim)

    def forward(self, board_tensor: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        B = board_tensor.shape[0]
        
        piece_emb = self.piece_embedding(board_tensor)
        square_ids = torch.arange(64, device=board_tensor.device).unsqueeze(0).expand(B, -1)
        square_emb = self.square_embedding(square_ids)
        board_emb = piece_emb + square_emb
        
        turn_emb = self.turn_embedding(metadata[:, 0]).unsqueeze(1)
        castling_emb = self.castling_embedding(metadata[:, 1]).unsqueeze(1)
        en_passant_emb = self.en_passant_embedding(metadata[:, 2]).unsqueeze(1)
        
        cls_token = torch.zeros(B, 1, piece_emb.shape[-1], device=board_tensor.device)
        tokens = torch.cat([cls_token, board_emb, turn_emb, castling_emb, en_passant_emb], dim=1)
        
        return self.norm(tokens)


class ChessTransformer(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_moves: int = 4672,
        use_flash_attention: bool = True,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.dim = dim
        self.depth = depth
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        self.board_encoder = ChessBoardEncoder(dim)
        self.rope = RotaryPositionalEmbedding(dim // num_heads, max_seq_len=128)
        
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, dropout, use_flash_attention)
            for _ in range(depth)
        ])
        
        self.norm = RMSNorm(dim)
        
        self.policy_head = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, num_moves)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
            nn.Tanh()
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, board_tensor: torch.Tensor, metadata: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.board_encoder(board_tensor, metadata)
        
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, self.rope, mask, use_reentrant=False)
            else:
                x = layer(x, self.rope, mask)
        
        x = self.norm(x)
        cls_token = x[:, 0, :]
        
        policy_logits = self.policy_head(cls_token)
        value = self.value_head(cls_token)
        
        return policy_logits, value

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

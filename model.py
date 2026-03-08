import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ------------------------------------------------------------------------------------------------------------

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

# ------------------------------------------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)

# ------------------------------------------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))

# ------------------------------------------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len, device=self.inv_freq.device)
        angles = positions[:, None] * self.inv_freq[None, :]
        sin = angles.sin()
        cos = angles.cos()
        self.register_buffer("sin_cache", sin, persistent=False)
        self.register_buffer("cos_cache", cos, persistent=False)
    
    def forward(self, x: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[2]
        total_len = start_pos + seq_len
        
        if total_len > self.sin_cache.shape[0]:
            self._build_cache(total_len)
        
        sin = self.sin_cache[start_pos:total_len].to(x.dtype)
        cos = self.cos_cache[start_pos:total_len].to(x.dtype)
        return sin, cos

# Apply RoPE to Q and K only
def apply_rotary_pos_emb(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    
    sin = sin[None, None, :, :]
    cos = cos[None, None, :, :]
    
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos
    
    return torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)

# Expand KV heads to match Q heads
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, n_kv_heads, T, head_dim = x.shape
    return x[:, :, None, :, :].expand(B, n_kv_heads, n_rep, T, head_dim).reshape(B, n_kv_heads * n_rep, T, head_dim)

# ------------------------------------------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int, dropout: float):
        super().__init__()
        
        assert d_model % n_q_heads == 0
        assert n_q_heads % n_kv_heads == 0
        
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_q_heads // n_kv_heads
        self.head_dim = d_model // n_q_heads
        
        self.w_q = nn.Linear(d_model, n_q_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(n_q_heads * self.head_dim, d_model, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.dropout_p = dropout
    
    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        B, T, _ = x.shape
        start_pos = past_kv[0].shape[2] if past_kv is not None else 0
        
        # Project to Q, K, V with different head counts
        q = self.w_q(x).view(B, T, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # RoPE applied to Q and K only
        sin, cos = self.rotary_emb(k, start_pos)
        q = apply_rotary_pos_emb(q, sin, cos)
        k = apply_rotary_pos_emb(k, sin, cos)
        
        # Concatenate with cached KV if present
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        new_kv = (k, v) if use_cache else None
        
        # Repeat K/V heads across query head groups (GQA)
        k_expanded = repeat_kv(k, self.n_rep)
        v_expanded = repeat_kv(v, self.n_rep)
        
        dropout_p = self.dropout_p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal= True
        )
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.w_o(attn_out), new_kv

# ------------------------------------------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attention = GroupedQueryAttention(d_model, n_q_heads, n_kv_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        attn_out, new_kv = self.attention(self.attn_norm(x), past_kv=past_kv, use_cache=use_cache)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.feed_forward(self.ffn_norm(x)))
        return x, new_kv

# ------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_q_heads: int, n_kv_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_q_heads, n_kv_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        past_kv_list: Optional[list] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        
        new_kv_list = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            layer_past_kv = past_kv_list[i] if past_kv_list is not None else None
            x, new_kv = layer(x, past_kv=layer_past_kv, use_cache=use_cache)
            if use_cache:
                new_kv_list.append(new_kv)
        
        return self.norm(x), new_kv_list

# ------------------------------------------------------------------------------------------------------------

class FlashLLaMA(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_q_heads: int,
        n_kv_heads: int,
        d_ff: int,
        dropout: float
    ):
        super().__init__()
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.decoder = Decoder(d_model, n_layers, n_q_heads, n_kv_heads, d_ff, dropout)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying: output projection shares weights with input embedding
        self.lm_head.weight = self.embedding.embedding.weight
    
    def forward(
        self,
        x: torch.Tensor,
        past_kv_list: Optional[list] = None,
        use_cache: bool = False
    ):
        h = self.embedding(x)
        h, new_kv_list = self.decoder(h, past_kv_list=past_kv_list, use_cache=use_cache)
        logits = self.lm_head(h)
        
        if use_cache:
            return logits, new_kv_list
        return logits

# ------------------------------------------------------------------------------------------------------------

def build_llama(
    vocab_size: int,
    d_model: int = 1024,
    num_layers: int = 12,
    num_q_heads: int = 8,
    num_kv_heads: int = 4,
    d_ff: int = 2048,
    dropout: float = 0.1
) -> FlashLLaMA:
    
    return FlashLLaMA(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=num_layers,
        n_q_heads=num_q_heads,
        n_kv_heads=num_kv_heads,
        d_ff=d_ff,
        dropout=dropout
    )

# ------------------------------------------------------------------------------------------------------------
"""
NEXUS ENGINE — 1.87M-parameter transformer world model.

Architecture:
  vocab=4500, dim=128, heads=4, layers=5, ffn=256, max_seq=512
  Approximate parameter count: ~1,870,000

Input:  tokenized world-state sequence  (B, T)
Output: next-token logits               (B, T, vocab)
        world embedding                 (B, T, dim)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class NexusConfig:
    vocab_size: int = 4500
    dim: int = 128
    n_heads: int = 4
    n_layers: int = 5
    ffn_dim: int = 256
    max_seq_len: int = 512
    dropout: float = 0.1
    tie_weights: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: NexusConfig):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads

        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) / scale
        attn_scores = attn_scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Attention entropy used as confusion signal
        attn_entropy = -(attn_weights * (attn_weights + 1e-9).log()).sum(dim=-1).mean(dim=1)

        attn_weights = self.dropout(attn_weights)
        out = (attn_weights @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out), attn_entropy


class FeedForward(nn.Module):
    def __init__(self, cfg: NexusConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.dim, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: NexusConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_entropy = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, attn_entropy


class NexusEngine(nn.Module):
    """
    1.87M-param world model.
    Returns logits for next-token prediction AND a confusion_map dict
    that agents read to decide if/where to intervene.
    """

    def __init__(self, cfg: Optional[NexusConfig] = None):
        super().__init__()
        self.cfg = cfg or NexusConfig()

        self.token_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.dim)
        self.pos_emb = nn.Embedding(self.cfg.max_seq_len, self.cfg.dim)
        self.drop = nn.Dropout(self.cfg.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(self.cfg) for _ in range(self.cfg.n_layers)])

        self.ln_f = nn.LayerNorm(self.cfg.dim)
        self.head = nn.Linear(self.cfg.dim, self.cfg.vocab_size, bias=False)

        if self.cfg.tie_weights:
            self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"Sequence length {T} exceeds max {self.cfg.max_seq_len}"

        positions = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(positions))

        layer_entropies = []
        for block in self.blocks:
            x, entropy = block(x)
            layer_entropies.append(entropy)

        x = self.ln_f(x)
        world_embedding = x  # (B, T, dim) — exposed for HWSM

        logits = self.head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))

        # Build confusion map from layer attention entropies
        stacked = torch.stack(layer_entropies, dim=0)  # (n_layers, B, n_heads)
        confusion_map = {
            "mean_entropy": stacked.mean().item(),
            "layer_entropies": stacked.mean(dim=-1).mean(dim=-1).tolist(),
            "max_layer_entropy": stacked.max().item(),
            "logit_variance": logits.var(dim=-1).mean().item(),
            "top1_confidence": logits.softmax(dim=-1).max(dim=-1).values.mean().item(),
        }

        return logits, world_embedding, confusion_map, loss

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 64, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.max_seq_len else idx[:, -self.cfg.max_seq_len:]
            logits, _, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

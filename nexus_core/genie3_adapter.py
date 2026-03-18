"""
Genie3Adapter — bridges NEXUS ENGINE world embeddings into
Genie 3's latent action space format.

Genie 3 (DeepMind) accepts:
  - A video latent tensor z_v  : (B, T, latent_dim)
  - A latent action vector a   : (B, action_dim)

This adapter projects NEXUS world embeddings → Genie 3 format,
and translates Genie 3 outputs back to world-state tokens.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class Genie3Adapter(nn.Module):
    """
    Maps:
      nexus_embedding (B, T, nexus_dim=128)
        → genie_video_latent (B, T, genie_latent_dim=512)
        → genie_action      (B, genie_action_dim=256)

    And reverse:
      genie_video_latent (B, T, 512)
        → nexus_tokens  (B, T)  — argmax over projected vocab logits
    """

    def __init__(
        self,
        nexus_dim: int = 128,
        genie_latent_dim: int = 512,
        genie_action_dim: int = 256,
        nexus_vocab_size: int = 4500,
    ):
        super().__init__()
        # NEXUS → Genie 3 video latent
        self.to_video_latent = nn.Sequential(
            nn.Linear(nexus_dim, genie_latent_dim * 2),
            nn.GELU(),
            nn.Linear(genie_latent_dim * 2, genie_latent_dim),
        )

        # Temporal pooling → action vector
        self.to_action = nn.Sequential(
            nn.Linear(genie_latent_dim, genie_action_dim * 2),
            nn.GELU(),
            nn.Linear(genie_action_dim * 2, genie_action_dim),
            nn.Tanh(),  # normalize action to [-1, 1]
        )

        # Genie 3 → NEXUS token projection (for inverse mapping)
        self.from_video_latent = nn.Sequential(
            nn.Linear(genie_latent_dim, nexus_dim * 2),
            nn.GELU(),
            nn.Linear(nexus_dim * 2, nexus_vocab_size),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, nexus_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        nexus_embedding: (B, T, nexus_dim)
        returns:
          video_latent: (B, T, genie_latent_dim)
          action:       (B, genie_action_dim)
        """
        video_latent = self.to_video_latent(nexus_embedding)
        pooled = video_latent.mean(dim=1)  # temporal mean pool
        action = self.to_action(pooled)
        return video_latent, action

    def decode(self, video_latent: torch.Tensor) -> torch.Tensor:
        """
        video_latent: (B, T, genie_latent_dim)
        returns token logits: (B, T, nexus_vocab_size)
        """
        return self.from_video_latent(video_latent)

    def forward(
        self,
        nexus_embedding: torch.Tensor,
        return_tokens: bool = False
    ):
        video_latent, action = self.encode(nexus_embedding)
        if return_tokens:
            token_logits = self.decode(video_latent)
            return video_latent, action, token_logits
        return video_latent, action

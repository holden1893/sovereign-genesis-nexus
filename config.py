from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # NEXUS ENGINE
    nexus_vocab_size: int = 4500
    nexus_dim: int = 128
    nexus_heads: int = 4
    nexus_layers: int = 5
    nexus_ffn_dim: int = 256
    nexus_max_seq_len: int = 512
    nexus_dropout: float = 0.1

    # Training
    learning_rate: float = 3e-4
    batch_size: int = 32
    max_curriculum_stages: int = 5
    confusion_threshold: float = 0.65
    agent_dispatch_cooldown: float = 0.5   # seconds

    # Genie 3 Adapter
    genie3_latent_dim: int = 512
    genie3_action_dim: int = 256

    # Backend
    host: str = "0.0.0.0"
    port: int = 8000
    redis_url: str = "redis://localhost:6379/0"

    # Checkpoints
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    class Config:
        env_file = ".env"

settings = Settings()

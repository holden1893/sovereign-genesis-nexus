"""
SOVEREIGN PIPELINE — main closed-loop training and inference engine.

Usage (training):
    python -m pipeline.sovereign_pipeline --mode train --prompt "A dark gothic castle"

Usage (inference):
    python -m pipeline.sovereign_pipeline --mode run --prompt "A dark gothic castle"
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

from config import settings
from nexus_core.nexus_model import NexusEngine, NexusConfig
from nexus_core.world_memory import WorldMemory
from nexus_core.confusion_signals import ConfusionSignalDetector
from nexus_core.genie3_adapter import Genie3Adapter
from agents.studio_head import StudioHead
from agents.physics_agent import PhysicsAgent
from agents.visual_agent import VisualAgent
from agents.asset_agent import AssetAgent
from agents.scene_agent import SceneAgent
from agents.narrative_agent import NarrativeAgent
from pipeline.curriculum import CURRICULUM, CurriculumStage

console = Console()


def build_system() -> dict:
    """Instantiate and wire all components."""
    cfg = NexusConfig(
        vocab_size=settings.nexus_vocab_size,
        dim=settings.nexus_dim,
        n_heads=settings.nexus_heads,
        n_layers=settings.nexus_layers,
        ffn_dim=settings.nexus_ffn_dim,
        max_seq_len=settings.nexus_max_seq_len,
        dropout=settings.nexus_dropout,
    )

    nexus = NexusEngine(cfg)
    adapter = Genie3Adapter(
        nexus_dim=cfg.dim,
        genie_latent_dim=settings.genie3_latent_dim,
        genie_action_dim=settings.genie3_action_dim,
        nexus_vocab_size=cfg.vocab_size,
    )
    world_memory = WorldMemory()
    detector = ConfusionSignalDetector(threshold=settings.confusion_threshold)

    agents = {
        "physics":   PhysicsAgent(),
        "visual":    VisualAgent(),
        "asset":     AssetAgent(),
        "scene":     SceneAgent(),
        "narrative": NarrativeAgent(),
    }
    studio_head = StudioHead(world_memory, agents)

    param_count = nexus.count_params()
    logger.success(f"NEXUS ENGINE: {param_count:,} parameters ({param_count/1e6:.2f}M)")

    return {
        "nexus": nexus,
        "adapter": adapter,
        "world_memory": world_memory,
        "detector": detector,
        "studio_head": studio_head,
        "agents": agents,
    }


def prompt_to_tokens(prompt: str, vocab_size: int = 4500, max_len: int = 64) -> torch.Tensor:
    """
    Minimal tokenizer stub.
    In production, replace with a trained BPE/tiktoken tokenizer.
    """
    tokens = [hash(w) % (vocab_size - 1) + 1 for w in prompt.lower().split()]
    tokens = tokens[:max_len]
    if not tokens:
        tokens = [1]
    return torch.tensor([tokens], dtype=torch.long)


async def run_step(
    system: dict,
    tokens: torch.Tensor,
    stage: CurriculumStage,
    step: int,
) -> dict:
    """One pipeline step: forward pass → confusion analysis → agent dispatch → memory update."""
    nexus: NexusEngine = system["nexus"]
    adapter: Genie3Adapter = system["adapter"]
    world_memory: WorldMemory = system["world_memory"]
    detector: ConfusionSignalDetector = system["detector"]
    studio_head: StudioHead = system["studio_head"]

    # 1. Forward pass through NEXUS ENGINE
    logits, world_embedding, confusion_map, _ = nexus(tokens)

    # 2. Update working memory
    world_memory.working.current_tokens = tokens
    world_memory.working.current_embedding = world_embedding
    world_memory.working.confusion_map = confusion_map

    # 3. Genie 3 encode
    video_latent, action = adapter.encode(world_embedding)

    # 4. Analyze confusion
    report = detector.analyze(confusion_map)

    # 5. Filter report to only active agents for this curriculum stage
    report.priority_agents = [a for a in report.priority_agents if a in stage.active_agents]

    # 6. Dispatch agents via StudioHead
    agent_results = await studio_head.process(report)

    # 7. Snapshot world memory
    snap = world_memory.snapshot()
    world_memory.advance()

    return {
        "step": step,
        "stage": stage.name,
        "confusion_map": confusion_map,
        "report_dominant": report.dominant_category,
        "priority_agents": report.priority_agents,
        "agent_results": agent_results,
        "action_norm": action.norm().item(),
        "studio_state": studio_head.status()["state"],
    }


async def train_loop(system: dict, prompt: str, device: str = "cpu"):
    """5-stage curriculum training loop."""
    nexus: NexusEngine = system["nexus"]
    world_memory: WorldMemory = system["world_memory"]

    nexus.to(device)
    optimizer = optim.AdamW(nexus.parameters(), lr=settings.learning_rate)

    tokens = prompt_to_tokens(prompt, settings.nexus_vocab_size).to(device)

    log_rows = []
    Path(settings.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for stage in CURRICULUM:
        console.rule(f"[bold cyan]Stage {stage.index}: {stage.name}[/bold cyan]")
        world_memory.load_scene(f"stage_{stage.index}", stage.env_overrides)
        confusion_history = []
        global_step = 0

        with Live(console=console, refresh_per_second=4) as live:
            for step in range(stage.max_steps):
                optimizer.zero_grad()

                # Build targets (shift right for next-token prediction)
                if tokens.size(1) > 1:
                    tgt = tokens[:, 1:].to(device)
                    inp = tokens[:, :-1].to(device)
                else:
                    inp, tgt = tokens, tokens

                logits, world_embedding, confusion_map, loss = nexus(inp, tgt)

                if loss is not None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(nexus.parameters(), 1.0)
                    optimizer.step()

                step_data = await run_step(system, inp, stage, step)
                mean_entropy = confusion_map.get("mean_entropy", 0.0)
                confusion_history.append(mean_entropy)

                # Live display
                if step % 50 == 0 or step == stage.max_steps - 1:
                    tbl = Table(title=f"Stage {stage.index} | Step {step}/{stage.max_steps}")
                    tbl.add_column("Metric"); tbl.add_column("Value")
                    tbl.add_row("Loss", f"{loss.item():.4f}" if loss else "N/A")
                    tbl.add_row("Mean Entropy", f"{mean_entropy:.4f}")
                    tbl.add_row("Dominant Agent", step_data["report_dominant"])
                    tbl.add_row("Studio State", step_data["studio_state"])
                    tbl.add_row("Action Norm", f"{step_data['action_norm']:.4f}")
                    live.update(Panel(tbl))

                    log_rows.append({
                        "stage": stage.index, "step": step,
                        "loss": loss.item() if loss else None,
                        "mean_entropy": mean_entropy,
                        "dominant": step_data["report_dominant"],
                    })

                # Curriculum advancement check
                if len(confusion_history) >= 50:
                    avg = sum(confusion_history[-50:]) / 50
                    if avg < stage.confusion_threshold:
                        console.print(f"[green]✓ Stage {stage.index} complete! avg_entropy={avg:.4f}[/green]")
                        break

        # Save checkpoint
        ckpt_path = Path(settings.checkpoint_dir) / f"nexus_stage{stage.index}.pt"
        torch.save(nexus.state_dict(), ckpt_path)
        logger.success(f"Checkpoint saved: {ckpt_path}")

    # Save training log
    log_path = Path(settings.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    with open(log_path / "training_log.json", "w") as f:
        json.dump(log_rows, f, indent=2)

    console.print("[bold green]✓ SOVEREIGN GENESIS TRAINING COMPLETE[/bold green]")


async def inference_loop(system: dict, prompt: str, max_tokens: int = 128, device: str = "cpu"):
    """Single-shot inference: text prompt → generated world token sequence."""
    nexus: NexusEngine = system["nexus"]
    world_memory: WorldMemory = system["world_memory"]

    nexus.to(device)
    nexus.eval()

    tokens = prompt_to_tokens(prompt, settings.nexus_vocab_size).to(device)

    with torch.no_grad():
        generated = nexus.generate(tokens, max_new_tokens=max_tokens)
        _, world_embedding, confusion_map, _ = nexus(generated)
        video_latent, action = system["adapter"].encode(world_embedding)

    report = system["detector"].analyze(confusion_map)
    console.print(Panel(
        f"[bold]Prompt:[/bold] {prompt}\n"
        f"[bold]Generated tokens:[/bold] {generated.shape[1]}\n"
        f"[bold]Mean entropy:[/bold] {confusion_map['mean_entropy']:.4f}\n"
        f"[bold]Dominant confusion:[/bold] {report.dominant_category}\n"
        f"[bold]Action norm:[/bold] {action.norm().item():.4f}",
        title="[bold cyan]NEXUS GENESIS OUTPUT[/bold cyan]"
    ))
    return generated, video_latent, action


def main():
    parser = argparse.ArgumentParser(description="SOVEREIGN GENESIS ENGINE")
    parser.add_argument("--mode", choices=["train", "run"], default="run")
    parser.add_argument("--prompt", type=str, default="A dark gothic castle with torch-lit corridors")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_tokens", type=int, default=128)
    args = parser.parse_args()

    system = build_system()

    if args.mode == "train":
        asyncio.run(train_loop(system, args.prompt, args.device))
    else:
        asyncio.run(inference_loop(system, args.prompt, args.max_tokens, args.device))


if __name__ == "__main__":
    main()

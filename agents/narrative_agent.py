"""
NarrativeAgent — lore and story coherence.
Triggered by layer-4 (deepest) attention entropy.
"""
import asyncio
from typing import Any, Dict
from nexus_core.world_memory import WorldMemory
from nexus_core.confusion_signals import ConfusionReport
from loguru import logger


class NarrativeAgent:
    """
    Checks narrative_state in TaskMemory for inconsistencies.
    In production, this agent would call an LLM (e.g., GLM-4.7, Qwen)
    to re-generate contradictory lore fragments.
    """

    async def run(self, world_memory: WorldMemory, report: ConfusionReport) -> Dict[str, Any]:
        logger.info(f"NarrativeAgent running (score={report.narrative_score:.3f})")
        ns = world_memory.task.narrative_state
        fixes = []

        # Check for conflicting faction assignments
        factions = {k: v for k, v in ns.items() if "faction" in k}
        seen_factions = set(factions.values())
        if len(seen_factions) > 3 and report.narrative_score > 0.75:
            # Collapse to primary faction
            primary = list(seen_factions)[0]
            for k in factions:
                ns[k] = primary
            fixes.append({"fix": "faction_collapse", "to": primary})

        # Ensure at least a minimal narrative exists
        if "current_quest" not in ns:
            ns["current_quest"] = "explore_world"
            fixes.append({"fix": "default_quest_injected"})

        await asyncio.sleep(0)
        return {"fixes": fixes, "narrative_keys": len(ns)}

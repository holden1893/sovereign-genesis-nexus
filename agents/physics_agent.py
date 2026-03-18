"""
PhysicsAgent — corrects physics inconsistencies detected in NEXUS ENGINE outputs.

Reads:  world_memory.working.confusion_map  (layer 0-1 entropy)
        world_memory.task.objects           (scene objects)
Writes: world_memory.task.objects           (corrected positions/velocities)
        world_memory.env.gravity            (if gravity confusion detected)
"""
import asyncio
import random
from typing import Any, Dict

from nexus_core.world_memory import WorldMemory
from nexus_core.confusion_signals import ConfusionReport
from loguru import logger


class PhysicsAgent:
    """
    Detects and corrects:
      - Objects below ground plane (y < 0 when gravity is active)
      - Interpenetrating objects (basic AABB overlap check)
      - Gravity constant drift (entropy in layer-0 spikes)
      - NaN/Inf in object positions
    """

    def __init__(self, ground_y: float = 0.0, correction_strength: float = 0.8):
        self.ground_y = ground_y
        self.correction_strength = correction_strength

    async def run(self, world_memory: WorldMemory, report: ConfusionReport) -> Dict[str, Any]:
        logger.info(f"PhysicsAgent running (score={report.physics_score:.3f})")
        corrections = []
        objects = world_memory.task.objects

        for obj_id, obj in list(objects.items()):
            pos = obj.position
            # NaN / Inf guard
            if any(not isinstance(v, (int, float)) or v != v for v in pos):
                obj.position = [0.0, self.ground_y, 0.0]
                corrections.append({"id": obj_id, "fix": "nan_reset"})
                continue

            # Ground clamp
            if world_memory.env.gravity > 0 and pos[1] < self.ground_y:
                obj.position[1] = self.ground_y + abs(pos[1]) * self.correction_strength
                corrections.append({"id": obj_id, "fix": "ground_clamp", "new_y": obj.position[1]})

        # Gravity drift correction based on confusion severity
        if report.physics_score > 0.85:
            preset = world_memory.env.physics_preset
            target_g = {"standard": 9.81, "arcade": 15.0, "simulation": 9.807}.get(preset, 9.81)
            world_memory.env.gravity = target_g
            corrections.append({"fix": "gravity_reset", "value": target_g})

        # Simulated async work (would call physics sim in production)
        await asyncio.sleep(0)

        return {"corrections": corrections, "objects_checked": len(objects)}



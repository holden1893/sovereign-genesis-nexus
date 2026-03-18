"""
SceneAgent — spatial composition and camera work.
Triggered by high logit_variance or scene-level entropy.
"""
import asyncio
from typing import Any, Dict
from nexus_core.world_memory import WorldMemory
from nexus_core.confusion_signals import ConfusionReport
from loguru import logger


class SceneAgent:
    """
    Responsibilities:
      - Detect object clustering (all objects near origin)
      - Re-distribute objects in scene space
      - Suggest camera angles based on object density
    """

    SCENE_BOUNDS = (-50.0, 50.0)

    async def run(self, world_memory: WorldMemory, report: ConfusionReport) -> Dict[str, Any]:
        logger.info(f"SceneAgent running (score={report.scene_score:.3f})")
        actions = []

        objects = list(world_memory.task.objects.values())
        if not objects:
            return {"actions": [], "note": "empty_scene"}

        # Detect clustering: all objects within 1 unit of origin
        positions = [o.position for o in objects]
        spread = max(
            (abs(p[0]) + abs(p[2]) for p in positions), default=0
        )

        if spread < 1.0 and len(objects) > 1:
            # Re-distribute in a grid
            import math
            grid_size = math.ceil(math.sqrt(len(objects)))
            for i, obj in enumerate(objects):
                gx = (i % grid_size) * 4.0 - (grid_size * 2)
                gz = (i // grid_size) * 4.0 - (grid_size * 2)
                obj.position = [gx, obj.position[1], gz]
            actions.append({"fix": "scene_redistribution", "grid_size": grid_size})

        # Camera suggestion
        if report.scene_score > 0.70:
            centroid = [
                sum(p[0] for p in positions) / len(positions),
                0.0,
                sum(p[2] for p in positions) / len(positions),
            ]
            camera_pos = [centroid[0], 10.0, centroid[2] + 20.0]
            actions.append({"fix": "camera_reframe", "pos": camera_pos, "target": centroid})

        await asyncio.sleep(0)
        return {"actions": actions}

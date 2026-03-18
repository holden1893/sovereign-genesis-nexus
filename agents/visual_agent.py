"""
VisualAgent — detects and corrects visual artifacts in NEXUS outputs.

Works on: world_memory.env (lighting params)
          world_memory.task.objects (material/texture flags)
"""
import asyncio
from typing import Any, Dict
from nexus_core.world_memory import WorldMemory
from nexus_core.confusion_signals import ConfusionReport
from loguru import logger


class VisualAgent:
    LIGHT_PRESETS = {
        "day":   {"ambient": 0.8, "time_of_day": 12.0},
        "dusk":  {"ambient": 0.4, "time_of_day": 18.0},
        "night": {"ambient": 0.1, "time_of_day": 23.0},
    }

    async def run(self, world_memory: WorldMemory, report: ConfusionReport) -> Dict[str, Any]:
        logger.info(f"VisualAgent running (score={report.visual_score:.3f})")
        fixes = []

        # Ambient light out of range
        al = world_memory.env.ambient_light
        if not 0.0 <= al <= 1.0:
            world_memory.env.ambient_light = max(0.0, min(1.0, al))
            fixes.append({"fix": "ambient_clamp", "value": world_memory.env.ambient_light})

        # High confusion → normalize lighting to preset
        if report.visual_score > 0.80:
            tod = world_memory.env.time_of_day
            if 6 <= tod < 18:
                preset = "day"
            elif 18 <= tod < 21:
                preset = "dusk"
            else:
                preset = "night"
            p = self.LIGHT_PRESETS[preset]
            world_memory.env.ambient_light = p["ambient"]
            fixes.append({"fix": "lighting_preset", "preset": preset})

        # Flag texture re-bake on objects with visual confusion
        if report.visual_score > 0.7:
            for obj in world_memory.task.objects.values():
                if "needs_retexture" not in obj.tags:
                    obj.tags.append("needs_retexture")
            fixes.append({"fix": "texture_rebake_flagged", "count": len(world_memory.task.objects)})

        await asyncio.sleep(0)
        return {"fixes": fixes}

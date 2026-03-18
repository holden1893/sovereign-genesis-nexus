"""
HWSM — Hierarchical World State Memory
Three tiers:
  1. EnvironmentMemory  — global, persistent (physics constants, biome, lighting rules)
  2. TaskMemory         — scene-scoped (active objects, NPCs, scene graph)
  3. WorkingMemory      — frame-scoped (current model outputs, agent states)

The pipeline reads/writes all three tiers. Agents query specific tiers.
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import torch


# ──────────────────────────────────────────
# Tier 1: Environment Memory
# ──────────────────────────────────────────
@dataclass
class EnvironmentMemory:
    """Global world rules — set once, rarely changed."""
    biome: str = "generic"
    gravity: float = 9.81
    time_of_day: float = 12.0        # 0–24
    ambient_light: float = 0.5       # 0–1
    physics_preset: str = "standard" # standard | arcade | simulation
    world_seed: int = 42
    extra: Dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                self.extra[k] = v


# ──────────────────────────────────────────
# Tier 2: Task Memory
# ──────────────────────────────────────────
@dataclass
class SceneObject:
    id: str
    name: str
    position: List[float]            # [x, y, z]
    rotation: List[float]            # [rx, ry, rz]
    scale: List[float]               # [sx, sy, sz]
    physics_enabled: bool = True
    tags: List[str] = field(default_factory=list)
    embedding: Optional[torch.Tensor] = None  # NEXUS embedding slice


@dataclass
class TaskMemory:
    """Scene-scoped state — reset between scenes."""
    scene_id: str = "default"
    objects: Dict[str, SceneObject] = field(default_factory=dict)
    npcs: Dict[str, Dict] = field(default_factory=dict)
    narrative_state: Dict[str, Any] = field(default_factory=dict)
    portal_map: Dict[str, str] = field(default_factory=dict)  # scene transitions

    def add_object(self, obj: SceneObject):
        self.objects[obj.id] = obj

    def remove_object(self, obj_id: str):
        self.objects.pop(obj_id, None)

    def clear(self):
        self.objects.clear()
        self.npcs.clear()
        self.narrative_state.clear()
        self.portal_map.clear()


# ──────────────────────────────────────────
# Tier 3: Working Memory
# ──────────────────────────────────────────
@dataclass
class WorkingMemory:
    """Frame-scoped — holds live model state for one inference step."""
    step: int = 0
    current_tokens: Optional[torch.Tensor] = None
    current_embedding: Optional[torch.Tensor] = None   # (B, T, dim)
    confusion_map: Dict[str, Any] = field(default_factory=dict)
    active_agents: List[str] = field(default_factory=list)
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def tick(self):
        self.step += 1
        self.timestamp = time.time()
        self.agent_outputs.clear()
        self.active_agents.clear()


# ──────────────────────────────────────────
# WorldMemory — unified HWSM interface
# ──────────────────────────────────────────
class WorldMemory:
    """
    Top-level HWSM.  The pipeline and every agent interact with this object.

    Usage:
        mem = WorldMemory()
        mem.env.gravity = 0.0          # edit tier-1
        mem.task.add_object(obj)       # edit tier-2
        mem.working.confusion_map = {} # edit tier-3
        mem.snapshot()                 # push working → history
    """

    def __init__(self, history_len: int = 64):
        self.env = EnvironmentMemory()
        self.task = TaskMemory()
        self.working = WorkingMemory()
        self._history: deque = deque(maxlen=history_len)

    # ── Snapshot / Replay ────────────────
    def snapshot(self) -> Dict[str, Any]:
        snap = {
            "step": self.working.step,
            "confusion_map": dict(self.working.confusion_map),
            "active_agents": list(self.working.active_agents),
            "agent_outputs": dict(self.working.agent_outputs),
            "timestamp": self.working.timestamp,
        }
        self._history.append(snap)
        return snap

    def get_history(self, n: int = 8) -> List[Dict]:
        return list(self._history)[-n:]

    # ── Confusion Helpers ─────────────────
    def is_confused(self, threshold: float = 0.65) -> bool:
        cm = self.working.confusion_map
        return cm.get("mean_entropy", 0.0) > threshold or cm.get("logit_variance", 0.0) > threshold

    def top_confusion_layer(self) -> int:
        entropies = self.working.confusion_map.get("layer_entropies", [])
        if not entropies:
            return 0
        return int(max(range(len(entropies)), key=lambda i: entropies[i]))

    # ── Scene Management ──────────────────
    def load_scene(self, scene_id: str, env_overrides: Optional[Dict] = None):
        self.task = TaskMemory(scene_id=scene_id)
        if env_overrides:
            self.env.update(**env_overrides)

    def advance(self):
        self.working.tick()

    def state_dict(self) -> Dict:
        return {
            "env": self.env.__dict__,
            "task_scene_id": self.task.scene_id,
            "task_object_ids": list(self.task.objects.keys()),
            "working_step": self.working.step,
        }

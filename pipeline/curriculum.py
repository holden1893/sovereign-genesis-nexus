"""
CurriculumStage — defines the 5 progressive training stages.
Each stage specifies prompt complexity, which agents are active,
what confusion thresholds trigger advancement.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class CurriculumStage:
    index: int
    name: str
    description: str
    active_agents: List[str]
    confusion_threshold: float   # advance to next stage if avg confusion drops below this
    max_steps: int               # max training steps before forced advancement
    env_overrides: Dict[str, Any] = field(default_factory=dict)
    prompt_complexity: str = "simple"  # simple | moderate | complex | full


CURRICULUM = [
    CurriculumStage(
        index=1,
        name="Static Worlds",
        description="No physics. Single room. Objects placed at rest.",
        active_agents=["visual", "asset"],
        confusion_threshold=0.40,
        max_steps=500,
        env_overrides={"gravity": 0.0, "physics_preset": "standard"},
        prompt_complexity="simple",
    ),
    CurriculumStage(
        index=2,
        name="Basic Dynamics",
        description="Gravity active. Rigid body collisions. Simple lighting.",
        active_agents=["physics", "visual", "asset"],
        confusion_threshold=0.50,
        max_steps=1000,
        env_overrides={"gravity": 9.81},
        prompt_complexity="simple",
    ),
    CurriculumStage(
        index=3,
        name="Complex Physics",
        description="Soft bodies, particles, fluid surfaces.",
        active_agents=["physics", "visual", "asset", "scene"],
        confusion_threshold=0.55,
        max_steps=2000,
        env_overrides={"physics_preset": "simulation"},
        prompt_complexity="moderate",
    ),
    CurriculumStage(
        index=4,
        name="Lighting and Multi-Room",
        description="Dynamic lighting, shadows, portal transitions between scenes.",
        active_agents=["physics", "visual", "asset", "scene"],
        confusion_threshold=0.60,
        max_steps=3000,
        prompt_complexity="moderate",
    ),
    CurriculumStage(
        index=5,
        name="Full Coherence",
        description="Persistent world state, NPC behavior, narrative arcs, multi-minute consistency.",
        active_agents=["physics", "visual", "asset", "scene", "narrative"],
        confusion_threshold=0.65,
        max_steps=5000,
        prompt_complexity="complex",
    ),
]

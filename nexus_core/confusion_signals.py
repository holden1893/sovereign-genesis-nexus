"""
ConfusionSignalDetector — wraps raw confusion_map from NEXUS ENGINE
and classifies signals into typed categories that agents can act on.
"""
from dataclasses import dataclass, field
from typing import Dict, List
import torch


@dataclass
class ConfusionReport:
    raw: Dict
    physics_score: float = 0.0
    visual_score: float = 0.0
    narrative_score: float = 0.0
    asset_score: float = 0.0
    scene_score: float = 0.0
    dominant_category: str = "none"
    priority_agents: List[str] = field(default_factory=list)


class ConfusionSignalDetector:
    """
    Reads a confusion_map dict (from NexusEngine.forward) and produces
    a ConfusionReport that StudioHead uses to dispatch agents.

    Heuristics (tunable):
      - High layer-0..1 entropy → physics confusion (early-layer geometry)
      - High layer-2..3 entropy → visual/asset confusion
      - High layer-4 entropy    → narrative/scene confusion
      - High logit_variance     → general uncertainty → all agents on alert
    """

    AGENT_NAMES = ["physics", "visual", "asset", "scene", "narrative"]

    def __init__(
        self,
        threshold: float = 0.65,
        layer_to_agent: Dict[int, str] = None,
    ):
        self.threshold = threshold
        self.layer_to_agent = layer_to_agent or {
            0: "physics",
            1: "physics",
            2: "visual",
            3: "asset",
            4: "narrative",
        }

    def analyze(self, confusion_map: Dict) -> ConfusionReport:
        report = ConfusionReport(raw=confusion_map)
        entropies = confusion_map.get("layer_entropies", [0.0] * 5)
        logit_var = confusion_map.get("logit_variance", 0.0)
        confidence = confusion_map.get("top1_confidence", 1.0)

        scores = {name: 0.0 for name in self.AGENT_NAMES}

        for layer_idx, entropy in enumerate(entropies):
            agent = self.layer_to_agent.get(layer_idx, "visual")
            scores[agent] = max(scores[agent], entropy)

        # Low confidence → boost all scores
        if confidence < 0.3:
            for k in scores:
                scores[k] = min(1.0, scores[k] + 0.2)

        # High logit variance → scene agent
        scores["scene"] = max(scores["scene"], min(1.0, logit_var))

        report.physics_score   = scores["physics"]
        report.visual_score    = scores["visual"]
        report.asset_score     = scores["asset"]
        report.scene_score     = scores["scene"]
        report.narrative_score = scores["narrative"]

        priority_agents = [
            name for name, score in sorted(scores.items(), key=lambda x: -x[1])
            if score > self.threshold
        ]

        report.priority_agents = priority_agents
        report.dominant_category = priority_agents[0] if priority_agents else "none"
        return report

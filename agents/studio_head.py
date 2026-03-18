"""
StudioHead — HFSM-based decision engine.

States (Hierarchical Finite State Machine):
  IDLE → MONITORING → DISPATCHING → REVIEWING → IDLE

The StudioHead:
  1. Receives ConfusionReport every pipeline step
  2. Decides which specialist agents to wake
  3. Merges agent outputs back into WorldMemory
  4. Escalates if agents fail to reduce confusion
"""
import asyncio
import time
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any

from nexus_core.confusion_signals import ConfusionReport
from nexus_core.world_memory import WorldMemory
from loguru import logger


class HFSMState(Enum):
    IDLE        = auto()
    MONITORING  = auto()
    DISPATCHING = auto()
    REVIEWING   = auto()
    ESCALATED   = auto()


class StudioHead:
    """
    Orchestrates 5 specialist agents using a Hierarchical FSM.

    Agent priority (default, overridden by confusion report):
      1. physics    — must be physically plausible before visuals matter
      2. visual     — artifacts, texture, lighting
      3. asset      — mesh, audio, animation quality
      4. scene      — spatial composition, cameras
      5. narrative  — lore, story coherence

    Escalation: if confusion does not drop after max_retries, StudioHead
    triggers a scene reset (writes to world_memory.task) and logs.
    """

    AGENT_PRIORITY = ["physics", "visual", "asset", "scene", "narrative"]

    def __init__(
        self,
        world_memory: WorldMemory,
        agents: Dict[str, Any],
        max_retries: int = 3,
        dispatch_timeout: float = 5.0,
    ):
        self.world_memory = world_memory
        self.agents = agents
        self.max_retries = max_retries
        self.dispatch_timeout = dispatch_timeout

        self.state = HFSMState.IDLE
        self._retry_count = 0
        self._last_confusion = 1.0
        self._dispatch_history: List[Dict] = []

    # ── State Transitions ────────────────────
    def _transition(self, new_state: HFSMState):
        logger.debug(f"StudioHead: {self.state.name} → {new_state.name}")
        self.state = new_state

    # ── Main Entry Point ─────────────────────
    async def process(self, report: ConfusionReport) -> Dict[str, Any]:
        """
        Called every pipeline step with a fresh ConfusionReport.
        Returns merged agent outputs.
        """
        self._transition(HFSMState.MONITORING)

        if not report.priority_agents:
            logger.info("StudioHead: no confusion detected, staying idle.")
            self._transition(HFSMState.IDLE)
            self._retry_count = 0
            return {}

        # Order agents by report priority, then fall back to default priority
        ordered = self._order_agents(report.priority_agents)

        self._transition(HFSMState.DISPATCHING)
        results = await self._dispatch(ordered, report)

        self._transition(HFSMState.REVIEWING)
        merged = self._review(results, report)

        if merged.get("confusion_reduced", True):
            self._retry_count = 0
            self._transition(HFSMState.IDLE)
        else:
            self._retry_count += 1
            if self._retry_count >= self.max_retries:
                await self._escalate(report)
            else:
                logger.warning(f"StudioHead: confusion not reduced (retry {self._retry_count}/{self.max_retries})")
                self._transition(HFSMState.IDLE)

        self._dispatch_history.append({
            "step": self.world_memory.working.step,
            "agents_dispatched": ordered,
            "results_keys": list(results.keys()),
            "timestamp": time.time(),
        })
        return merged

    # ── Agent Ordering ───────────────────────
    def _order_agents(self, priority_list: List[str]) -> List[str]:
        seen = set()
        ordered = []
        for a in priority_list:
            if a not in seen and a in self.agents:
                ordered.append(a)
                seen.add(a)
        for a in self.AGENT_PRIORITY:
            if a not in seen and a in self.agents:
                ordered.append(a)
                seen.add(a)
        return ordered

    # ── Dispatch ─────────────────────────────
    async def _dispatch(self, agent_names: List[str], report: ConfusionReport) -> Dict:
        tasks = {}
        for name in agent_names:
            agent = self.agents.get(name)
            if agent and hasattr(agent, "run"):
                tasks[name] = asyncio.create_task(
                    asyncio.wait_for(
                        agent.run(self.world_memory, report),
                        timeout=self.dispatch_timeout
                    )
                )
        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
                logger.success(f"  ✓ {name.capitalize()}Agent completed")
            except asyncio.TimeoutError:
                logger.error(f"  ✗ {name.capitalize()}Agent timed out")
                results[name] = {"error": "timeout"}
            except Exception as e:
                logger.error(f"  ✗ {name.capitalize()}Agent error: {e}")
                results[name] = {"error": str(e)}
        return results

    # ── Review ───────────────────────────────
    def _review(self, results: Dict, report: ConfusionReport) -> Dict:
        merged = {"agent_results": results}
        current_entropy = report.raw.get("mean_entropy", self._last_confusion)
        merged["confusion_reduced"] = current_entropy < self._last_confusion
        merged["entropy_delta"] = self._last_confusion - current_entropy
        self._last_confusion = current_entropy

        # Write agent outputs to world memory
        self.world_memory.working.agent_outputs.update(results)
        self.world_memory.working.active_agents = list(results.keys())
        return merged

    # ── Escalation ───────────────────────────
    async def _escalate(self, report: ConfusionReport):
        logger.critical(
            f"StudioHead ESCALATED after {self._retry_count} retries. "
            f"dominant_category={report.dominant_category}. Requesting scene reset."
        )
        self._transition(HFSMState.ESCALATED)
        self.world_memory.task.clear()
        self._retry_count = 0
        self._transition(HFSMState.IDLE)

    def status(self) -> Dict:
        return {
            "state": self.state.name,
            "retry_count": self._retry_count,
            "last_confusion": self._last_confusion,
            "dispatch_history_len": len(self._dispatch_history),
        }

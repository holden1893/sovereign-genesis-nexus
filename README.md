# SOVEREIGN GENESIS ENGINE

A closed-loop AI game studio fusing a 1.87M-param NEXUS ENGINE world model
with a Genie 3 adapter and 5 specialist AI studio agents.

## Architecture

```
Text Prompt
    │
    ▼
┌─────────────────────────────────────┐
│         SOVEREIGN PIPELINE          │  sovereign_pipeline.py
│  Stage 1 → 2 → 3 → 4 → 5           │
│  (Static → Physics → Full World)    │
└──────────────┬──────────────────────┘
               │  confusion signals
               ▼
┌─────────────────────────────────────┐
│          STUDIO HEAD                │  agents/studio_head.py
│   HFSM Decision Engine              │
│   Priority-based agent dispatch     │
└──┬───────┬────────┬────────┬────────┘
   │       │        │        │
   ▼       ▼        ▼        ▼
Physics  Visual  Asset   Scene/Narrative
Agent    Agent   Agent   Agent
               │
               ▼
┌─────────────────────────────────────┐
│          NEXUS ENGINE               │  nexus_core/nexus_model.py
│   1.87M-param Transformer World     │
│   Model (vocab=4500, dim=128,       │
│   heads=4, layers=5, ffn=256)       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│       WORLD MEMORY (HWSM)           │  nexus_core/world_memory.py
│  ├── Environment Memory (global)    │
│  ├── Task Memory (scene-level)      │
│  └── Working Memory (frame-level)   │
└─────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│        GENIE 3 ADAPTER              │  nexus_core/genie3_adapter.py
│  Bridges NEXUS outputs to           │
│  Genie 3 latent action space        │
└─────────────────────────────────────┘
```

## Curriculum Stages

| Stage | Name              | Focus                          |
|-------|-------------------|--------------------------------|
| 1     | Static Worlds     | No physics, single room        |
| 2     | Basic Dynamics    | Gravity, rigid body collision  |
| 3     | Complex Physics   | Soft bodies, particles, fluid  |
| 4     | Lighting & Multi-Room | Dynamic lighting, portals  |
| 5     | Full Coherence    | Persistent state, narrative    |

## Quickstart

```bash
pip install -r requirements.txt
python -m pipeline.sovereign_pipeline --prompt "A dark gothic castle with torch-lit corridors"
```

## File Map

| File                              | Role                                  |
|-----------------------------------|---------------------------------------|
| nexus_core/nexus_model.py         | 1.87M transformer world model         |
| nexus_core/world_memory.py        | HWSM 3-tier memory hierarchy          |
| nexus_core/confusion_signals.py   | Uncertainty/confusion metrics         |
| nexus_core/genie3_adapter.py      | Genie 3 latent space bridge           |
| pipeline/sovereign_pipeline.py    | 5-stage curriculum training loop      |
| pipeline/curriculum.py            | Stage definitions and progression     |
| agents/studio_head.py             | HFSM decision engine + dispatch       |
| agents/physics_agent.py           | Physics inconsistency correction      |
| agents/visual_agent.py            | Visual artifact detection and fix     |
| agents/asset_agent.py             | 3D asset and audio management         |
| agents/scene_agent.py             | Spatial composition + camera          |
| agents/narrative_agent.py         | Story/lore consistency                |
| backend/main.py                   | FastAPI backend + WebSocket server    |
| frontend/src/app/page.tsx         | Next.js UI with tldraw canvas         |

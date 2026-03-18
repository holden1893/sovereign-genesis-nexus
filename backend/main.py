"""
SOVEREIGN GENESIS — FastAPI Backend
Connects the frontend (Next.js) to the pipeline via REST + WebSocket.
"""
import asyncio
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from loguru import logger

from config import settings
from pipeline.sovereign_pipeline import build_system, run_step, prompt_to_tokens
from pipeline.curriculum import CURRICULUM

app = FastAPI(title="SOVEREIGN GENESIS ENGINE", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system state
SYSTEM = None
ACTIVE_ROOMS: Dict[str, List[WebSocket]] = {}


def get_system():
    global SYSTEM
    if SYSTEM is None:
        SYSTEM = build_system()
    return SYSTEM


# ── WebSocket Hub ─────────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.rooms: Dict[str, List[WebSocket]] = {}

    async def connect(self, room_id: str, ws: WebSocket):
        await ws.accept()
        self.rooms.setdefault(room_id, []).append(ws)
        logger.info(f"WS connect: room={room_id}")

    def disconnect(self, room_id: str, ws: WebSocket):
        if room_id in self.rooms:
            self.rooms[room_id].remove(ws)

    async def broadcast(self, room_id: str, message: str):
        for ws in self.rooms.get(room_id, []):
            try:
                await ws.send_text(message)
            except Exception:
                pass


manager = ConnectionManager()


@app.websocket("/ws/{room_id}")
async def websocket_endpoint(ws: WebSocket, room_id: str):
    await manager.connect(room_id, ws)
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(room_id, ws)


# ── REST Endpoints ────────────────────────────────────────────────────────────
class CreateRequest(BaseModel):
    intent: str
    stage_index: Optional[int] = 1
    max_tokens: Optional[int] = 64


class StepRequest(BaseModel):
    room_id: str
    prompt: str
    stage_index: Optional[int] = 1


@app.post("/create")
async def create_world(req: CreateRequest):
    system = get_system()
    nexus = system["nexus"]
    nexus.eval()

    tokens = prompt_to_tokens(req.intent, settings.nexus_vocab_size)
    stage = CURRICULUM[min(req.stage_index - 1, len(CURRICULUM) - 1)]

    with torch.no_grad():
        generated = nexus.generate(tokens, max_new_tokens=req.max_tokens)

    _, world_embedding, confusion_map, _ = nexus(generated)
    _, action = system["adapter"].encode(world_embedding)
    report = system["detector"].analyze(confusion_map)

    return {
        "status": "created",
        "prompt": req.intent,
        "stage": stage.name,
        "generated_length": generated.shape[1],
        "confusion": confusion_map,
        "dominant_agent": report.dominant_category,
        "action_norm": round(action.norm().item(), 4),
        "world_state": system["world_memory"].state_dict(),
    }


@app.post("/step")
async def pipeline_step(req: StepRequest):
    system = get_system()
    stage = CURRICULUM[0]
    tokens = prompt_to_tokens(req.prompt, settings.nexus_vocab_size)
    result = await run_step(system, tokens, stage, step=system["world_memory"].working.step)
    await manager.broadcast(req.room_id, json.dumps({"type": "step", "data": result}))
    return result


@app.get("/status")
async def status():
    system = get_system()
    nexus = system["nexus"]
    return {
        "nexus_params": nexus.count_params(),
        "world_memory": system["world_memory"].state_dict(),
        "studio_head": system["studio_head"].status(),
        "curriculum_stages": len(CURRICULUM),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "SOVEREIGN GENESIS ENGINE v1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=True)

import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional

app = FastAPI(title="SOVEREIGN GENESIS ENGINE", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ConnectionManager:
    def __init__(self):
        self.rooms = {}
    async def connect(self, room_id, ws):
        await ws.accept()
        self.rooms.setdefault(room_id, []).append(ws)
    def disconnect(self, room_id, ws):
        if room_id in self.rooms:
            self.rooms[room_id].remove(ws)
    async def broadcast(self, room_id, message):
        for ws in self.rooms.get(room_id, []):
            try:
                await ws.send_text(message)
            except: pass

manager = ConnectionManager()

def generate_world(intent, stage=1):
    rng = np.random.default_rng(hash(intent) % (2**32))
    confusion = float(rng.uniform(0.2, 0.8))
    agents = ["physics","visual","asset","scene","narrative"]
    return {
        "status": "created", "prompt": intent, "stage": f"stage_{stage}",
        "generated_length": int(rng.integers(32,128)),
        "confusion": confusion,
        "dominant_agent": agents[int(confusion*len(agents))],
        "action_norm": round(float(np.linalg.norm(rng.standard_normal(256))),4),
        "world_state": {
            "biome": "procedural", "gravity": 9.81,
            "time_of_day": float(rng.uniform(0,24)),
            "ambient_light": float(rng.uniform(0.3,0.9)),
            "world_seed": int(rng.integers(0,99999))
        }
    }

class CreateRequest(BaseModel):
    intent: str
    stage_index: int = 1
    max_tokens: int = 64

class StepRequest(BaseModel):
    room_id: str
    prompt: str
    stage_index: int = 1

@app.websocket("/ws/{room_id}")
async def ws_endpoint(ws: WebSocket, room_id: str):
    await manager.connect(room_id, ws)
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type":"pong"}))
    except WebSocketDisconnect:
        manager.disconnect(room_id, ws)

@app.post("/create")
async def create_world(req: CreateRequest):
    return generate_world(req.intent, req.stage_index)

@app.post("/step")
async def step(req: StepRequest):
    result = generate_world(req.prompt, req.stage_index)
    await manager.broadcast(req.room_id, json.dumps({"type":"step","data":result}))
    return result

@app.get("/status")
async def status():
    return {"nexus_params":1870000,"mode":"lite","curriculum_stages":5,"online":True}

@app.get("/health")
async def health():
    return {"status":"ok","engine":"SOVEREIGN GENESIS ENGINE v1.0"}

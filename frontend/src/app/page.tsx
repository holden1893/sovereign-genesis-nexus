'use client';

import { Tldraw } from '@tldraw/tldraw';
import '@tldraw/tldraw/tldraw.css';
import { useEffect, useState, useRef, useCallback } from 'react';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';
const WS  = process.env.NEXT_PUBLIC_WS_URL  ?? 'ws://localhost:8000';

interface ConfusionMap {
  mean_entropy: number;
  logit_variance: number;
  top1_confidence: number;
  layer_entropies: number[];
}

interface WorldState {
  stage: string;
  dominant_agent: string;
  action_norm: number;
  confusion: ConfusionMap;
  generated_length: number;
}

const STAGE_COLORS: Record<number, string> = {
  1: '#3b82f6', 2: '#8b5cf6', 3: '#f59e0b', 4: '#10b981', 5: '#ef4444',
};

export default function SovereignGenesis() {
  const [logs, setLogs]         = useState<string[]>([]);
  const [intent, setIntent]     = useState('');
  const [loading, setLoading]   = useState(false);
  const [stage, setStage]       = useState(1);
  const [worldState, setWorldState] = useState<WorldState | null>(null);
  const [connected, setConnected]   = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const roomId    = useRef(`room-${Math.random().toString(36).slice(2, 8)}`);
  const logEndRef = useRef<HTMLDivElement>(null);

  const addLog = useCallback((msg: string) => {
    setLogs(prev => [...prev.slice(-199), `[${new Date().toLocaleTimeString()}] ${msg}`]);
  }, []);

  useEffect(() => {
    const ws = new WebSocket(`${WS}/ws/${roomId.current}`);
    socketRef.current = ws;

    ws.onopen    = () => { setConnected(true);  addLog('⚡ WebSocket connected'); };
    ws.onclose   = () => { setConnected(false); addLog('⚠ WebSocket disconnected'); };
    ws.onerror   = () =>   addLog('✗ WebSocket error');
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === 'step') {
          addLog(`STEP ${msg.data.step} | entropy=${msg.data.confusion_map?.mean_entropy?.toFixed(3)} | agent=${msg.data.report_dominant}`);
        }
      } catch {}
    };

    return () => ws.close();
  }, [addLog]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const spawnWorld = async () => {
    if (!intent.trim() || loading) return;
    setLoading(true);
    addLog(`🌍 Spawning world: "${intent}" (Stage ${stage})`);
    try {
      const res = await fetch(`${API}/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ intent, stage_index: stage }),
      });
      const data: WorldState = await res.json();
      setWorldState(data);
      addLog(`✓ World created | stage=${data.stage} | tokens=${data.generated_length} | dominant=${data.dominant_agent}`);
      addLog(`  entropy=${data.confusion?.mean_entropy?.toFixed(4)} | confidence=${data.confusion?.top1_confidence?.toFixed(4)}`);
    } catch (e) {
      addLog(`✗ Error: ${e}`);
    } finally {
      setLoading(false);
    }
  };

  const stageColor = STAGE_COLORS[stage] ?? '#6b7280';

  return (
    <main className="h-screen flex flex-col bg-[#0a0a0f] text-white font-mono overflow-hidden">

      {/* ── Header ── */}
      <div className="px-6 py-4 border-b border-gray-800 flex items-center gap-4 flex-shrink-0">
        <div>
          <h1 className="text-2xl font-bold tracking-tight" style={{ color: stageColor }}>
            SOVEREIGN GENESIS ENGINE
          </h1>
          <p className="text-xs text-gray-500">NEXUS ENGINE 1.87M · Genie 3 Adapter · 5 Studio Agents</p>
        </div>
        <div className="ml-auto flex items-center gap-3">
          <span className={`text-xs px-2 py-1 rounded ${connected ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
            {connected ? '● LIVE' : '○ OFFLINE'}
          </span>
        </div>
      </div>

      {/* ── Stage Selector + Prompt ── */}
      <div className="px-6 py-3 border-b border-gray-800 flex-shrink-0 space-y-3">
        <div className="flex gap-2">
          {[1,2,3,4,5].map(s => (
            <button
              key={s}
              onClick={() => setStage(s)}
              className="px-3 py-1 rounded text-xs font-bold transition-all"
              style={{
                background: stage === s ? STAGE_COLORS[s] : '#1f2937',
                color: stage === s ? '#fff' : '#9ca3af',
                outline: stage === s ? `2px solid ${STAGE_COLORS[s]}` : 'none',
              }}
            >
              Stage {s}
            </button>
          ))}
          <span className="text-xs text-gray-500 ml-2 self-center">
            {['Static Worlds','Basic Dynamics','Complex Physics','Lighting & Multi-Room','Full Coherence'][stage-1]}
          </span>
        </div>

        <div className="flex gap-2">
          <input
            className="flex-1 p-3 rounded bg-gray-900 text-white text-sm border border-gray-700 focus:border-blue-500 focus:outline-none"
            placeholder="Describe your world. Press Enter or click GENERATE."
            value={intent}
            onChange={e => setIntent(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && spawnWorld()}
          />
          <button
            onClick={spawnWorld}
            disabled={loading || !intent.trim()}
            className="px-6 py-3 rounded font-bold text-sm transition-all disabled:opacity-40"
            style={{ background: loading ? '#374151' : stageColor }}
          >
            {loading ? '⟳ GENERATING...' : '⚡ GENERATE'}
          </button>
        </div>
      </div>

      {/* ── Main Content ── */}
      <div className="flex flex-1 overflow-hidden">

        {/* Left: Logs + World State */}
        <div className="w-80 flex flex-col border-r border-gray-800 flex-shrink-0">

          {/* World State Card */}
          {worldState && (
            <div className="p-4 border-b border-gray-800 space-y-2">
              <p className="text-xs text-gray-400 uppercase tracking-widest">Last World State</p>
              <div className="grid grid-cols-2 gap-1 text-xs">
                <span className="text-gray-500">Stage</span>
                <span style={{ color: stageColor }}>{worldState.stage}</span>
                <span className="text-gray-500">Tokens</span>
                <span className="text-white">{worldState.generated_length}</span>
                <span className="text-gray-500">Entropy</span>
                <span className={worldState.confusion?.mean_entropy > 0.65 ? 'text-red-400' : 'text-green-400'}>
                  {worldState.confusion?.mean_entropy?.toFixed(4)}
                </span>
                <span className="text-gray-500">Confidence</span>
                <span className="text-blue-400">{worldState.confusion?.top1_confidence?.toFixed(4)}</span>
                <span className="text-gray-500">Agent</span>
                <span className="text-yellow-400">{worldState.dominant_agent}</span>
              </div>
              {/* Entropy bar per layer */}
              {worldState.confusion?.layer_entropies?.length > 0 && (
                <div className="mt-2">
                  <p className="text-xs text-gray-500 mb-1">Layer Entropy</p>
                  <div className="flex gap-1">
                    {worldState.confusion.layer_entropies.map((e, i) => (
                      <div key={i} className="flex-1">
                        <div
                          className="rounded-sm"
                          style={{
                            height: `${Math.max(4, e * 40)}px`,
                            background: e > 0.65 ? '#ef4444' : e > 0.40 ? '#f59e0b' : '#10b981',
                          }}
                        />
                        <p className="text-center text-gray-600" style={{fontSize:'9px'}}>L{i}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Log Stream */}
          <div className="flex-1 p-4 overflow-y-auto">
            <p className="text-xs text-gray-500 uppercase tracking-widest mb-2">Pipeline Log</p>
            {logs.map((l, i) => (
              <p key={i} className="text-xs text-gray-400 mb-1 leading-relaxed break-all">
                {l}
              </p>
            ))}
            <div ref={logEndRef} />
          </div>
        </div>

        {/* Right: tldraw canvas */}
        <div className="flex-1">
          <Tldraw />
        </div>
      </div>
    </main>
  );
}

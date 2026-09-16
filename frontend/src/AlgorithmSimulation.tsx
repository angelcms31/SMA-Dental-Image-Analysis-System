import React, { useEffect, useRef, useState } from 'react';

// ---------------------------------------------------------------------
// AlgorithmSimulation.tsx
// Four tabs:
//   SOP 1 / SOP 2 / SOP 3 -- ONE panel, plain SMA, each highlighting
//                            the specific problem that objective fixes
//                            (single-leader collapse, random-init gaps,
//                            or a fixed step size that never adapts).
//   All Objectives         -- TWO panels, SMA vs ESMA, with all three
//                            mechanisms combined on the ESMA side --
//                            this is where the comparison finally shows up.
//
// Simplified 2D teaching aid, not the real 4D search. Dots = agents,
// gold star = the true best answer (hidden from the algorithm itself).
// ---------------------------------------------------------------------

interface Agent {
  x: number;
  y: number;
  f: number;
}

const PEAKS = [
  { x: 0.75, y: 0.72, h: 1.0, s: 0.09 },
  { x: 0.22, y: 0.25, h: 0.62, s: 0.10 },
  { x: 0.68, y: 0.20, h: 0.55, s: 0.08 },
  { x: 0.18, y: 0.78, h: 0.5, s: 0.09 },
];
const BEST_PEAK = PEAKS[0];
const N_AGENTS = 12;
const K_LEADERS = 3;
const MAX_ITERS = 40;
const CANVAS_SIZE = 320;

function fitness(x: number, y: number): number {
  let v = 0;
  for (const p of PEAKS) {
    const d = ((x - p.x) ** 2 + (y - p.y) ** 2) / (2 * p.s * p.s);
    v = Math.max(v, p.h * Math.exp(-d));
  }
  return v;
}

function latinHypercubeInit(): Agent[] {
  const px = [...Array(N_AGENTS).keys()].sort(() => Math.random() - 0.5);
  const py = [...Array(N_AGENTS).keys()].sort(() => Math.random() - 0.5);
  return px.map((pi, i) => {
    const x = (pi + Math.random()) / N_AGENTS;
    const y = (py[i] + Math.random()) / N_AGENTS;
    return { x, y, f: fitness(x, y) };
  });
}

function randomUniformInit(): Agent[] {
  return Array.from({ length: N_AGENTS }, () => {
    const x = Math.random();
    const y = Math.random();
    return { x, y, f: fitness(x, y) };
  });
}

type TabKey = 1 | 2 | 3 | 'all';

interface Mechanism {
  multiLeader: boolean;
  lhsInit: boolean;
  adaptive: boolean;
}

const BASELINE: Mechanism = { multiLeader: false, lhsInit: false, adaptive: false };
const FULL: Mechanism = { multiLeader: true, lhsInit: true, adaptive: true };

const SOP_INFO: Record<1 | 2 | 3, { title: string; desc: string }> = {
  1: {
    title: 'SOP 1 \u2014 Single-Leader Dependence',
    desc: 'Standard SMA: every agent follows ONE leader. Watch the agents converge onto a single point \u2014 if that point isn\u2019t the true best, the whole population is stuck there.',
  },
  2: {
    title: 'SOP 2 \u2014 Random Uniform Initialization',
    desc: 'Standard SMA: starting positions are placed completely at random. Notice how agents can clump together, leaving other areas of the space unexplored from the very first iteration.',
  },
  3: {
    title: 'SOP 3 \u2014 Fixed Parameter Scheduling',
    desc: 'Standard SMA: the step size a(t) never changes. It doesn\u2019t shrink to fine-tune a good answer, and doesn\u2019t grow again if the search gets stuck.',
  },
};

function stepAgents(agents: Agent[], a: number, m: Mechanism): { agents: Agent[]; leaderIdx: Set<number> } {
  const ranked = [...agents].sort((p, q) => q.f - p.f);
  const leaderCount = m.multiLeader ? K_LEADERS : 1;
  const leaders = ranked.slice(0, leaderCount);
  const leaderIdx = m.multiLeader ? new Set(leaders.map((l) => agents.indexOf(l))) : new Set<number>();
  const wsum = m.multiLeader ? (K_LEADERS * (K_LEADERS + 1)) / 2 : 1;

  const next = agents.map((ag) => {
    let L: Agent;
    if (!m.multiLeader) {
      L = leaders[0];
    } else {
      let r = Math.random() * wsum;
      let li = 0;
      let acc = 0;
      for (let j = 0; j < K_LEADERS; j++) {
        acc += K_LEADERS - j;
        if (r <= acc) {
          li = j;
          break;
        }
      }
      L = leaders[li];
    }
    const nx = Math.min(1, Math.max(0, L.x + (Math.random() * 2 - 1) * a * 0.5));
    const ny = Math.min(1, Math.max(0, L.y + (Math.random() * 2 - 1) * a * 0.5));
    const nf = fitness(nx, ny);
    return nf > ag.f ? { x: nx, y: ny, f: nf } : ag;
  });
  return { agents: next, leaderIdx };
}

function drawPanel(canvas: HTMLCanvasElement, agents: Agent[], leaderIdx: Set<number>, accentColor: string) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const W = CANVAS_SIZE;
  const res = 42;
  const cell = W / res;

  for (let i = 0; i < res; i++) {
    for (let j = 0; j < res; j++) {
      const v = fitness((i + 0.5) / res, 1 - (j + 0.5) / res);
      const t = Math.min(1, v);
      const c1 = [246, 244, 240];
      const c2 = [124, 90, 210];
      const r = Math.round(c1[0] + (c2[0] - c1[0]) * t);
      const g = Math.round(c1[1] + (c2[1] - c1[1]) * t);
      const b = Math.round(c1[2] + (c2[2] - c1[2]) * t);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(i * cell, j * cell, cell + 0.5, cell + 0.5);
    }
  }

  ctx.fillStyle = '#F2C230';
  ctx.beginPath();
  const sx = BEST_PEAK.x * W;
  const sy = (1 - BEST_PEAK.y) * W;
  for (let k = 0; k < 5; k++) {
    const ang = -Math.PI / 2 + (k * 2 * Math.PI) / 5;
    const ang2 = ang + Math.PI / 5;
    ctx.lineTo(sx + Math.cos(ang) * 8, sy + Math.sin(ang) * 8);
    ctx.lineTo(sx + Math.cos(ang2) * 3.5, sy + Math.sin(ang2) * 3.5);
  }
  ctx.closePath();
  ctx.fill();

  agents.forEach((ag, i) => {
    const px = ag.x * W;
    const py = (1 - ag.y) * W;
    const isLeader = leaderIdx.has(i);
    if (isLeader) {
      ctx.beginPath();
      ctx.arc(px, py, 8, 0, 7);
      ctx.strokeStyle = '#EF9F27';
      ctx.lineWidth = 2.2;
      ctx.stroke();
    }
    ctx.globalAlpha = 0.68;
    ctx.beginPath();
    ctx.arc(px, py, 5, 0, 7);
    ctx.fillStyle = isLeader ? '#EF9F27' : accentColor;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.arc(px, py, 5, 0, 7);
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 1;
    ctx.stroke();
  });
}

function usePanelState(m: Mechanism) {
  const [agents, setAgents] = useState<Agent[]>(() => (m.lhsInit ? latinHypercubeInit() : randomUniformInit()));
  const [a, setA] = useState(1.0);
  const aRef = useRef(1.0); // step() reads/writes through this ref so it
                             // always uses the CURRENT step size, even
                             // though it's invoked from a setInterval
                             // callback captured on an earlier render.
  const [best, setBest] = useState(() => Math.max(...agents.map((ag) => ag.f)));
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const leaderIdxRef = useRef<Set<number>>(new Set());

  const reset = () => {
    const fresh = m.lhsInit ? latinHypercubeInit() : randomUniformInit();
    setAgents(fresh);
    aRef.current = 1.0;
    setA(1.0);
    setBest(Math.max(...fresh.map((ag) => ag.f)));
    leaderIdxRef.current = new Set();
  };

  const step = () => {
    setAgents((prev) => {
      const { agents: next, leaderIdx: li } = stepAgents(prev, aRef.current, m);
      leaderIdxRef.current = li;
      const b = Math.max(...next.map((ag) => ag.f));
      setBest((oldBest) => {
        const progressed = b > oldBest + 1e-4;
        if (m.adaptive) {
          aRef.current = progressed ? Math.max(0.05, aRef.current * 0.9) : Math.min(1.0, aRef.current + 0.03);
          setA(aRef.current);
        }
        return Math.max(oldBest, b);
      });
      return next;
    });
  };

  useEffect(() => {
    if (canvasRef.current) drawPanel(canvasRef.current, agents, leaderIdxRef.current, '#577E89');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agents]);

  return { canvasRef, a, best, reset, step };
}

function SopPanel({ label, canvasRef, a, best, accent }: any) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-sm font-semibold" style={{ color: accent }}>
          {label}
        </span>
        <span className="text-xs text-slate-400">
          best: {best.toFixed(3)} &middot; a(t) = {a.toFixed(2)}
        </span>
      </div>
      <canvas ref={canvasRef} width={CANVAS_SIZE} height={CANVAS_SIZE} className="w-full aspect-square rounded-lg border border-slate-200" />
    </div>
  );
}

const TABS: { key: TabKey; label: string }[] = [
  { key: 1, label: 'SOP 1' },
  { key: 2, label: 'SOP 2' },
  { key: 3, label: 'SOP 3' },
  { key: 'all', label: 'All Objectives' },
];

export default function AlgorithmSimulation({ onClose }: { onClose?: () => void }) {
  const [tab, setTab] = useState<TabKey>(1);
  const isAllTab = tab === 'all';
  const sopInfo = !isAllTab ? SOP_INFO[tab as 1 | 2 | 3] : null;

  const [iter, setIter] = useState(0);
  const [running, setRunning] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // SOP tabs: single SMA panel, plain baseline. "All Objectives" tab:
  // SMA (baseline) vs ESMA (all three mechanisms combined).
  const sma = usePanelState(BASELINE);
  const esma = usePanelState(FULL);

  const resetAll = () => {
    setIter(0);
    setRunning(false);
    sma.reset();
    esma.reset();
    if (timerRef.current) clearInterval(timerRef.current);
  };

  useEffect(() => {
    resetAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab]);

  useEffect(() => {
    if (running) {
      timerRef.current = setInterval(() => {
        setIter((prev) => {
          if (prev >= MAX_ITERS) {
            setRunning(false);
            return prev;
          }
          sma.step();
          if (isAllTab) esma.step();
          return prev + 1;
        });
      }, 450);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running]);

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-6 overflow-y-auto">
      <div className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full p-6">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-lg font-semibold text-slate-800">Algorithm Simulation</h2>
            <p className="text-xs text-slate-400 mt-0.5">Simplified 2D view of the real search behavior.</p>
          </div>
          {onClose && (
            <button onClick={onClose} className="text-slate-400 hover:text-slate-600 text-2xl leading-none px-2" aria-label="Close simulation">
              ×
            </button>
          )}
        </div>

        <div className="inline-flex bg-slate-100 rounded-full p-1 gap-1 mb-4 flex-wrap">
          {TABS.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all ${
                tab === t.key ? 'text-white shadow-sm' : 'text-slate-500 hover:text-slate-700'
              }`}
              style={tab === t.key ? { backgroundColor: '#3A5661' } : undefined}
            >
              {t.label}
            </button>
          ))}
        </div>

        {sopInfo && <p className="text-sm font-medium text-slate-700 mb-1">{sopInfo.title}</p>}
        {sopInfo && <p className="text-xs text-slate-400 mb-3 leading-relaxed">{sopInfo.desc}</p>}
        {isAllTab && <p className="text-sm font-medium text-slate-700 mb-3">All three objectives combined: SMA vs. ESMA</p>}

        <div className="flex items-center gap-3 mb-4">
          <button
            onClick={() => setRunning((r) => !r)}
            disabled={iter >= MAX_ITERS}
            className="text-white px-5 py-2 rounded-full font-medium text-sm disabled:opacity-40 transition-colors"
            style={{ backgroundColor: '#3A5661' }}
          >
            {running ? '⏸ Pause' : '▶ Play'}
          </button>
          <button onClick={resetAll} className="px-5 py-2 rounded-full font-medium text-sm border border-slate-300 text-slate-600 hover:bg-slate-50 transition-colors">
            ↻ Reset
          </button>
          <span className="text-sm text-slate-500 ml-auto">
            Iteration {iter} / {MAX_ITERS}
          </span>
        </div>

        {isAllTab ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            <SopPanel label="SMA" canvasRef={sma.canvasRef} a={sma.a} best={sma.best} accent="#577E89" />
            <SopPanel label="ESMA" canvasRef={esma.canvasRef} a={esma.a} best={esma.best} accent="#E1A36F" />
          </div>
        ) : (
          <div className="max-w-sm mx-auto">
            <SopPanel label="SMA" canvasRef={sma.canvasRef} a={sma.a} best={sma.best} accent="#577E89" />
          </div>
        )}

        <div className="mt-4 pt-4 border-t border-slate-100 grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: '#577E89', opacity: 0.75, border: '1px solid #fff' }} />
            <span className="text-xs text-slate-600">Agent</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3.5 h-3.5 rounded-full shrink-0 border-2" style={{ borderColor: '#EF9F27' }} />
            <span className="text-xs text-slate-600">Current leader(s)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm shrink-0" style={{ color: '#F2C230' }}>&#9733;</span>
            <span className="text-xs text-slate-600">True best answer</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-600">Darker dot = overlapping agents</span>
          </div>
        </div>
      </div>
    </div>
  );
}
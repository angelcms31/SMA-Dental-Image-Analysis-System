import React, { useCallback, useEffect, useRef, useState } from 'react';
import axios from 'axios';

// ---------------------------------------------------------------------
// AlgorithmSimulation.tsx
//
// Every dot, leader ring, star and curve in this view is data returned
// by the backend endpoint POST /simulate/agents/ (simulation_api.py),
// which runs the real standard_sma() and enhanced_sma() from
// sma_algorithms.py on the uploaded OPG image and records the position
// of every agent at every iteration. This file only draws; it contains
// no search logic and no synthetic fitness landscape.
//
// An agent is a vector of d thresholds (d = 4), so its position is
// 4-dimensional. Each panel plots two of those thresholds against each
// other; the axis pickers choose which two.
//
// The heat map behind the agents is also real: it is a 2D slice of the
// Kapur-entropy landscape of the uploaded image (the two plotted
// thresholds vary, the others are held at their exact-optimum values),
// computed by the backend. The global and local optimum stars are the
// peaks of that slice.
// ---------------------------------------------------------------------

interface AlgoRun {
  algorithm: string;
  thresholds: number[];
  fitness: number;
  runtime_sec: number;
  iterations_used: number;
  k_leaders: number;
  convergence: number[];
  a_trace: number[];
  positions: number[][][]; // [iteration][agent][threshold]
  leaders: number[][]; // [iteration][leader rank] -> agent index
  segmented_image: string;
  segmented_color?: string;
  init_unsorted?: number[][];
}

interface Landscape {
  step: number;
  grid: number[][]; // grid[x index][y index], real Kapur entropy on a 2D slice
  max: number;
  floor: number;
  local_optima: { x: number; y: number; fitness: number }[];
}

interface SimResponse {
  status: 'success' | 'error';
  message?: string;
  params?: { d: number; N: number; T: number; seed: number };
  optimum?: { thresholds: number[]; fitness: number };
  landscapes?: Record<string, Landscape>; // key "i-j" with i < j
  input_image?: string;
  histogram?: number[]; // pixel count per gray level, 0-255
  standard?: AlgoRun;
  enhanced?: AlgoRun;
}

type View = 'agents' | 'chart' | 'images';

const STANDARD_COLOR = '#577E89';
const ENHANCED_COLOR = '#E1A36F';
const LEADER_COLOR = '#EF9F27';
const STAR_COLOR = '#F2C230';
const PANEL_BG = '#0B1017';
interface LandscapeView {
  value: (xi: number, yi: number) => number; // entropy at grid cell
  size: number;
  step: number;
  max: number;
  floor: number;
  local: { x: number; y: number }[];
  flipped: boolean; // true when the x-axis threshold is the larger one
}

const ANIM_INTERVAL_MS = 500; // ms per iteration; raise to slow down, lower to speed up

const PLOT = 360;
const PAD = { left: 34, right: 8, top: 8, bottom: 26 };
const INNER = PLOT - PAD.left - PAD.right;

const sortAsc = (v: number[]) => [...v].sort((p, q) => p - q);

// ---------------------------------------------------------------------
// Agents panel: grid of N strata per axis, one dot per agent.
// ---------------------------------------------------------------------
function drawAgents(
  canvas: HTMLCanvasElement,
  agents: number[][],
  leaderIdx: number[],
  opts: {
    xDim: number;
    yDim: number;
    strata: number;
    showEmptyStrata: boolean;
    star: number[] | null;
    accent: string;
    landscape: LandscapeView | null;
    leaderLabel: string | null;
  }
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const { xDim, yDim, strata, showEmptyStrata, star, accent, landscape, leaderLabel } = opts;
  if (landscape && star) {
    drawLandscapePanel(ctx, agents, leaderIdx, { xDim, yDim, star, accent, landscape, leaderLabel });
    return;
  }
  const px = (v: number) => PAD.left + (v / 255) * INNER;
  const py = (v: number) => PAD.top + INNER - (v / 255) * INNER;
  const cell = INNER / strata;

  ctx.clearRect(0, 0, PLOT, PLOT);
  ctx.fillStyle = PANEL_BG;
  ctx.fillRect(0, 0, PLOT, PLOT);

  // strata that contain no agent at the start (only drawn at iteration 0)
  if (showEmptyStrata) {
    const usedX = new Set<number>();
    const usedY = new Set<number>();
    agents.forEach((a) => {
      usedX.add(Math.min(strata - 1, Math.floor((a[xDim] / 255) * strata)));
      usedY.add(Math.min(strata - 1, Math.floor((a[yDim] / 255) * strata)));
    });
    ctx.fillStyle = 'rgba(214, 170, 40, 0.20)';
    for (let s = 0; s < strata; s++) {
      if (!usedX.has(s)) ctx.fillRect(PAD.left + s * cell, PAD.top, cell, INNER);
      if (!usedY.has(s)) ctx.fillRect(PAD.left, PAD.top + INNER - (s + 1) * cell, INNER, cell);
    }
  }

  // strata grid
  ctx.strokeStyle = 'rgba(130, 160, 180, 0.22)';
  ctx.lineWidth = 1;
  for (let s = 0; s <= strata; s++) {
    const o = Math.round(s * cell) + 0.5;
    ctx.beginPath();
    ctx.moveTo(PAD.left + o, PAD.top);
    ctx.lineTo(PAD.left + o, PAD.top + INNER);
    ctx.moveTo(PAD.left, PAD.top + o);
    ctx.lineTo(PAD.left + INNER, PAD.top + o);
    ctx.stroke();
  }

  // exact optimum (answer key), only when thresholds are shown sorted
  if (star) {
    const sx = px(star[xDim]);
    const sy = py(star[yDim]);
    ctx.fillStyle = STAR_COLOR;
    ctx.beginPath();
    for (let k = 0; k < 5; k++) {
      const ang = -Math.PI / 2 + (k * 2 * Math.PI) / 5;
      const ang2 = ang + Math.PI / 5;
      ctx.lineTo(sx + Math.cos(ang) * 9, sy + Math.sin(ang) * 9);
      ctx.lineTo(sx + Math.cos(ang2) * 4, sy + Math.sin(ang2) * 4);
    }
    ctx.closePath();
    ctx.fill();
  }

  // agents
  const leaders = new Set(leaderIdx);
  agents.forEach((a, i) => {
    const x = px(a[xDim]);
    const y = py(a[yDim]);
    const isLeader = leaders.has(i);
    if (isLeader) {
      ctx.beginPath();
      ctx.arc(x, y, 8.5, 0, Math.PI * 2);
      ctx.strokeStyle = LEADER_COLOR;
      ctx.lineWidth = 2.2;
      ctx.stroke();
    }
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    ctx.arc(x, y, 4.6, 0, Math.PI * 2);
    ctx.fillStyle = isLeader ? LEADER_COLOR : '#E8EEF2';
    ctx.fill();
    ctx.globalAlpha = 1;
  });

  // frame + axis labels
  ctx.strokeStyle = accent;
  ctx.lineWidth = 1.5;
  ctx.strokeRect(PAD.left + 0.5, PAD.top + 0.5, INNER, INNER);
  ctx.fillStyle = '#8FA3AE';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(`threshold ${xDim + 1}  (0–255)`, PAD.left + INNER / 2, PLOT - 8);
  ctx.save();
  ctx.translate(11, PAD.top + INNER / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(`threshold ${yDim + 1}  (0–255)`, 0, 0);
  ctx.restore();
}

// ---------------------------------------------------------------------
// Landscape panel: real Kapur-entropy slice as a heat map, with the
// global optimum, the local optima of the slice, and the agents on top.
// ---------------------------------------------------------------------
function drawLandscapePanel(
  ctx: CanvasRenderingContext2D,
  agents: number[][],
  leaderIdx: number[],
  o: { xDim: number; yDim: number; star: number[]; accent: string; landscape: LandscapeView; leaderLabel: string | null }
) {
  const { xDim, yDim, star, accent, landscape } = o;
  const W = PLOT;

  // Fill the whole box: thresholds are sorted (x-axis threshold < y-axis
  // threshold), so on equal 0-255 axes half of the square can never occur.
  // Instead the x-axis covers 0..split and the y-axis covers split..255,
  // where `split` lies between the two thresholds of every marked optimum,
  // so every point of the box is a possible pair.
  // The window is placed around the true best answer so the star sits
  // near the middle of the box (60% across, 40% up), not on the edge.
  const optX = star[xDim];
  const optY = star[yDim];
  const zoomed = !landscape.flipped && optX + 6 < optY;
  const split = zoomed ? (optX + optY) / 2 : 0;
  const span = 2.5 * (split - optX);
  const xLo = zoomed ? Math.max(0, split - span) : 0;
  const xHi = zoomed ? split : 255;
  const yLo = zoomed ? split : 0;
  const yHi = zoomed ? Math.min(255, split + span) : 255;
  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
  const px = (v: number) => ((clamp(v, xLo, xHi) - xLo) / (xHi - xLo)) * W;
  const py = (v: number) => W - ((clamp(v, yLo, yHi) - yLo) / (yHi - yLo)) * W;

  // real Kapur-entropy landscape, sampled (bilinear) from the backend grid
  // and drawn as soft cells: off-white for low entropy, purple for high
  const n = landscape.size;
  const sample = (xv: number, yv: number) => {
    const gx = clamp(xv / landscape.step, 0, n - 1);
    const gy = clamp(yv / landscape.step, 0, n - 1);
    const x0 = Math.floor(gx);
    const y0 = Math.floor(gy);
    const x1 = Math.min(n - 1, x0 + 1);
    const y1 = Math.min(n - 1, y0 + 1);
    const fx = gx - x0;
    const fy = gy - y0;
    return (
      landscape.value(x0, y0) * (1 - fx) * (1 - fy) +
      landscape.value(x1, y0) * fx * (1 - fy) +
      landscape.value(x0, y1) * (1 - fx) * fy +
      landscape.value(x1, y1) * fx * fy
    );
  };
  const res = 42;
  const cell = W / res;
  const range = landscape.max - landscape.floor || 1;
  const c1 = [246, 244, 240];
  const c2 = [124, 90, 210];
  ctx.clearRect(0, 0, W, W);
  for (let i = 0; i < res; i++) {
    for (let j = 0; j < res; j++) {
      const xv = xLo + ((i + 0.5) / res) * (xHi - xLo);
      const yv = yHi - ((j + 0.5) / res) * (yHi - yLo);
      const impossible = !zoomed && (landscape.flipped ? xv < yv : xv > yv);
      const t = impossible ? 0 : Math.pow(clamp((sample(xv, yv) - landscape.floor) / range, 0, 1), 1.6);
      const r = Math.round(c1[0] + (c2[0] - c1[0]) * t);
      const g = Math.round(c1[1] + (c2[1] - c1[1]) * t);
      const b = Math.round(c1[2] + (c2[2] - c1[2]) * t);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(i * cell, j * cell, cell + 0.5, cell + 0.5);
    }
  }

  // gold star: the exact optimum (answer key)
  ctx.fillStyle = STAR_COLOR;
  ctx.beginPath();
  const sx = px(star[xDim]);
  const sy = py(star[yDim]);
  for (let k = 0; k < 5; k++) {
    const ang = -Math.PI / 2 + (k * 2 * Math.PI) / 5;
    const ang2 = ang + Math.PI / 5;
    ctx.lineTo(sx + Math.cos(ang) * 8, sy + Math.sin(ang) * 8);
    ctx.lineTo(sx + Math.cos(ang2) * 3.5, sy + Math.sin(ang2) * 3.5);
  }
  ctx.closePath();
  ctx.fill();

  // agents; leaders get an orange ring
  const leaders = new Set(leaderIdx);
  agents.forEach((a, i) => {
    const x = clamp(px(a[xDim]), 5, W - 5);
    const y = clamp(py(a[yDim]), 5, W - 5);
    const isLeader = leaders.has(i);
    if (isLeader) {
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, 7);
      ctx.strokeStyle = LEADER_COLOR;
      ctx.lineWidth = 2.2;
      ctx.stroke();
    }
    ctx.globalAlpha = 0.68;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 7);
    ctx.fillStyle = isLeader ? LEADER_COLOR : accent;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 7);
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 1;
    ctx.stroke();
  });
}

// ---------------------------------------------------------------------
// Convergence chart: best Kapur entropy per iteration, both algorithms.
// ---------------------------------------------------------------------
const CH_W = 640;
const CH_H = 260;
const CH_PAD = { top: 16, right: 16, bottom: 28, left: 52 };

// Gray-level histogram of the input image with the thresholds found by
// each algorithm drawn as vertical lines.
function drawHistogram(canvas: HTMLCanvasElement, hist: number[], stdT: number[], esmaT: number[]) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  const pad = { left: 44, right: 12, top: 12, bottom: 30 };
  const iw = W - pad.left - pad.right;
  const ih = H - pad.top - pad.bottom;
  // the 0 and 255 bins are often huge (background / saturation); scale to the rest
  const peak = Math.max(1, ...hist.slice(1, 255));
  const xAt = (g: number) => pad.left + (g / 255) * iw;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = '#94A3B8';
  const bw = iw / 256;
  hist.forEach((v, g) => {
    const h = Math.min(1, v / peak) * ih;
    ctx.fillRect(pad.left + g * bw, pad.top + ih - h, bw + 0.5, h);
  });
  ctx.strokeStyle = '#CBD5E1';
  ctx.lineWidth = 1;
  ctx.strokeRect(pad.left, pad.top, iw, ih);
  ctx.fillStyle = '#64748B';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  [0, 50, 100, 150, 200, 250].forEach((g) => ctx.fillText(String(g), xAt(g), H - 16));
  ctx.fillText('Gray level', pad.left + iw / 2, H - 3);
  ctx.textAlign = 'right';
  ctx.fillText(String(peak), pad.left - 4, pad.top + 9);
  ctx.fillText('0', pad.left - 4, pad.top + ih);
  const lines = (ts: number[], color: string, dash: number[]) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash(dash);
    ts.forEach((t) => {
      ctx.beginPath();
      ctx.moveTo(xAt(t), pad.top);
      ctx.lineTo(xAt(t), pad.top + ih);
      ctx.stroke();
    });
    ctx.setLineDash([]);
  };
  lines(stdT, STANDARD_COLOR, [5, 4]);
  lines(esmaT, ENHANCED_COLOR, []);
}

function drawConvergence(canvas: HTMLCanvasElement, std: number[], esma: number[], optimum: number, upTo: number) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const innerW = CH_W - CH_PAD.left - CH_PAD.right;
  const innerH = CH_H - CH_PAD.top - CH_PAD.bottom;
  const T = std.length - 1;

  ctx.clearRect(0, 0, CH_W, CH_H);
  ctx.fillStyle = '#FAFAF8';
  ctx.fillRect(0, 0, CH_W, CH_H);

  const all = [...std, ...esma, optimum];
  const minV = Math.min(...all);
  const maxV = Math.max(...all);
  const span = maxV - minV || 1;
  const lo = minV - span * 0.08;
  const hi = maxV + span * 0.08;
  const xAt = (i: number) => CH_PAD.left + (T === 0 ? 0 : (i / T) * innerW);
  const yAt = (v: number) => CH_PAD.top + innerH - ((v - lo) / (hi - lo)) * innerH;

  ctx.strokeStyle = '#E5E1D8';
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = CH_PAD.top + (g / 4) * innerH;
    ctx.beginPath();
    ctx.moveTo(CH_PAD.left, y);
    ctx.lineTo(CH_W - CH_PAD.right, y);
    ctx.stroke();
  }

  // exact optimum from dynamic programming (answer key)
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = STAR_COLOR;
  ctx.beginPath();
  ctx.moveTo(CH_PAD.left, yAt(optimum));
  ctx.lineTo(CH_W - CH_PAD.right, yAt(optimum));
  ctx.stroke();
  ctx.setLineDash([]);

  const line = (curve: number[], color: string) => {
    // ESMA can stop early, so its curve may be shorter than Standard SMA's
    const end = Math.min(upTo, curve.length - 1);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let i = 0; i <= end; i++) {
      if (i === 0) ctx.moveTo(xAt(i), yAt(curve[i]));
      else ctx.lineTo(xAt(i), yAt(curve[i]));
    }
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(xAt(end), yAt(curve[end]), 4.2, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  };
  line(std, STANDARD_COLOR);
  line(esma, ENHANCED_COLOR);

  ctx.fillStyle = '#94A3A8';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText(hi.toFixed(3), CH_PAD.left - 6, CH_PAD.top + 4);
  ctx.fillText(lo.toFixed(3), CH_PAD.left - 6, CH_PAD.top + innerH);
  ctx.textAlign = 'left';
  ctx.fillText('iter 0', CH_PAD.left, CH_H - 8);
  ctx.textAlign = 'right';
  ctx.fillText(`iter ${T}`, CH_W - CH_PAD.right, CH_H - 8);
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-50 rounded-lg px-3 py-2">
      <p className="text-[10px] uppercase tracking-wide text-slate-400">{label}</p>
      <p className="text-sm font-semibold text-slate-700">{value}</p>
    </div>
  );
}

export default function AlgorithmSimulation({
  onClose,
  selectedFile,
  apiBase = 'http://localhost:8000',
}: {
  onClose?: () => void;
  selectedFile: File | null;
  apiBase?: string;
}) {
  const N = 30;
  const T = 100;
  const [seed, setSeed] = useState(42);

  const [status, setStatus] = useState<'idle' | 'loading' | 'error' | 'done'>('idle');
  const [errorMsg, setErrorMsg] = useState('');
  const [data, setData] = useState<SimResponse | null>(null);

  const [view, setView] = useState<View>('agents');
  const [iter, setIter] = useState(0);
  const [running, setRunning] = useState(false);
  const xDim: number = 0; // threshold 1 on the x-axis
  const yDim: number = 1; // threshold 2 on the y-axis
  
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const stdCanvas = useRef<HTMLCanvasElement>(null);
  const esmaCanvas = useRef<HTMLCanvasElement>(null);
  const chartCanvas = useRef<HTMLCanvasElement>(null);
  const histCanvas = useRef<HTMLCanvasElement>(null);

  const runSimulation = useCallback(async () => {
    if (!selectedFile) return;
    setStatus('loading');
    setErrorMsg('');
    setRunning(false);
    setIter(0);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('d', '4');
      formData.append('N', String(N));
      formData.append('T', String(T));
      formData.append('seed', String(seed));
      const res = await axios.post<SimResponse>(`${apiBase}/simulate/agents/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      if (res.data.status !== 'success' || !res.data.standard || !res.data.enhanced || !res.data.optimum) {
        setStatus('error');
        setErrorMsg(res.data.message || 'Backend returned an error while running the simulation.');
        return;
      }
      setData(res.data);
      setStatus('done');
    } catch (err: any) {
      setStatus('error');
      const msg = err?.message || String(err);
      setErrorMsg(
        msg.toLowerCase().includes('network')
          ? `Could not reach the backend at ${apiBase}. Start it first: uvicorn main:app --reload --port 8000`
          : err?.response?.status === 404
          ? 'The backend has no /simulate/agents/ endpoint yet. Add simulation_api.py and include its router in main.py.'
          : msg
      );
    }
  }, [selectedFile, N, T, seed, apiBase]);

  useEffect(() => {
    if (selectedFile) runSimulation();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedFile]);

  const std = data?.standard;
  const esma = data?.enhanced;
  const optimum = data?.optimum;
  const strata = data?.params?.N ?? N;
  const maxIter = std ? std.positions.length - 1 : 0;

  useEffect(() => {
    if (running && data) {
      timerRef.current = setInterval(() => {
        setIter((prev) => {
          if (prev >= maxIter) {
            setRunning(false);
            return prev;
          }
          return prev + 1;
        });
      }, ANIM_INTERVAL_MS);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [running, data, maxIter]);

  // At iteration 0 the raw starting positions are shown (this is where the
  // difference between the two starting populations is visible). From iteration 1 on,
  // each agent's thresholds are shown smallest-to-largest so both panels
  // share the same axes and the exact optimum can be marked.
  const rawStart: boolean = false; // agents are always drawn on the landscape

  // real landscape slice for the chosen pair of thresholds (if any)
  let landscape: LandscapeView | null = null;
  if (data?.landscapes && xDim !== yDim) {
    const lo = Math.min(xDim, yDim);
    const hi = Math.max(xDim, yDim);
    const L = data.landscapes[`${lo}-${hi}`];
    if (L) {
      const flip = xDim > yDim; // stored as grid[lo threshold][hi threshold]
      landscape = {
        value: (xi, yi) => (flip ? L.grid[yi][xi] : L.grid[xi][yi]),
        size: L.grid.length,
        step: L.step,
        max: L.max,
        floor: L.floor,
        local: L.local_optima.map((p) => (flip ? { x: p.y, y: p.x } : { x: p.x, y: p.y })),
        flipped: flip,
      };
    }
  }
  const esmaIdx = esma ? Math.min(iter, esma.positions.length - 1) : 0;
  const stdFrame = std ? (rawStart ? std.positions[0] : std.positions[iter].map(sortAsc)) : [];
  const esmaFrame = esma
    ? rawStart && esma.init_unsorted
      ? esma.init_unsorted
      : esma.positions[esmaIdx]
    : [];

  useEffect(() => {
    if (view !== 'agents' || !std || !esma || !optimum) return;
    const star = rawStart ? null : optimum.thresholds;
    if (stdCanvas.current)
      drawAgents(stdCanvas.current, stdFrame, std.leaders[iter], {
        xDim, yDim, strata, showEmptyStrata: rawStart, star, accent: STANDARD_COLOR,
        landscape, leaderLabel: 'Xb (single best)',
      });
    if (esmaCanvas.current)
      drawAgents(esmaCanvas.current, esmaFrame, esma.leaders[esmaIdx], {
        xDim, yDim, strata, showEmptyStrata: rawStart, star, accent: STANDARD_COLOR,
        landscape, leaderLabel: `top-${esma.k_leaders} leaders`,
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, iter, data]);

  useEffect(() => {
    if (view === 'chart' && chartCanvas.current && std && esma && optimum) {
      drawConvergence(chartCanvas.current, std.convergence, esma.convergence, optimum.fitness, iter);
    }
  }, [view, iter, data, std, esma, optimum]);

  useEffect(() => {
    if (view === 'images' && histCanvas.current && data?.histogram && std && esma) {
      drawHistogram(histCanvas.current, data.histogram, std.thresholds, esma.thresholds);
    }
  }, [view, data, std, esma]);

  const stdBest = std ? std.convergence[iter] : 0;
  const esmaBest = esma ? esma.convergence[esmaIdx] : 0;
  const esmaStopped = !!esma && iter >= esma.iterations_used && esma.iterations_used < maxIter;

  const tabBtn = (key: View, label: string) => (
    <button
      onClick={() => setView(key)}
      className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all ${
        view === key ? 'text-white shadow-sm' : 'text-slate-500 hover:text-slate-700'
      }`}
      style={view === key ? { backgroundColor: '#3A5661' } : undefined}
    >
      {label}
    </button>
  );


  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-start justify-center p-6 overflow-y-auto">
      <div className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full p-6">
        <div className="flex items-center justify-between mb-1">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-semibold text-slate-800">Algorithm Simulation</h2>
              <span className="text-[10px] font-semibold uppercase tracking-wide bg-emerald-100 text-emerald-700 rounded-full px-2 py-0.5">
                Live backend run
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Real run of both algorithms on your uploaded image.
            </p>
          </div>
          {onClose && (
            <button onClick={onClose} className="text-slate-400 hover:text-slate-600 text-2xl leading-none px-2" aria-label="Close simulation">
              ×
            </button>
          )}
        </div>

        {!selectedFile && (
          <div className="mt-6 mb-2 text-center py-10 border border-dashed border-slate-200 rounded-xl">
            <p className="text-sm text-slate-500">Upload an OPG image in the analyzer first.</p>
            <p className="text-xs text-slate-400 mt-1">The simulation runs the real algorithms on that image.</p>
          </div>
        )}

        {selectedFile && status === 'loading' && (
          <div className="mt-6 mb-2 text-center py-10">
            <p className="text-sm text-slate-500">Running Standard SMA and ESMA on your image (N={N}, T={T})…</p>
          </div>
        )}

        {status === 'error' && (
          <div className="mt-4 mb-2 bg-red-50 border border-red-200 text-red-700 text-sm rounded-xl px-4 py-3">
            <p className="font-medium mb-1">Couldn&apos;t run the live simulation.</p>
            <p className="text-xs">{errorMsg}</p>
            <button onClick={runSimulation} className="mt-3 text-xs font-medium text-white px-4 py-1.5 rounded-full" style={{ backgroundColor: '#3A5661' }}>
              Retry
            </button>
          </div>
        )}

        {status === 'done' && std && esma && optimum && (
          <>
            <div className="flex flex-wrap items-center gap-2 mt-4 mb-3 text-xs text-slate-500">
              <label className="flex items-center gap-1">
                seed
                <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value) || 0)} className="w-20 border border-slate-200 rounded px-1.5 py-0.5" />
              </label>
              <button onClick={() => setSeed(100 + Math.floor(Math.random() * 900))} className="border border-slate-200 rounded-full px-3 py-1 hover:bg-slate-50">
                Randomize seed
              </button>
              <button onClick={runSimulation} className="text-white rounded-full px-3 py-1 font-medium" style={{ backgroundColor: '#3A5661' }}>
                Re-run on backend
              </button>
            </div>

            <div className="flex items-center gap-3 mb-4">
              <button
                onClick={() => setRunning((r) => !r)}
                disabled={iter >= maxIter}
                className="text-white px-5 py-2 rounded-full font-medium text-sm disabled:opacity-40 transition-colors"
                style={{ backgroundColor: '#3A5661' }}
              >
                {running ? 'Pause' : 'Play'}
              </button>
              <button
                onClick={() => {
                  setRunning(false);
                  setIter((i) => Math.min(maxIter, i + 1));
                }}
                disabled={iter >= maxIter}
                className="px-4 py-2 rounded-full font-medium text-sm border border-slate-300 text-slate-600 hover:bg-slate-50 disabled:opacity-40"
              >
                Step
              </button>
              <button
                onClick={() => {
                  setRunning(false);
                  setIter(0);
                }}
                className="px-4 py-2 rounded-full font-medium text-sm border border-slate-300 text-slate-600 hover:bg-slate-50"
              >
                Reset
              </button>
              <span className="text-sm text-slate-500 ml-auto">
                Iteration {iter} / {maxIter}
              </span>
            </div>

            <div className="inline-flex bg-slate-100 rounded-full p-1 gap-1 mb-4">
              {tabBtn('agents', 'Agents')}
              {tabBtn('chart', 'Convergence')}
              {tabBtn('images', 'Segmented Result')}
            </div>

            {view === 'agents' && (
              <>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-sm font-semibold" style={{ color: STANDARD_COLOR }}>
                        SMA
                      </span>
                      <span className="text-xs text-slate-400">
                        best: {stdBest.toFixed(3)} &middot; a(t) = {std.a_trace[iter].toFixed(2)}
                      </span>
                    </div>
                    <canvas ref={stdCanvas} width={PLOT} height={PLOT} className="w-full aspect-square rounded-lg border border-slate-200" />
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-sm font-semibold" style={{ color: ENHANCED_COLOR }}>
                        ESMA
                      </span>
                      <span className="text-xs text-slate-400">
                        best: {esmaBest.toFixed(3)} &middot; a(t) = {esma.a_trace[esmaIdx].toFixed(2)}
                        {esmaStopped ? ` \u00b7 stopped at ${esma.iterations_used}` : ''}
                      </span>
                    </div>
                    <canvas ref={esmaCanvas} width={PLOT} height={PLOT} className="w-full aspect-square rounded-lg border border-slate-200" />
                  </div>
                </div>

                <div className="mt-4 pt-4 border-t border-slate-100 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs text-slate-600">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: STANDARD_COLOR, opacity: 0.75, border: '1px solid #fff' }} />
                    <span className="text-xs text-slate-600">Agent</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-3.5 h-3.5 rounded-full shrink-0 border-2" style={{ borderColor: LEADER_COLOR }} />
                    <span className="text-xs text-slate-600">Current leader(s)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm shrink-0" style={{ color: STAR_COLOR }}>&#9733;</span>
                    <span className="text-xs text-slate-600">True best answer</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-600">Darker area = higher entropy</span>
                  </div>
                </div>
              </>
            )}

            {view === 'chart' && (
              <>
                <canvas ref={chartCanvas} width={CH_W} height={CH_H} className="w-full rounded-lg border border-slate-200" />
                <div className="grid grid-cols-3 gap-3 mt-4">
                  <Stat label="SMA best (so far)" value={stdBest.toFixed(4)} />
                  <Stat label="ESMA best (so far)" value={esmaBest.toFixed(4)} />
                  <Stat label="Exact optimum" value={optimum.fitness.toFixed(4)} />
                </div>
                <div className="flex flex-wrap items-center gap-4 mt-4 pt-4 border-t border-slate-100 text-xs text-slate-500">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: STANDARD_COLOR }} />
                    Standard SMA
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: ENHANCED_COLOR }} />
                    Enhanced SMA (ESMA){esma.iterations_used < maxIter ? `, stopped at iteration ${esma.iterations_used}` : ''}
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-4 border-t border-dashed" style={{ borderColor: STAR_COLOR }} />
                    Exact optimum (answer key)
                  </span>
                </div>
              </>
            )}

            {view === 'images' && (
              <>
                {data?.histogram && data?.input_image && (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
                    <div>
                      <p className="text-xs font-semibold mb-1.5 text-slate-600">Input image</p>
                      <img src={data.input_image} alt="Input OPG" className="w-full rounded-lg border border-slate-200" />
                    </div>
                    <div>
                      <p className="text-xs font-semibold mb-1.5 text-slate-600">The histogram of image</p>
                      <canvas ref={histCanvas} width={520} height={260} className="w-full rounded-lg border border-slate-200" />
                      <p className="text-[11px] text-slate-500 mt-1">
                        <span style={{ color: STANDARD_COLOR }}>- - -</span> SMA thresholds &nbsp;
                        <span style={{ color: ENHANCED_COLOR }}>——</span> ESMA thresholds
                      </p>
                    </div>
                  </div>
                )}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs font-semibold mb-1.5" style={{ color: STANDARD_COLOR }}>Standard SMA</p>
                    <div className="grid grid-cols-2 gap-2">
                      {std.segmented_color && (
                        <img src={std.segmented_color} alt="Standard SMA segmentation (colour)" className="w-full rounded-lg border border-slate-200" />
                      )}
                      <img src={std.segmented_image} alt="Standard SMA segmentation" className="w-full rounded-lg border border-slate-200" />
                    </div>
                  </div>
                  <div>
                    <p className="text-xs font-semibold mb-1.5" style={{ color: ENHANCED_COLOR }}>Enhanced SMA (ESMA)</p>
                    <div className="grid grid-cols-2 gap-2">
                      {esma.segmented_color && (
                        <img src={esma.segmented_color} alt="ESMA segmentation (colour)" className="w-full rounded-lg border border-slate-200" />
                      )}
                      <img src={esma.segmented_image} alt="ESMA segmentation" className="w-full rounded-lg border border-slate-200" />
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3 mt-4">
                  <Stat label="SMA entropy / thresholds" value={`${std.fitness.toFixed(4)} / [${std.thresholds.join(', ')}]`} />
                  <Stat label="ESMA entropy / thresholds" value={`${esma.fitness.toFixed(4)} / [${esma.thresholds.join(', ')}]`} />
                  <Stat label="SMA iterations / runtime" value={`${std.iterations_used} / ${std.runtime_sec.toFixed(3)} s`} />
                  <Stat label="ESMA iterations / runtime" value={`${esma.iterations_used} / ${esma.runtime_sec.toFixed(3)} s`} />
                </div>
                <p className="text-[11px] text-slate-400 mt-3">
                  Exact optimum: {optimum.fitness.toFixed(4)} at [{optimum.thresholds.join(', ')}].
                </p>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
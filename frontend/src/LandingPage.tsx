import React from 'react';
import Navbar, { type Navigate } from './Navbar';

const FONT_DISPLAY = "font-['Inter_Tight',sans-serif]";
const INK = 'var(--ink)';
const MUTED = 'var(--muted)';

// Plain-language facts about what the tool does (4 = the conditions the
// detection model can flag: caries, deep caries, impacted, periapical lesion).
const STATS: { value: string; label: string }[] = [
  { value: '4', label: 'Conditions it can flag' },
  { value: '1', label: 'Click to analyze' },
  { value: 'PDF', label: 'Report you can save' },
];

const STEPS: { title: string; body: string; icon: React.ReactNode }[] = [
  {
    title: 'Upload your OPG',
    body: 'Drag your panoramic dental image (OPG) into the analyzer or click "Choose an image". PNG and JPEG work.',
    icon: <path d="M12 16V4M7 9l5-5 5 5M5 20h14" />,
  },
  {
    title: 'Click Analyze',
    body: 'Press "Analyze OPG" and the system checks your image. It usually takes under a minute.',
    icon: <path d="M7 4.5v15l12-7.5z" />,
  },
  {
    title: 'See what was found',
    body: 'Your image and the highlighted areas appear side by side, with each finding explained in plain words.',
    icon: <path d="M4 5h7v14H4zM13 5h7v14h-7z" />,
  },
  {
    title: 'Save your report',
    body: 'Download a PDF with your image, the highlighted areas and the list of findings.',
    icon: <path d="M7 3h7l5 5v13H7zM14 3v5h5M10 13h6M10 17h6" />,
  },
];

const FEATURES: { title: string; body: string }[] = [
  {
    title: 'Learns from the best',
    body: 'The search follows a blend of its best answers so far, not just a single one.',
  },
  {
    title: 'Starts evenly',
    body: 'It begins by looking across the whole image evenly, so nothing is missed early on.',
  },
  {
    title: 'Adjusts itself',
    body: 'It tunes how widely it searches while it runs, based on how well it is doing.',
  },
];

export default function LandingPage({ onNavigate }: { onNavigate: Navigate }) {
  const onStart = () => onNavigate('analyzer');
  return (
    <div className="galaxy-bg min-h-screen font-['Figtree',sans-serif]" style={{ color: INK }}>
      <Navbar onNavigate={onNavigate} />

      {/* Hero — full width, fills the screen below the navbar */}
      <section className="nebula relative overflow-hidden min-h-[calc(100svh-4rem)] flex flex-col text-[#F0FDF4]">
        {/* Faint watermark, like a big brand word behind the hero */}
        <span
          aria-hidden="true"
          className={`pointer-events-none select-none absolute bottom-[3.5rem] left-1/2 -translate-x-1/2 text-[34vw] xl:text-[460px]
                      leading-[0.8] font-light tracking-tighter text-white/[0.06] whitespace-nowrap ${FONT_DISPLAY}`}
        >
          ESMA
        </span>

        <div className="relative z-10 flex-1 flex items-center w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-16">
          <div className="w-full grid grid-cols-1 lg:grid-cols-[1.2fr_0.8fr] gap-10 items-center">
            <div>
              <span className="inline-flex items-center gap-2 text-[12px] font-medium bg-white/10 border border-white/15 backdrop-blur px-3 py-1.5 rounded-full">
                <span className="w-1.5 h-1.5 rounded-full bg-[#34D399] shadow-[0_0_8px_#34D399]" />
                Research prototype
              </span>
              <h1 className={`mt-5 text-[34px] sm:text-[60px] lg:text-[68px] xl:text-[84px] font-light leading-[0.98] tracking-[-0.035em] ${FONT_DISPLAY}`}>
                Smarter Dental
                <br />
                <span className="bg-gradient-to-r from-[#34D399] to-[#BBF7D0] bg-clip-text text-transparent drop-shadow-[0_0_24px_rgba(52,211,153,0.35)]">Orthopantomogram</span>
                <br />
                Segmentation
              </h1>
              <p className="mt-6 text-[16px] leading-relaxed max-w-lg text-white/75">
                Upload a panoramic dental image (OPG) and see the areas that may need a closer look highlighted
                for you, with a simple report you can save.
              </p>
              <div className="mt-8 flex flex-wrap items-center gap-4">
                <button
                  onClick={onStart}
                  className="neu-btn min-h-[50px] px-6 rounded-lg bg-btn text-btn-fg text-[14.5px] font-semibold
                             inline-flex items-center gap-3 hover:bg-[var(--btn-hover)] transition-colors"
                >
                  Get started
                  <ArrowIcon />
                </button>
                <a href="#how-it-works" className="text-[14px] font-semibold underline underline-offset-4 decoration-1 hover:opacity-60">
                  How it works
                </a>
              </div>
            </div>

            <HeroVisual />
          </div>
        </div>

        {/* Stats strip pinned to the bottom of the hero */}
        <div className="relative z-10 border-t border-white/10 bg-black/20 backdrop-blur-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid grid-cols-1 sm:grid-cols-3">
            {STATS.map((s, i) => (
              <div
                key={s.label}
                className={`flex items-center gap-3 py-5 sm:py-6 px-2 sm:px-6 border-white/10 ${
                  i > 0 ? 'border-t sm:border-t-0 sm:border-l' : ''
                }`}
              >
                <span className={`text-[40px] sm:text-[52px] font-light leading-none tracking-tight ${FONT_DISPLAY}`}>
                  {s.value}
                </span>
                <span className="text-[13px] leading-tight max-w-[6rem] text-white/65">
                  {s.label}
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-14 sm:pt-20">
        {/* How it works — dark section */}
        <section id="how-it-works" className="scroll-mt-24 rounded-[28px] bg-[var(--how-bg)] border border-white/10 shadow-[var(--card-shadow)] text-white px-5 sm:px-8 py-10 sm:py-12">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pb-8 border-b border-white/10">
            <h2 className={`text-[30px] sm:text-[38px] font-light leading-[1.1] tracking-tight ${FONT_DISPLAY}`}>
              From OPG to report,
              <br />
              in four steps
            </h2>
            <p className="text-[13.5px] leading-relaxed text-white/65 md:max-w-sm md:justify-self-end self-end">
              No setup inside the app. Upload one panoramic image, click Analyze, and see the highlighted
              areas next to your original image.
            </p>
          </div>
          <ol className="mt-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {STEPS.map((step, i) => (
              <li key={step.title} className="rounded-2xl bg-[var(--how-card)] bg-gradient-to-b from-white/[0.07] to-transparent border border-white/10 shadow-[inset_0_1px_0_rgba(255,255,255,0.12),0_10px_24px_rgba(0,0,0,0.3)] p-2.5 flex flex-col">
                <div className="relative aspect-[16/10] rounded-xl bg-gradient-to-br from-[var(--how-art-from)] to-[var(--how-art-to)] shadow-[inset_0_2px_10px_rgba(0,0,0,0.35)] flex items-center justify-center">
                  <span className={`absolute top-3 left-3.5 text-[13px] text-white/50 ${FONT_DISPLAY}`}>0{i + 1}</span>
                  <span className="w-14 h-14 rounded-2xl bg-gradient-to-b from-white/20 to-white/5 border border-white/15 shadow-[inset_0_1px_0_rgba(255,255,255,0.25),0_6px_14px_rgba(0,0,0,0.3)] flex items-center justify-center text-[var(--how-icon)]">
                    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7"
                      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      {step.icon}
                    </svg>
                  </span>
                </div>
                <div className="px-2 pt-4 pb-3 border-t border-white/10 mt-2.5">
                  <h3 className="text-[15.5px] font-medium">{step.title}</h3>
                  <p className="mt-2 text-[12.5px] leading-relaxed text-white/55">{step.body}</p>
                </div>
              </li>
            ))}
          </ol>
        </section>

        {/* Algorithms */}
        <section id="algorithms" className="scroll-mt-24 grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-center py-14 sm:py-20">
          <div className="px-1 sm:px-4">
            <h2 className={`text-[30px] sm:text-[38px] font-light leading-[1.1] tracking-tight ${FONT_DISPLAY}`}>
              Inspired by nature.
              <br />
              Built for dental images.
            </h2>
            <p className="mt-4 text-[14px] leading-relaxed max-w-md" style={{ color: MUTED }}>
              The analysis uses a search method inspired by how slime mould finds food. It works out the best
              way to separate teeth, bone and soft tissue in your image, then a trained model marks the areas
              worth a closer look.
            </p>
            <div className="mt-6 flex flex-wrap gap-2">
              <Chip color="#0F766E">Slime mould–inspired search</Chip>
              <Chip color="#E39446">Trained detection model</Chip>
            </div>
          </div>

          <div className="rounded-[24px] bg-gradient-to-br from-hero-from to-hero-to border border-[var(--neu-edge)] shadow-[9px_9px_22px_var(--neu-dark),-9px_-9px_22px_var(--neu-light),inset_0_1px_0_rgba(255,255,255,0.45)] p-6 sm:p-8">
            <div className="flex items-end gap-3">
              <span className={`text-[64px] sm:text-[76px] font-light leading-none tracking-tight ${FONT_DISPLAY}`}>3</span>
              <span className="text-[14px] font-medium leading-tight pb-2">
                smart features
                <br />
                behind every analysis
              </span>
            </div>
            <ul className="mt-6 flex flex-col gap-2.5">
              {FEATURES.map((e) => (
                <li key={e.title} className="neu-sm rounded-xl px-4 py-3">
                  <p className="text-[14px] font-semibold">{e.title}</p>
                  <p className="text-[12.5px] leading-relaxed mt-0.5" style={{ color: MUTED }}>{e.body}</p>
                </li>
              ))}
            </ul>
          </div>
        </section>

        {/* Notes */}
        <section id="notes" className="scroll-mt-24 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="neu rounded-[24px] p-6">
            <h3 className={`text-[20px] font-normal tracking-tight ${FONT_DISPLAY}`}>Before you start</h3>
            <ul className="mt-3 space-y-2 text-[13px] leading-relaxed list-disc pl-4" style={{ color: MUTED }}>
              <li>Make sure the FastAPI backend is running — the analyzer sends the OPG there for processing.</li>
              <li>Use a full panoramic (OPG) image; cropped or intraoral photos give unreliable results.</li>
              <li>The analysis can take a little while on larger images.</li>
            </ul>
          </div>
          <div className="rounded-[24px] bg-warn-bg border border-warn-line shadow-[9px_9px_22px_var(--neu-dark),-9px_-9px_22px_var(--neu-light),inset_0_1px_0_rgba(255,255,255,0.45)] text-warn-ink p-6">
            <h3 className={`text-[20px] font-normal tracking-tight ${FONT_DISPLAY}`}>⚠ Not a medical diagnosis</h3>
            <p className="mt-3 text-[13px] leading-relaxed">
              This is a research prototype for academic demonstration only. Findings come from a trained
              detection model and threshold-based segmentation, not a validated diagnostic tool. Please consult
              a licensed dentist for any actual diagnosis or treatment.
            </p>
          </div>
        </section>

        {/* Final CTA */}
        <section className="my-14 sm:my-20 flex flex-col items-center text-center">
          <h2 className={`text-[30px] sm:text-[44px] font-light leading-[1.1] tracking-tight ${FONT_DISPLAY}`}>
            Ready to try it?
          </h2>
          <button
            onClick={onStart}
            className="neu-btn mt-6 min-h-[50px] px-6 rounded-lg bg-btn text-btn-fg text-[14.5px] font-semibold
                       inline-flex items-center gap-3 hover:bg-[var(--btn-hover)] transition-colors"
          >
            Get started
            <ArrowIcon />
          </button>
        </section>

        <footer className="border-t border-line-strong py-6 text-[12px] flex flex-wrap justify-between gap-2" style={{ color: MUTED }}>
          <span>ESMA Dental — slime mould–inspired dental image analysis</span>
          <span>Research prototype · not for clinical use</span>
        </footer>
      </div>
    </div>
  );
}

function Chip({ color, children }: { color: string; children: React.ReactNode }) {
  return (
    <span className="neu-sm inline-flex items-center gap-2 text-[12.5px] font-semibold px-3 py-1.5 rounded-full">
      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
      {children}
    </span>
  );
}

function ArrowIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M5 12h14M13 6l6 6-6 6" />
    </svg>
  );
}

// 4-point sparkle centered on (0, 0)
const sparkle = (r: number) =>
  `M0,${-r} L${r * 0.22},${-r * 0.22} L${r},0 L${r * 0.22},${r * 0.22} L0,${r} L${-r * 0.22},${r * 0.22} L${-r},0 L${-r * 0.22},${-r * 0.22}Z`;

// Orbits wrap around the tooth: the full ring is drawn behind it, and the
// near (lower) half is drawn again in front of it.
const OUTER = { cx: 200, cy: 210, rx: 185, ry: 50, tilt: 10 };
const INNER = { cx: 200, cy: 185, rx: 150, ry: 38, tilt: -14 };
type Orbit = typeof OUTER;

const ringPath = (o: Orbit) =>
  `M${o.cx - o.rx},${o.cy} a${o.rx},${o.ry} 0 1,0 ${2 * o.rx},0 a${o.rx},${o.ry} 0 1,0 ${-2 * o.rx},0`;
// lower arc only (sweep 0 goes through the bottom in SVG's y-down space)
const frontArcPath = (o: Orbit) => `M${o.cx - o.rx},${o.cy} A${o.rx},${o.ry} 0 0,0 ${o.cx + o.rx},${o.cy}`;
const tiltOf = (o: Orbit) => (o.tilt ? `rotate(${o.tilt} ${o.cx} ${o.cy})` : undefined);

// Glowing "search agents" that travel the orbits, a nod to the slime mould
// population. With reduced motion they sit still at angle `still` (degrees).
const AGENTS: { orbit: Orbit; r: number; color: string; dur: number; begin: number; still: number }[] = [
  { orbit: OUTER, r: 5, color: '#34D399', dur: 14, begin: 0, still: 160 },
  { orbit: OUTER, r: 4.5, color: '#22D3EE', dur: 14, begin: -4.7, still: 60 },
  { orbit: OUTER, r: 5, color: '#E39446', dur: 14, begin: -9.3, still: 330 },
  { orbit: INNER, r: 4, color: '#2DD4BF', dur: 10, begin: -2, still: 120 },
  { orbit: INNER, r: 4, color: '#E39446', dur: 10, begin: -7, still: 250 },
];

const SPARKLES: { x: number; y: number; r: number; delay: string }[] = [
  { x: 70, y: 70, r: 7, delay: '0s' },
  { x: 340, y: 95, r: 9, delay: '0.8s' },
  { x: 365, y: 300, r: 6, delay: '1.6s' },
  { x: 45, y: 300, r: 5, delay: '2.2s' },
  { x: 290, y: 40, r: 4, delay: '1.2s' },
];

// Glassy tooth wrapped by two orbit rings with travelling agents, plus a glow
// halo and sparkles, standing in for the 3D render in the reference design.
function HeroVisual() {
  const reduceMotion =
    typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // front = only the part of each agent's path that passes in front of the tooth
  const agents = (front: boolean) =>
    AGENTS.map((a, i) => {
      const o = a.orbit;
      const dot = reduceMotion ? (
        <circle
          cx={o.cx + o.rx * Math.cos((a.still * Math.PI) / 180)}
          cy={o.cy + o.ry * Math.sin((a.still * Math.PI) / 180)}
          r={a.r}
          fill={a.color}
          filter="url(#agent-glow)"
        />
      ) : (
        <circle r={a.r} fill={a.color} filter="url(#agent-glow)">
          <animateMotion dur={`${a.dur}s`} begin={`${a.begin}s`} repeatCount="indefinite" path={ringPath(o)} />
        </circle>
      );
      return (
        <g key={i} transform={tiltOf(o)}>
          <g clipPath={front ? `url(#front-${o === OUTER ? 'outer' : 'inner'})` : undefined}>{dot}</g>
        </g>
      );
    });

  return (
    <div className="relative min-h-[320px] sm:min-h-[440px] flex items-center justify-center" aria-hidden="true">
      {/* back layer: halo, far side of the rings, all agents, sparkles */}
      <svg viewBox="0 0 400 360" className="absolute inset-0 w-full h-full overflow-visible">
        <defs>
          <radialGradient id="hero-halo">
            <stop offset="0" stopColor="#6EE7B7" stopOpacity="0.55" />
            <stop offset="0.45" stopColor="#34D399" stopOpacity="0.18" />
            <stop offset="1" stopColor="#34D399" stopOpacity="0" />
          </radialGradient>
          <filter id="agent-glow" x="-200%" y="-200%" width="500%" height="500%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <circle cx="200" cy="175" r="165" fill="url(#hero-halo)" />

        <g transform={tiltOf(OUTER)}>
          <path d={ringPath(OUTER)} fill="none" stroke="#34D399" strokeOpacity="0.45" strokeWidth="1.6" />
        </g>
        <g transform={tiltOf(INNER)}>
          <path d={ringPath(INNER)} fill="none" stroke="#2DD4BF" strokeOpacity="0.4" strokeWidth="1.3" strokeDasharray="4 6" />
        </g>

        {agents(false)}

        {SPARKLES.map((sp, i) => (
          <path
            key={i}
            d={sparkle(sp.r)}
            transform={`translate(${sp.x} ${sp.y})`}
            fill="#ECFDF5"
            className="motion-safe:animate-pulse"
            style={{ animationDelay: sp.delay, animationDuration: '3s' }}
          />
        ))}
      </svg>

      <svg viewBox="0 0 24 24" className="relative w-[230px] sm:w-[300px] lg:w-[340px] drop-shadow-[0_30px_40px_rgba(15,90,80,0.3)]">
        <defs>
          <linearGradient id="tooth-fill" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#FFFFFF" />
            <stop offset="0.55" stopColor="#EEFBF3" />
            <stop offset="1" stopColor="#A7E3C6" />
          </linearGradient>
          <linearGradient id="tooth-shine" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#FFFFFF" stopOpacity="0.95" />
            <stop offset="1" stopColor="#FFFFFF" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path
          d="M7 3c-2.5 0-4 2-4 4.5 0 3 1.5 4.5 2 7.5.4 2.6 1 6 2.6 6 1.4 0 1.6-3 2.4-5 .4-1 1.6-1 2 0 .8 2 1 5 2.4 5 1.6 0 2.2-3.4 2.6-6 .5-3 2-4.5 2-7.5C21 5 19.5 3 17 3c-2 0-3 1-5 1S9 3 7 3z"
          fill="url(#tooth-fill)"
          stroke="#FFFFFF"
          strokeWidth="0.25"
        />
        <path d="M6.2 4.6c-1.4.4-2 1.7-2 3 0 .9.3 1.7.6 2.4.5-2.2 1.6-4 3.3-5-.6-.4-1.2-.5-1.9-.4z" fill="url(#tooth-shine)" />
      </svg>

      {/* front layer: near half of each ring and the agents passing in front of the tooth */}
      <svg viewBox="0 0 400 360" className="absolute inset-0 w-full h-full overflow-visible pointer-events-none">
        <defs>
          {/* clip to below each ring's center line, in the ring's own (tilted) space */}
          <clipPath id="front-outer" clipPathUnits="userSpaceOnUse">
            <rect x="-100" y={OUTER.cy} width="600" height="400" />
          </clipPath>
          <clipPath id="front-inner" clipPathUnits="userSpaceOnUse">
            <rect x="-100" y={INNER.cy} width="600" height="400" />
          </clipPath>
        </defs>

        <g transform={tiltOf(OUTER)}>
          <path d={frontArcPath(OUTER)} fill="none" stroke="#10B981" strokeWidth="2.2" filter="url(#agent-glow)" />
        </g>
        <g transform={tiltOf(INNER)}>
          <path d={frontArcPath(INNER)} fill="none" stroke="#14B8A6" strokeWidth="1.8" strokeDasharray="4 6" filter="url(#agent-glow)" />
        </g>

        {agents(true)}
      </svg>
    </div>
  );
}

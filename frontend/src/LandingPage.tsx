import React, { useEffect, useState } from 'react';
import Navbar, { type Navigate } from './Navbar';
import Footer from './Footer';

const FONT_DISPLAY = "font-['IBM_Plex_Sans',sans-serif]";
const INK = 'var(--ink)';
const MUTED = 'var(--muted)';

// Headline KPIs. Images = DENTEX (705 OPGs) + Mendeley OPG (100 OPGs).
// Accuracy and the enhanced-vs-standard count come from
// backend/dentex_report.txt (300 DENTEX images x 20 runs): ESMA hits the
// optimal thresholds 93.1% of the time (Standard SMA 1.5%) and beats
// Standard SMA on 300/300 images (0 ties, 0 losses).
const DATASETS = [
  { name: 'DENTEX', count: 705, color: '#34D399' },
  { name: 'Mendeley', count: 100, color: '#F0FDF4' },
];

const KPIS: { value: string; label: string; note: React.ReactNode; visual: React.ReactNode }[] = [
  {
    value: '805',
    label: 'OPG images',
    note: (
      <span className="inline-flex flex-wrap gap-x-3">
        {DATASETS.map((d) => (
          <span key={d.name} className="inline-flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: d.color }} />
            {d.name} {d.count}
          </span>
        ))}
      </span>
    ),
    visual: <Pie slices={DATASETS.map((d) => ({ value: d.count, color: d.color }))} />,
  },
  {
    value: '93.1%',
    label: 'Threshold accuracy',
    note: 'vs 1.5% for Standard SMA',
    visual: <Ring pct={93.1} />,
  },
  {
    value: '300/300',
    label: 'Images where ESMA beat SMA',
    note: '0 ties · 0 losses',
    visual: <DotGrid filled={30} total={30} />,
  },
];

type Step = { title: string; body: string; details: string[]; icon: React.ReactNode };

const STEPS: Step[] = [
  {
    title: 'Upload your OPG',
    body: 'Drag your panoramic dental image (OPG) into the analyzer or click "Choose an image". PNG and JPEG work.',
    details: [
      "Click \"Get started\" to open the analyzer, then drag your image onto the upload area or click \"Choose an image\".",
      "Only PNG and JPEG files are accepted.",
      "Use a full panoramic X-ray (OPG). Cropped or intraoral photos give unreliable results.",
      "A preview and the image size appear so you can check the file. You can still pick a different image before analyzing.",
    ],
    icon: <path d="M12 16V4M7 9l5-5 5 5M5 20h14" />,
  },
  {
    title: 'Click Analyze',
    body: 'Press "Analyze OPG" and the system checks your image. It usually takes under a minute.',
    details: [
      "Press \"Analyze OPG\". The image is sent to the FastAPI backend, so the backend must be running.",
      "First, the Enhanced Slime Mould Algorithm (ESMA) finds the best thresholds to separate teeth, bone and soft tissue.",
      "Then a trained detection model (YOLOv8) marks the areas that may need a closer look.",
      "It usually takes under a minute; larger images can take a little longer.",
    ],
    icon: <path d="M7 4.5v15l12-7.5z" />,
  },
  {
    title: 'See what was found',
    body: 'The highlighted areas appear next to a list of what was found, with each finding explained in plain words.',
    details: [
      "The highlighted image appears with \"What was found\" beside it. You can hide that panel to see the image larger.",
      "Colored boxes mark four kinds of findings: Caries, Deep Caries, Impacted and Periapical Lesion.",
      "Findings are grouped by type, with their location (from the patient's point of view) and a confidence score.",
      "Click the small original OPG to explore it with a zoom lens that reveals the findings, or expand the image to full screen.",
    ],
    icon: <path d="M4 5h7v14H4zM13 5h7v14h-7z" />,
  },
  {
    title: 'Save your report',
    body: 'Download a PDF with your image, the highlighted areas and the list of findings.',
    details: [
      "Click \"Download report\" to save everything as a PDF.",
      "The PDF includes your image, the highlighted areas, the image details and the list of findings with confidence.",
      "You can also download the highlighted image on its own as a PNG.",
      "The report is for reference only. It is not a medical diagnosis.",
    ],
    icon: <path d="M7 3h7l5 5v13H7zM14 3v5h5M10 13h6M10 17h6" />,
  },
];

export default function LandingPage({ onNavigate }: { onNavigate: Navigate }) {
  const onStart = () => onNavigate('analyzer');
  const [openStep, setOpenStep] = useState<number | null>(null);
  return (
    <div className="galaxy-bg min-h-screen font-['IBM_Plex_Sans',sans-serif]" style={{ color: INK }}>
      <Navbar onNavigate={onNavigate} />

      {/* Hero — full width, fills the screen below the navbar */}
      <section className="nebula relative overflow-hidden min-h-[calc(100svh-4rem)] flex flex-col text-[#F0FDF4]">
        <div className="relative z-10 flex-1 flex items-center w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-16">
          <div className="w-full grid grid-cols-1 lg:grid-cols-[1.2fr_0.8fr] gap-10 items-center">
            <div>
              <h1 className={`text-[34px] sm:text-[60px] lg:text-[68px] xl:text-[84px] font-bold leading-[0.98] tracking-[-0.035em] ${FONT_DISPLAY}`}>
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
                  className="neu-btn min-h-[50px] px-6 rounded-lg bg-btn text-btn-fg text-[16px] font-semibold
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
            {KPIS.map((k, i) => (
              <div
                key={k.label}
                className={`flex items-center gap-4 sm:gap-5 py-5 sm:py-6 px-2 sm:px-6 border-white/10 ${
                  i > 0 ? 'border-t sm:border-t-0 sm:border-l' : ''
                }`}
              >
                <div className="shrink-0 w-14 h-14 sm:w-16 sm:h-16 grid place-items-center">{k.visual}</div>
                <div className="min-w-0">
                  <div className={`text-[34px] sm:text-[38px] lg:text-[44px] font-bold leading-none tracking-tight ${FONT_DISPLAY}`}>
                    {k.value}
                  </div>
                  <div className="mt-1.5 text-[13px] font-medium text-white/85">{k.label}</div>
                  <div className="mt-0.5 text-[12px] text-white/50">{k.note}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-14 sm:pt-20">
        {/* How it works — dark section */}
        <section id="how-it-works" className="scroll-mt-24 rounded-[28px] bg-[var(--how-bg)] border border-white/10 shadow-[var(--card-shadow)] text-white px-5 sm:px-8 py-10 sm:py-12">
          <div className="pb-8 border-b border-white/10">
            <h2 className={`text-[30px] sm:text-[38px] font-bold leading-[1.1] tracking-tight ${FONT_DISPLAY}`}>
              How ESMA Dental Analyzes Your Dental X-Ray
            </h2>
          </div>
          <ol className="mt-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {STEPS.map((step, i) => (
              <li key={step.title}>
                <button
                  type="button"
                  onClick={() => setOpenStep(i)}
                  aria-haspopup="dialog"
                  className="group relative w-full aspect-[16/10] sm:aspect-[5/4] lg:aspect-[4/5] overflow-hidden rounded-[24px] lg:rounded-[28px] text-left border border-white/10
                             shadow-[inset_0_1px_0_rgba(255,255,255,0.12),0_14px_30px_rgba(0,0,0,0.35)]
                             transition-[transform,box-shadow] duration-300 ease-out hover:-translate-y-1
                             hover:shadow-[inset_0_1px_0_rgba(255,255,255,0.16),0_22px_40px_rgba(0,0,0,0.45)]
                             focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#34D399]"
                >
                  <div aria-hidden="true" className="absolute inset-0 bg-gradient-to-b from-[var(--how-art-from)] to-[var(--how-art-to)]" />

                  {/* Frosted fade at the bottom, like a photo card caption */}
                  <div
                    aria-hidden="true"
                    className="absolute inset-x-0 bottom-0 h-[38%] backdrop-blur-md bg-gradient-to-t from-black/55 via-black/25 to-transparent"
                    style={{
                      maskImage: 'linear-gradient(to top, black 45%, transparent)',
                      WebkitMaskImage: 'linear-gradient(to top, black 45%, transparent)',
                    }}
                  />

                  {/* Art: the glow and icon zoom in on hover; the label and title stay put */}
                  <div
                    aria-hidden="true"
                    className="absolute inset-0 sm:bottom-auto sm:h-[70%] flex items-center justify-end pr-6 sm:justify-center sm:pr-0
                               transition-transform duration-500 ease-out group-hover:scale-[1.18] group-focus-visible:scale-[1.18]"
                  >
                    <div className="absolute inset-0 bg-[radial-gradient(circle_at_80%_50%,rgba(52,211,153,0.32),transparent_45%)] sm:bg-[radial-gradient(circle_at_50%_55%,rgba(52,211,153,0.32),transparent_55%)]" />
                    <span
                      className="relative sm:mt-8 w-16 h-16 sm:w-20 sm:h-20 rounded-[18px] sm:rounded-[22px] bg-gradient-to-b from-white/25 to-white/5 border border-white/20
                                 shadow-[inset_0_1px_0_rgba(255,255,255,0.3),0_10px_24px_rgba(0,0,0,0.3)]
                                 flex items-center justify-center text-[var(--how-icon)]"
                    >
                      <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"
                        strokeLinecap="round" strokeLinejoin="round">
                        {step.icon}
                      </svg>
                    </span>
                  </div>

                  <span className="absolute top-5 inset-x-0 px-5 text-left sm:px-0 sm:text-center text-[17px] font-bold tracking-wide text-white/85">
                    Step {String(i + 1).padStart(2, '0')}
                  </span>

                  <h3 className="absolute inset-x-0 bottom-0 px-5 pb-5 text-[18px] font-bold leading-tight text-white">
                    {step.title}
                  </h3>
                </button>
              </li>
            ))}
          </ol>
        </section>

        {openStep !== null && (
          <StepModal index={openStep} onChange={setOpenStep} onClose={() => setOpenStep(null)} />
        )}

        {/* Algorithms */}
        <section id="algorithms" className="scroll-mt-24 grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-center py-14 sm:py-20">
          <div className="px-1 sm:px-4">
            <h2 className={`text-[30px] sm:text-[38px] font-bold leading-[1.1] tracking-tight ${FONT_DISPLAY}`}>
              Inspired by nature.
              <br />
              Built for dental images.
            </h2>
            <p className="mt-4 text-[15px] leading-relaxed max-w-md" style={{ color: MUTED }}>
              The analysis uses a search method inspired by how slime mould finds food. It works out the best
              way to separate teeth, bone and soft tissue in your image, then a trained model marks the areas
              worth a closer look.
            </p>
            <div className="mt-6 flex flex-wrap gap-2">
              <Chip color="#0F766E">Slime mould–inspired search</Chip>
              <Chip color="#E39446">Trained detection model</Chip>
            </div>
          </div>

          <div className="rounded-[24px] bg-gradient-to-br from-hero-from to-hero-to border border-[var(--neu-edge)] shadow-[9px_9px_22px_var(--neu-dark),-9px_-9px_22px_var(--neu-light),inset_0_1px_0_rgba(255,255,255,0.45)] p-8 sm:p-10
                          min-h-[240px] sm:min-h-[280px] flex items-center justify-center">
            <div className="flex items-center gap-5 sm:gap-6">
              <span
                className={`text-[96px] sm:text-[128px] font-bold leading-none tracking-tighter bg-gradient-to-b from-[var(--hl-from)] to-[var(--hl-to)]
                            bg-clip-text text-transparent ${FONT_DISPLAY}`}
              >
                3
              </span>
              <span className="w-px self-stretch my-3 bg-[var(--line-strong)] opacity-60" aria-hidden="true" />
              <span className={`text-[22px] sm:text-[28px] font-bold leading-[1.15] tracking-tight ${FONT_DISPLAY}`}>
                smart features
                <br />
                <span className="font-normal" style={{ color: MUTED }}>behind every analysis</span>
              </span>
            </div>
          </div>
        </section>

        {/* Final CTA */}
        <section className="my-14 sm:my-20 flex flex-col items-center text-center">
          <h2 className={`text-[30px] sm:text-[44px] font-bold leading-[1.1] tracking-tight ${FONT_DISPLAY}`}>
            Ready to try it?
          </h2>
          <button
            onClick={onStart}
            className="neu-btn mt-6 min-h-[50px] px-6 rounded-lg bg-btn text-btn-fg text-[16px] font-semibold
                       inline-flex items-center gap-3 hover:bg-[var(--btn-hover)] transition-colors"
          >
            Get started
            <ArrowIcon />
          </button>
        </section>
      </div>

      <Footer />
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

// The gradient tile with the step number and icon; shared by the cards and the modal.
function StepArt({ step, index, className = '' }: { step: Step; index: number; className?: string }) {
  return (
    <div className={`absolute inset-0 bg-gradient-to-br from-[var(--how-art-from)] to-[var(--how-art-to)] flex items-center justify-center ${className}`}>
      <span className={`absolute top-3 left-3.5 text-[13px] text-white/50 ${FONT_DISPLAY}`}>0{index + 1}</span>
      <span className="w-14 h-14 rounded-2xl bg-gradient-to-b from-white/20 to-white/5 border border-white/15 shadow-[inset_0_1px_0_rgba(255,255,255,0.25),0_6px_14px_rgba(0,0,0,0.3)] flex items-center justify-center text-[var(--how-icon)]">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7"
          strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          {step.icon}
        </svg>
      </span>
    </div>
  );
}

function StepModal({ index, onChange, onClose }: { index: number; onChange: (i: number) => void; onClose: () => void }) {
  const step = STEPS[index];
  const last = STEPS.length - 1;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === 'ArrowRight' && index < last) onChange(index + 1);
      if (e.key === 'ArrowLeft' && index > 0) onChange(index - 1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [index, last, onChange, onClose]);

  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = prev;
    };
  }, []);

  return (
    <div className="fixed inset-0 z-[60] bg-black/60 backdrop-blur-sm flex items-center justify-center p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="step-title"
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-lg max-h-[90vh] overflow-y-auto rounded-[24px] bg-[var(--how-bg)] border border-white/10 text-white
                   shadow-[0_24px_60px_rgba(0,0,0,0.45)] p-2.5"
      >
        <div className="relative aspect-[16/7] rounded-[18px] overflow-hidden">
          <StepArt step={step} index={index} />
          <button
            type="button"
            onClick={onClose}
            autoFocus
            aria-label="Close"
            className="absolute top-2.5 right-2.5 w-9 h-9 rounded-full bg-black/30 hover:bg-black/50 text-white/80 hover:text-white flex items-center justify-center"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2"
              strokeLinecap="round" aria-hidden="true">
              <path d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        </div>

        <div className="px-4 sm:px-5 pt-5 pb-4">
          <p className="text-[12px] font-semibold tracking-wider uppercase text-[#34D399]">
            Step {index + 1} of {STEPS.length}
          </p>
          <h3 id="step-title" className={`mt-1 text-[24px] font-bold tracking-tight ${FONT_DISPLAY}`}>{step.title}</h3>
          <p className="mt-2 text-[15px] leading-relaxed text-white/90">{step.body}</p>
          <ul className="mt-4 space-y-3">
            {step.details.map((d) => (
              <li key={d} className="flex gap-3 text-[15px] leading-relaxed text-white/75">
                <span className="mt-[9px] w-1.5 h-1.5 rounded-full bg-[#34D399] shrink-0" aria-hidden="true" />
                {d}
              </li>
            ))}
          </ul>

          <div className="mt-6 pt-4 border-t border-white/10 flex items-center justify-between gap-3">
            <button
              type="button"
              onClick={() => onChange(index - 1)}
              disabled={index === 0}
              className="min-h-[44px] px-4 rounded-lg text-[16px] font-semibold text-white/80 hover:text-white hover:bg-white/10 disabled:opacity-30 disabled:hover:bg-transparent
                         inline-flex items-center gap-2"
            >
              <span className="rotate-180 flex"><ArrowIcon /></span>
              Previous
            </button>
            <button
              type="button"
              onClick={index < last ? () => onChange(index + 1) : onClose}
              className="neu-btn min-h-[44px] px-5 rounded-lg bg-btn text-btn-fg text-[16px] font-semibold hover:bg-[var(--btn-hover)] transition-colors
                         inline-flex items-center gap-2"
            >
              {index < last ? (
                <>
                  Next step
                  <ArrowIcon />
                </>
              ) : (
                'Done'
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
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

// --- KPI mini visuals (decorative; the numbers beside them carry the meaning) ---

const KPI_GREEN = '#34D399';
const KPI_TRACK = 'rgba(255,255,255,0.12)';

function Ring({ pct }: { pct: number }) {
  const r = 26;
  const c = 2 * Math.PI * r;
  return (
    <svg viewBox="0 0 64 64" className="w-full h-full -rotate-90" aria-hidden="true">
      <circle cx="32" cy="32" r={r} fill="none" stroke={KPI_TRACK} strokeWidth="7" />
      <circle
        cx="32" cy="32" r={r} fill="none" stroke={KPI_GREEN} strokeWidth="7" strokeLinecap="round"
        strokeDasharray={`${(pct / 100) * c} ${c}`}
      />
    </svg>
  );
}

function Pie({ slices }: { slices: { value: number; color: string }[] }) {
  const total = slices.reduce((t, sl) => t + sl.value, 0);
  const point = (frac: number) => {
    const a = frac * 2 * Math.PI - Math.PI / 2; // start at 12 o'clock
    return `${32 + 28 * Math.cos(a)} ${32 + 28 * Math.sin(a)}`;
  };
  // where each slice starts, as a fraction of the circle
  const starts = slices.map((_, i) => slices.slice(0, i).reduce((t, sl) => t + sl.value, 0) / total);
  return (
    <svg viewBox="0 0 64 64" className="w-full h-full" aria-hidden="true">
      {slices.map((sl, i) => {
        const frac = sl.value / total;
        const d = `M32 32 L${point(starts[i])} A28 28 0 ${frac > 0.5 ? 1 : 0} 1 ${point(starts[i] + frac)} Z`;
        return <path key={i} d={d} fill={sl.color} stroke="rgba(0,0,0,0.35)" strokeWidth="1.5" strokeLinejoin="round" />;
      })}
    </svg>
  );
}

function DotGrid({ filled, total }: { filled: number; total: number }) {
  const cols = 6;
  return (
    <svg viewBox="0 0 64 64" className="w-full h-full" aria-hidden="true">
      {Array.from({ length: total }, (_, i) => (
        <rect
          key={i} x={3 + (i % cols) * 10} y={8 + Math.floor(i / cols) * 10} width="8" height="8" rx="2"
          fill={i < filled ? KPI_GREEN : KPI_TRACK}
        />
      ))}
    </svg>
  );
}

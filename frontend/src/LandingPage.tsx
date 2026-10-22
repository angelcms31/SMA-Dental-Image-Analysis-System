import React from 'react';
import Navbar, { type Navigate } from './Navbar';

const FONT_DISPLAY = "font-['Inter_Tight',sans-serif]";
const INK = 'var(--ink)';
const MUTED = 'var(--muted)';

// Real run settings the analyzer sends to the backend (see OpgAnalyzer runAlgorithm).
const STATS: { value: string; label: string }[] = [
  { value: '2', label: 'Algorithms compared' },
  { value: '4', label: 'Thresholds per image' },
  { value: '30', label: 'Search agents' },
  { value: '100', label: 'Max iterations' },
];

const STEPS: { title: string; body: string; icon: React.ReactNode }[] = [
  {
    title: 'Upload your OPG',
    body: 'Drag an orthopantomogram (OPG) into the analyzer or click "Choose OPG". PNG and JPEG are supported.',
    icon: <path d="M12 16V4M7 9l5-5 5 5M5 20h14" />,
  },
  {
    title: 'Run the algorithms',
    body: 'Press Standard or ESMA to run one algorithm, or "Run both" to process them at the same time.',
    icon: <path d="M7 4.5v15l12-7.5z" />,
  },
  {
    title: 'Compare side by side',
    body: 'The Standard SMA and ESMA results appear next to each other. Hover any image and the crosshair marks the same spot on all of them.',
    icon: <path d="M4 5h7v14H4zM13 5h7v14h-7z" />,
  },
  {
    title: 'Review the report',
    body: 'Switch between Findings and Metrics, check which algorithm scored better, and download the PDF report.',
    icon: <path d="M7 3h7l5 5v13H7zM14 3v5h5M10 13h6M10 17h6" />,
  },
];

const ENHANCEMENTS: { title: string; body: string }[] = [
  {
    title: 'Multi-leader guidance',
    body: 'Agents follow a fitness-weighted blend of the top leaders instead of a single best solution.',
  },
  {
    title: 'Quasi-uniform initialization',
    body: 'The starting population is spread evenly across the search space instead of placed at random.',
  },
  {
    title: 'Adaptive feedback control',
    body: 'Exploration settings adjust during the run based on convergence rate and population diversity.',
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
                Compare the Standard Slime Mould Algorithm against the Enhanced SMA (ESMA) on orthopantomograms (OPG),
                with segmentation, flagged findings and metrics side by side.
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
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid grid-cols-2 lg:grid-cols-4">
            {STATS.map((s, i) => (
              <div
                key={s.label}
                className={`flex items-center gap-3 py-5 sm:py-6 px-2 sm:px-6 border-white/10 ${i % 2 === 1 ? 'border-l' : ''} ${
                  i === 2 ? 'lg:border-l' : ''
                } ${i < 2 ? 'border-b lg:border-b-0' : ''}`}
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
              No setup inside the app. Upload one panoramic image, run the algorithms and the results land
              in two screens you can compare at a glance.
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
              Two algorithms. One OPG.
              <br />
              Side by side.
            </h2>
            <p className="mt-4 text-[14px] leading-relaxed max-w-md" style={{ color: MUTED }}>
              Both search for the multilevel thresholds that maximize Kapur's entropy. The Standard SMA follows
              the original slime mould algorithm; ESMA adds three enhancements to search more effectively.
            </p>
            <div className="mt-6 flex flex-wrap gap-2">
              <Chip color="#0F766E">Standard SMA</Chip>
              <Chip color="#E39446">Enhanced SMA (ESMA)</Chip>
            </div>
          </div>

          <div className="rounded-[24px] bg-gradient-to-br from-hero-from to-hero-to border border-[var(--neu-edge)] shadow-[9px_9px_22px_var(--neu-dark),-9px_-9px_22px_var(--neu-light),inset_0_1px_0_rgba(255,255,255,0.45)] p-6 sm:p-8">
            <div className="flex items-end gap-3">
              <span className={`text-[64px] sm:text-[76px] font-light leading-none tracking-tight ${FONT_DISPLAY}`}>3</span>
              <span className="text-[14px] font-medium leading-tight pb-2">
                enhancements in ESMA
                <br />
                over the Standard SMA
              </span>
            </div>
            <ul className="mt-6 flex flex-col gap-2.5">
              {ENHANCEMENTS.map((e) => (
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
              <li>Running both algorithms can take a little while on larger images.</li>
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
            Ready to compare?
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
          <span>ESMA Dental — Standard vs Enhanced Slime Mould Algorithm</span>
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

// Glassy tooth with an orbit ring and two floating cards for the algorithms,
// standing in for the 3D render in the reference design.
function HeroVisual() {
  return (
    <div className="relative min-h-[320px] sm:min-h-[440px] flex items-center justify-center" aria-hidden="true">
      <svg viewBox="0 0 400 360" className="absolute inset-0 w-full h-full">
        <ellipse cx="200" cy="215" rx="185" ry="48" fill="none" stroke="white" strokeOpacity="0.6" strokeWidth="1.2" />
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

      <FloatCard className="left-0 sm:left-2 top-[18%]" color="#0F766E" title="Standard SMA" sub="Single leader search" />
      <FloatCard className="right-0 sm:right-2 bottom-[16%]" color="#E39446" title="ESMA" sub="Multi-leader + adaptive" />
    </div>
  );
}

function FloatCard({ className, color, title, sub }: { className: string; color: string; title: string; sub: string }) {
  return (
    <div className={`absolute ${className} rounded-xl bg-glass backdrop-blur px-3 py-2 border border-[var(--neu-edge)] shadow-[inset_0_1px_0_rgba(255,255,255,0.5),0_12px_26px_rgba(15,90,80,0.2)]`}>
      <p className="flex items-center gap-1.5 text-[12px] font-semibold">
        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
        {title}
      </p>
      <p className="text-[10.5px] text-muted mt-0.5">{sub}</p>
    </div>
  );
}

import React, { useState, type ChangeEvent, type DragEvent } from 'react';
import axios from 'axios';
import { API_BASE } from './api';
import Navbar, { type Navigate } from './Navbar';

interface DetectedRegion {
  quadrant: 'Q1' | 'Q2' | 'Q3' | 'Q4';
  label: string;
  bbox: [number, number, number, number];
  area_px?: number;        // only from the SMA heuristic; YOLO regions omit it
  mean_intensity?: number;
  confidence?: number;
}

interface AnalyzeResult {
  status: 'success' | 'error';
  message?: string;
  algorithm?: string;
  thresholds?: number[];
  kapur_entropy_fitness?: number;
  psnr?: number | null;
  ssim?: number;
  runtime_sec?: number;
  iterations_used?: number;
  convergence_curve?: number[];
  segmented_image?: string;
  annotated_image?: string;
  detected_regions?: DetectedRegion[];
  quadrant_summary?: Record<'Q1' | 'Q2' | 'Q3' | 'Q4', number>;
  disclaimer?: string;
}

type CrossPos = { x: number; y: number } | null;

// FDI quadrants named from the PATIENT's side (Q1 shows on the image's
// left because the patient faces the viewer -- note the "R" marker).
const QUADRANT_LABELS: Record<string, string> = {
  Q1: 'Upper Right',
  Q2: 'Upper Left',
  Q3: 'Lower Left',
  Q4: 'Lower Right',
};

// Same hues the backend draws its YOLO boxes with (predict_yolo.py
// DIAGNOSIS_COLORS, converted from BGR), so chips match the boxes.
const DIAGNOSIS_ACCENT: Record<string, string> = {
  Caries: '#E8A33D',
  'Deep Caries': '#E8604C',
  Impacted: '#8B5FBF',
  'Periapical Lesion': '#4C8FE8',
};
const DIAGNOSIS_DEFAULT_ACCENT = '#B4B4B4';

const DIAGNOSIS_INFO: Record<string, string> = {
  Caries: 'Tooth decay (cavity) affecting the enamel or dentin.',
  'Deep Caries': 'Decay extending close to or into the pulp -- more advanced than Caries.',
  Impacted: 'A tooth that has not fully erupted, often blocked by another tooth or bone.',
  'Periapical Lesion': 'Infection or inflammation at the root tip, usually from advanced decay or trauma.',
};

const FONT_HEAD = "font-['Inter_Tight',sans-serif]";
const FONT_DISPLAY = "font-['Inter_Tight',sans-serif]";
const FONT_MONO = "font-['IBM_Plex_Mono',monospace] tabular-nums";

// The analyzer shows a single result from the enhanced pipeline (ESMA
// segmentation + the YOLO model trained on it). The Standard SMA comparison
// was removed from the UI; see git history to bring it back.
export default function OpgAnalyzer({ onNavigate }: { onNavigate: Navigate }) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [originalPreview, setOriginalPreview] = useState<string | null>(null);
  const [imageDims, setImageDims] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [statusText, setStatusText] = useState<string>('');
  const [fullscreenImage, setFullscreenImage] = useState<string | null>(null);
  const [crosshair, setCrosshair] = useState<CrossPos>(null);
  const [isExporting, setIsExporting] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  const done = result?.status === 'success';
  const regions = done ? result.detected_regions ?? [] : [];
  const resultImage = done ? result.annotated_image ?? null : null;

  const loadFile = (file: File) => {
    if (!/^image\/(png|jpeg)$/.test(file.type)) {
      setError('That file is not supported. Please choose a PNG or JPEG image of your OPG.');
      return;
    }
    setError(null);
    const url = URL.createObjectURL(file);
    setSelectedFile(file);
    setOriginalPreview(url);
    setImageDims(null);
    setResult(null);
    setStatusText('');
    const probe = new Image();
    probe.onload = () => setImageDims(`${probe.naturalWidth} × ${probe.naturalHeight}`);
    probe.src = url;
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) loadFile(e.target.files[0]);
    e.target.value = '';
  };

  const handleDrop = (e: DragEvent<HTMLElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (isProcessing) return;
    const file = e.dataTransfer.files[0];
    if (file) loadFile(file);
  };

  // Segmentation metrics and the detector run in parallel; the detector's
  // annotated image and regions win when it succeeds.
  const analyze = async (file: File): Promise<AnalyzeResult> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('d', '4');
    formData.append('N', '30');
    formData.append('T', '100');

    const yoloFormData = new FormData();
    yoloFormData.append('file', file);
    yoloFormData.append('conf', '0.25');
    yoloFormData.append('iou', '0.35');
    yoloFormData.append('use_esma', 'true');
    yoloFormData.append('sma_algorithm', 'enhanced_v2');
    yoloFormData.append('weights_path', 'runs_yolo/train_v2_mendeley/weights/best.pt');

    const [metricsRes, yoloRes] = await Promise.all([
      axios.post<AnalyzeResult>(`${API_BASE}/analyze/enhanced/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      }),
      axios.post<AnalyzeResult>(`${API_BASE}/analyze/detect-teeth/`, yoloFormData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      }),
    ]);

    return {
      ...metricsRes.data,
      annotated_image: yoloRes.data.status === 'success' ? yoloRes.data.annotated_image : metricsRes.data.annotated_image,
      detected_regions: yoloRes.data.status === 'success' ? yoloRes.data.detected_regions : metricsRes.data.detected_regions,
    };
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setError(null);
    setIsProcessing(true);
    setStatusText('Analyzing your OPG… this can take a minute.');
    try {
      const res = await analyze(selectedFile);
      setResult(res);
      if (res.status === 'success') {
        setStatusText('Analysis finished. Your results are ready.');
        window.scrollTo({ top: 0, behavior: 'smooth' });
      } else {
        setStatusText('');
        setError(res.message ?? "We couldn't analyze this image. Please try a different one.");
      }
    } catch (err) {
      console.error('Error analyzing image:', err);
      setStatusText('');
      setError(
        "We couldn't finish the analysis because the analysis server didn't respond. " +
          'Please check that the backend is running, then try again.'
      );
    } finally {
      setIsProcessing(false);
    }
  };

  const fileBase = selectedFile ? selectedFile.name.replace(/\.[^.]+$/, '') : 'opg';

  // Start over with a new image.
  const reset = () => {
    setSelectedFile(null);
    setOriginalPreview(null);
    setImageDims(null);
    setResult(null);
    setStatusText('');
    setError(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const downloadReportPdf = async () => {
    if (!originalPreview || !result) return;
    setIsExporting(true);
    try {
      // loaded on demand so jsPDF doesn't slow down the first page load
      const { downloadPdfReport } = await import('./report');
      await downloadPdfReport(
        {
          fileName: selectedFile?.name ?? 'opg',
          imageDims,
          originalUrl: originalPreview,
          result,
          quadrantLabels: QUADRANT_LABELS,
        },
        `${fileBase}_report.pdf`
      );
    } catch (err) {
      console.error('Error creating PDF report:', err);
      setError("We couldn't create the PDF report. Please try again.");
    } finally {
      setIsExporting(false);
    }
  };

  // One screen at a time: upload -> ready -> analyzing -> results.
  const phase: 'upload' | 'ready' | 'analyzing' | 'results' = done
    ? 'results'
    : isProcessing
      ? 'analyzing'
      : selectedFile
        ? 'ready'
        : 'upload';

  return (
    <div className="galaxy-bg min-h-screen text-ink font-['Figtree',sans-serif] pb-16">
      <Navbar onNavigate={onNavigate} />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-5 flex flex-col gap-5">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <button
              onClick={() => onNavigate('landing')}
              className="-ml-1 mb-2 min-h-[32px] px-1 inline-flex items-center gap-1.5 text-[13px] font-semibold
                         text-muted hover:text-ink transition-colors"
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2"
                strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M19 12H5M11 6l-6 6 6 6" />
              </svg>
              Back to home
            </button>
            <h1 className={`text-[30px] sm:text-[38px] font-light tracking-tight leading-tight ${FONT_DISPLAY}`}>
              {phase === 'results' ? 'Your results' : 'Analyze your OPG'}
            </h1>
            <p className="text-[14px] text-muted mt-1">
              {phase === 'upload' && 'Start by adding a panoramic dental image (OPG).'}
              {phase === 'ready' && 'Looks good! Click “Analyze OPG” when you are ready.'}
              {phase === 'analyzing' && 'Hang tight — this usually takes under a minute.'}
              {phase === 'results' && 'Here are the areas that may need a closer look.'}
            </p>
          </div>
          <Steps current={phase === 'upload' ? 1 : phase === 'results' ? 3 : 2} />
        </div>

        <p className="sr-only" aria-live="polite">{statusText}</p>

        {error && (
          <div role="alert" className="flex items-start gap-3 rounded-2xl border border-[#F3B7AA] bg-[#FDEDEA] text-[#8A2A1B] px-4 py-3 text-[13.5px] dark:bg-[#3A1F1B] dark:border-[#6B3328] dark:text-[#FFC9BD]">
            <span className="mt-0.5 shrink-0" aria-hidden="true">⚠</span>
            <p className="flex-1">{error}</p>
            <button onClick={() => setError(null)} className="font-semibold hover:underline shrink-0">
              Dismiss
            </button>
          </div>
        )}

        {phase !== 'results' ? (
          /* ---------- upload / ready / analyzing: one focused card ---------- */
          <section
            onDragOver={(e) => {
              e.preventDefault();
              if (!isProcessing) setIsDragging(true);
            }}
            onDragLeave={(e) => {
              if (!e.currentTarget.contains(e.relatedTarget as Node)) setIsDragging(false);
            }}
            onDrop={handleDrop}
            className={`${CARD} w-full max-w-3xl mx-auto p-4 sm:p-6 transition-shadow ${
              isDragging ? 'ring-2 ring-[#0F766E] ring-offset-2 ring-offset-page' : ''
            }`}
          >
            {phase === 'upload' ? (
              <>
                <label
                  className={`neu-inset rounded-2xl flex flex-col items-center justify-center text-center gap-3
                              px-6 py-12 sm:py-16 cursor-pointer transition-colors ${isDragging ? 'text-link' : ''}`}
                >
                  <span className="neu-btn w-16 h-16 rounded-2xl bg-btn text-btn-fg flex items-center justify-center">
                    <UploadIcon />
                  </span>
                  <span className="text-[18px] font-semibold">
                    {isDragging ? 'Drop it here' : 'Drag & drop your OPG here'}
                  </span>
                  <span className="text-[14px] text-muted">
                    or <span className="font-semibold text-link underline underline-offset-2">choose an image</span> from
                    your computer
                  </span>
                  <span className="text-[12px] text-faint">PNG or JPEG · a full panoramic image works best</span>
                  <input type="file" accept="image/png, image/jpeg" onChange={handleFileChange} disabled={isProcessing} className="sr-only" />
                </label>
                <div className="mt-5 flex items-start gap-3 text-[13px] text-muted">
                  <InfoIcon />
                  <p>
                    <b className="text-ink">What is an OPG?</b> An orthopantomogram is a panoramic dental image that
                    shows all the teeth and both jaws in one wide picture. Dental clinics can give you a copy.
                  </p>
                </div>
              </>
            ) : (
              <>
                <div className="relative">
                  <Stage src={originalPreview!} alt="Your OPG" pos={crosshair} onPos={setCrosshair} />
                  {phase === 'ready' && (
                    <button
                      onClick={reset}
                      aria-label="Remove image"
                      title="Remove image"
                      className="absolute top-2 right-2 w-9 h-9 rounded-full bg-black/60 text-white inline-flex items-center
                                 justify-center hover:bg-black/80 transition-colors"
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4"
                        strokeLinecap="round" aria-hidden="true">
                        <path d="M6 6l12 12M18 6L6 18" />
                      </svg>
                    </button>
                  )}
                  {phase === 'analyzing' && (
                    <div className="absolute inset-0 rounded-xl bg-black/60 backdrop-blur-[2px] flex flex-col items-center justify-center gap-3 text-white text-center px-4">
                      <span className="animate-spin rounded-full h-10 w-10 border-[3px] border-white/25 border-t-[#34D399]" />
                      <p className="text-[16px] font-semibold">Analyzing your OPG…</p>
                      <p className="text-[13px] text-white/80">Looking for areas that may need a closer look.</p>
                    </div>
                  )}
                </div>

                <div className="mt-4 flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
                  <div className="min-w-0">
                    <p className="text-[15px] font-semibold truncate">{selectedFile?.name}</p>
                    <p className="text-xs text-muted">{imageDims ? `Image size: ${imageDims} px` : 'Loading…'}</p>
                  </div>
                  {phase === 'ready' && (
                    <div className="flex items-center gap-4 text-[13px] font-semibold">
                      <label className="text-link hover:underline cursor-pointer">
                        Choose a different image
                        <input type="file" accept="image/png, image/jpeg" onChange={handleFileChange} disabled={isProcessing} className="sr-only" />
                      </label>
                      <button onClick={reset} className="text-danger hover:underline">
                        Remove image
                      </button>
                    </div>
                  )}
                </div>

                {phase === 'ready' && (
                  <>
                    <button
                      onClick={handleAnalyze}
                      className="neu-btn mt-5 w-full min-h-[54px] rounded-2xl bg-btn text-btn-fg text-[16px] font-semibold
                                 inline-flex items-center justify-center gap-2 hover:bg-[var(--btn-hover)]"
                    >
                      <SparkleIcon />
                      Analyze OPG
                    </button>
                    <p className="mt-2 text-center text-[12.5px] text-muted">
                      We'll highlight areas on your image that may need a dentist's attention.
                    </p>
                  </>
                )}
              </>
            )}
          </section>
        ) : (
          /* ---------------------------- results ---------------------------- */
          <>
            <section className={`${CARD} p-4 sm:p-5 flex flex-col md:flex-row md:items-center gap-4`}>
              <div className="min-w-0 flex-1">
                <p className="text-[13px] text-muted truncate">{selectedFile?.name}</p>
                <p className={`text-[22px] font-semibold tracking-tight ${FONT_HEAD}`}>
                  {regions.length === 0
                    ? 'No areas were flagged'
                    : `${plural(regions.length, 'area', 'areas')} may need a closer look`}
                </p>
                {regions.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {groupByLabel(regions).map(([label, items]) => (
                      <span
                        key={label}
                        className="neu-sm inline-flex items-center gap-1.5 text-[12px] font-semibold px-2.5 py-1 rounded-full"
                      >
                        <span
                          className="w-[7px] h-[7px] rounded-full"
                          style={{ backgroundColor: DIAGNOSIS_ACCENT[label] ?? DIAGNOSIS_DEFAULT_ACCENT }}
                        />
                        {label} × {items.length}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              <div className="flex flex-wrap gap-2 shrink-0">
                <button
                  onClick={downloadReportPdf}
                  disabled={isExporting}
                  className="neu-btn min-h-[44px] px-5 rounded-full bg-btn text-btn-fg text-[14px] font-semibold
                             inline-flex items-center gap-2 hover:bg-[var(--btn-hover)] disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <DownloadIcon />
                  {isExporting ? 'Creating PDF…' : 'Download report'}
                </button>
                <button
                  onClick={reset}
                  className="neu-sm min-h-[44px] px-5 rounded-full text-[14px] font-semibold hover:text-link"
                >
                  Analyze another OPG
                </button>
              </div>
            </section>

            {/* your original next to the highlighted result */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Frame title="Your OPG" subtitle="The image you uploaded" onExpand={() => setFullscreenImage(originalPreview)}>
                <Stage src={originalPreview!} alt="Your OPG" pos={crosshair} onPos={setCrosshair} capHeight />
              </Frame>
              <Frame
                title="Highlighted areas"
                subtitle="Colored boxes mark what was found"
                accent="#0F766E"
                onExpand={resultImage ? () => setFullscreenImage(resultImage) : undefined}
                onDownload={resultImage ? () => triggerDownload(resultImage, `${fileBase}_findings.png`) : undefined}
              >
                {resultImage ? (
                  <Stage src={resultImage} alt="Highlighted areas" pos={crosshair} onPos={setCrosshair} capHeight />
                ) : (
                  <EmptyStage>No highlighted image was returned.</EmptyStage>
                )}
              </Frame>
            </div>

            <section className={`${CARD} p-5 flex flex-col gap-3`}>
              <div>
                <h2 className={`text-[17px] font-semibold ${FONT_HEAD}`}>What was found</h2>
                <p className="text-[12.5px] text-muted">Grouped by type. Locations are from the patient's point of view.</p>
              </div>
              <FindingsList regions={regions} />
            </section>

            <details className={`${CARD} group`}>
              <summary className="cursor-pointer list-none px-5 py-4 flex items-center justify-between gap-3">
                <span>
                  <span className={`block text-[16px] font-semibold ${FONT_HEAD}`}>Technical details</span>
                  <span className="block text-[12.5px] text-muted">The exact numbers behind this result, for researchers.</span>
                </span>
                <span className="text-muted transition-transform group-open:rotate-180" aria-hidden="true">▾</span>
              </summary>
              <div className="px-5 pb-5">
                <MetricsPanel result={result} />
              </div>
            </details>
          </>
        )}

        {/* How to read the results (only useful once there are results) */}
        <section className={`${CARD} p-5 ${phase === 'results' ? '' : 'max-w-3xl w-full mx-auto'}`}>
          {phase === 'results' && (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-x-8 gap-y-4 text-[12.5px] text-muted mb-5">
              <div>
                <h3 className="text-[11.5px] font-semibold tracking-wider uppercase mb-2">What the colors mean</h3>
                <ul className="space-y-1.5">
                  {Object.keys(DIAGNOSIS_ACCENT).map((d) => (
                    <li key={d} className="flex items-start gap-2">
                      <span
                        className="w-2.5 h-2.5 rounded-full mt-1 shrink-0"
                        style={{ backgroundColor: DIAGNOSIS_ACCENT[d] }}
                      />
                      <span>
                        <span className="text-ink font-semibold">{d}</span> — {DIAGNOSIS_INFO[d]}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <h3 className="text-[11.5px] font-semibold tracking-wider uppercase mb-2">What is “confidence”?</h3>
                <p className="leading-relaxed">
                  How sure the computer model is about each highlighted area. It is not a medical certainty —
                  values around 25–45% are normal for this research tool.
                </p>
              </div>
              <div>
                <h3 className="text-[11.5px] font-semibold tracking-wider uppercase mb-2">Tips</h3>
                <ul className="space-y-1 leading-relaxed list-disc pl-4">
                  <li>Point at either image to mark the same spot on both.</li>
                  <li>Use the expand button to view an image full screen.</li>
                  <li>“Download report” saves everything as a PDF.</li>
                </ul>
              </div>
            </div>
          )}
          <div className="bg-warn-bg border border-warn-line text-warn-ink text-[12.5px] rounded-xl px-4 py-3">
            <strong>⚠ Not a medical diagnosis.</strong> This is a research tool for academic demonstration
            only. The highlighted areas may be wrong or incomplete. Please consult a licensed dentist for any
            actual diagnosis or treatment.
          </div>
        </section>
      </main>

      {fullscreenImage && (
        <div
          onClick={() => setFullscreenImage(null)}
          className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-8 cursor-zoom-out"
        >
          <button
            onClick={() => setFullscreenImage(null)}
            className="absolute top-5 right-6 w-11 h-11 rounded-full border border-white/30 text-white/80 hover:text-white text-2xl leading-none"
            aria-label="Close"
          >
            ×
          </button>
          <img
            src={fullscreenImage}
            alt="Expanded view"
            className="max-w-full max-h-full object-contain rounded-lg"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
}

const CARD = 'neu rounded-[22px]';

const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;

// Findings grouped by condition, most frequent first.
function groupByLabel(regions: DetectedRegion[]): [string, DetectedRegion[]][] {
  return Object.entries(
    regions.reduce<Record<string, DetectedRegion[]>>((acc, r) => {
      (acc[r.label] ??= []).push(r);
      return acc;
    }, {})
  ).sort((a, b) => b[1].length - a[1].length);
}

// Three-step guide at the top of the analyzer.
function Steps({ current }: { current: 1 | 2 | 3 }) {
  const steps = ['Upload your OPG', 'Click Analyze', 'Review the results'];
  return (
    <ol className="flex flex-wrap items-center gap-x-3 gap-y-2 text-[13px]" aria-label="Steps">
      {steps.map((label, i) => {
        const n = i + 1;
        const state = n < current ? 'done' : n === current ? 'current' : 'todo';
        return (
          <li key={label} className="flex items-center gap-2" aria-current={state === 'current' ? 'step' : undefined}>
            <span
              className={`w-7 h-7 rounded-full flex items-center justify-center text-[12px] font-bold ${
                state === 'todo' ? 'neu-inset text-faint' : 'neu-btn bg-btn text-btn-fg'
              }`}
            >
              {state === 'done' ? '✓' : n}
            </span>
            <span className={state === 'todo' ? 'text-faint' : 'font-semibold'}>{label}</span>
            {n < steps.length && <span className="hidden sm:block w-8 h-px bg-line-strong" />}
          </li>
        );
      })}
    </ol>
  );
}

function InfoIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className="shrink-0 mt-0.5 text-link">
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5M12 8v.01" />
    </svg>
  );
}

function SparkleIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8z" />
    </svg>
  );
}

function ExpandIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" aria-hidden="true">
      <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 16V4M7 9l5-5 5 5M5 20h14" />
    </svg>
  );
}

function DownloadIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 4v11M7 10l5 5 5-5M5 20h14" />
    </svg>
  );
}

function triggerDownload(href: string, filename: string) {
  const a = document.createElement('a');
  a.href = href;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function Frame({
  title,
  subtitle,
  accent,
  right,
  onExpand,
  onDownload,
  footer,
  children,
}: {
  title: string;
  subtitle?: string;
  accent?: string;
  right?: React.ReactNode;
  onExpand?: () => void;
  onDownload?: () => void;
  footer?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <article className={`${CARD} overflow-hidden flex flex-col min-w-0`}>
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2.5 border-b border-line">
        <span className="flex items-center gap-2 min-w-0">
          {accent && <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: accent }} />}
          <span className="min-w-0">
            <span className="block text-[13.5px] font-semibold text-ink truncate">{title}</span>
            {subtitle && <span className="block text-[11.5px] text-muted">{subtitle}</span>}
          </span>
        </span>
        <span className="flex items-center gap-2.5 text-xs text-muted">
          {right}
          {onDownload && (
            <button
              onClick={onDownload}
              aria-label={`Download ${title} image`}
              title="Download image (PNG)"
              className="neu-sm w-9 h-9 rounded-lg inline-flex items-center justify-center hover:text-link transition-colors"
            >
              <DownloadIcon />
            </button>
          )}
          <button
            onClick={onExpand}
            disabled={!onExpand}
            aria-label={`Expand ${title}`}
            className="neu-sm w-9 h-9 rounded-lg inline-flex items-center justify-center
                       hover:text-link transition-colors disabled:opacity-35"
          >
            <ExpandIcon />
          </button>
        </span>
      </div>
      <div className="p-2.5">{children}</div>
      {footer && <div className="px-4 pb-3">{footer}</div>}
    </article>
  );
}

function EmptyStage({ children }: { children: React.ReactNode }) {
  return (
    <div className="neu-inset aspect-[2/1] rounded-xl flex items-center justify-center text-sm text-faint">
      {children}
    </div>
  );
}

// Image stage with a crosshair that mirrors the pointer position (as a
// fraction of the image) so both frames point at the same spot.
// With `capHeight` the image is limited to part of the screen height and
// centered on the dark stage, so tall OPGs don't take over the page.
function Stage({
  src,
  alt,
  pos,
  onPos,
  capHeight = false,
}: {
  src: string;
  alt: string;
  pos: CrossPos;
  onPos: (p: CrossPos) => void;
  capHeight?: boolean;
}) {
  const image = (
    <div
      className={`relative overflow-hidden ${capHeight ? 'w-fit max-w-full' : 'rounded-xl bg-[var(--stage)]'}`}
      onPointerMove={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        onPos({ x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height });
      }}
      onPointerLeave={() => onPos(null)}
    >
      <img
        src={src}
        alt={alt}
        className={capHeight ? 'block max-w-full w-auto h-auto max-h-[58vh]' : 'block w-full h-auto'}
      />
      {pos && (
        <div className="pointer-events-none absolute inset-0">
          <div
            className="absolute w-[26px] h-[26px] -ml-[13px] -mt-[13px] rounded-full border-[1.5px] border-white"
            style={{ left: `${pos.x * 100}%`, top: `${pos.y * 100}%` }}
          />
        </div>
      )}
    </div>
  );
  return capHeight ? (
    <div className="rounded-xl overflow-hidden bg-[var(--stage)] flex justify-center">{image}</div>
  ) : (
    image
  );
}

// What one method flagged under its image: one row per condition (count,
// where, highest confidence), with every single area tucked in a disclosure.
function FindingsList({ regions }: { regions: DetectedRegion[] }) {
  if (regions.length === 0) {
    return <p className="text-[13px] text-muted py-1">No areas were flagged on this image.</p>;
  }
  const groups = groupByLabel(regions);
  const pct = (c?: number) => (c != null ? `${Math.round(c * 100)}%` : null);

  return (
    <div className="flex flex-col gap-2">
      {groups.map(([label, items]) => {
        const accent = DIAGNOSIS_ACCENT[label] ?? DIAGNOSIS_DEFAULT_ACCENT;
        const where = [...new Set(items.map((r) => QUADRANT_LABELS[r.quadrant]))].join(', ');
        const best = Math.max(...items.map((r) => r.confidence ?? 0));
        return (
          <div key={label} className="neu-inset flex items-center gap-3 px-3 py-2.5 rounded-xl">
            <span className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: accent }} />
            <div className="min-w-0">
              <p className="text-[13.5px] font-semibold">
                {label}{' '}
                <span className="font-normal text-muted">
                  · {plural(items.length, 'area', 'areas')} · {where}
                </span>
              </p>
              <p className="text-xs text-muted">{DIAGNOSIS_INFO[label] ?? ''}</p>
            </div>
            {best > 0 && (
              <span className="ml-auto text-right shrink-0">
                <span className={`block text-[13px] font-semibold ${FONT_MONO}`}>{pct(best)}</span>
                <span className="block text-[10.5px] text-muted">{items.length > 1 ? 'top confidence' : 'confidence'}</span>
              </span>
            )}
          </div>
        );
      })}

      {regions.length > 1 && (
        <details className="text-[12.5px] text-muted">
          <summary className="cursor-pointer font-semibold hover:text-ink w-fit py-1">
            Show all {regions.length} areas
          </summary>
          <ul className="mt-1 divide-y divide-line">
            {regions.map((r, i) => (
              <li key={i} className="flex items-center gap-2 py-1.5">
                <span
                  className="w-2 h-2 rounded-full shrink-0"
                  style={{ backgroundColor: DIAGNOSIS_ACCENT[r.label] ?? DIAGNOSIS_DEFAULT_ACCENT }}
                />
                <span className="text-ink">{r.label}</span>
                <span>· {QUADRANT_LABELS[r.quadrant]}</span>
                {r.confidence != null && <span className={`ml-auto ${FONT_MONO}`}>{pct(r.confidence)}</span>}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

// The raw numbers behind the result, shown under "Technical details".
function MetricsPanel({ result }: { result: AnalyzeResult | null }) {
  if (result?.status !== 'success') return <p className="text-sm text-muted">No numbers available.</p>;
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-5">
      <MetricRow label="Brightness cut-offs (thresholds)" value={result.thresholds?.join(', ') ?? '-'} />
      <MetricRow label="Segmentation quality (Kapur's entropy)" value={result.kapur_entropy_fitness?.toFixed(4) ?? '-'} />
      <MetricRow label="Image fidelity (PSNR, dB)" value={result.psnr != null ? result.psnr.toFixed(4) : '∞'} />
      <MetricRow label="Structure kept (SSIM)" value={result.ssim?.toFixed(6) ?? '-'} />
      <MetricRow label="Processing time (s)" value={result.runtime_sec?.toFixed(4) ?? '-'} />
      <MetricRow
        label="Steps used (iterations)"
        value={result.iterations_used != null ? `${result.iterations_used} / 100` : '-'}
      />
    </div>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2 py-2.5 border-b border-line">
      <span className="text-[12.5px] text-muted">{label}</span>
      <span className={`text-sm font-medium ${FONT_MONO}`}>{value}</span>
    </div>
  );
}


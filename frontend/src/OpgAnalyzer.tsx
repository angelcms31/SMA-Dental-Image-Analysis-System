import React, { useState, type ChangeEvent, type DragEvent } from 'react';
import axios from 'axios';
import AlgorithmSimulation from './AlgorithmSimulation';
import { API_BASE } from './api';
import Navbar, { NAV_BUTTON, type Navigate } from './Navbar';

type AlgoTab = 'standard' | 'enhanced';

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

// "Clinic" palette: airy white surfaces, sky blue for Standard, warm orange for ESMA.
const TAB_META: Record<AlgoTab, { label: string; short: string; accent: string }> = {
  standard: { label: 'Standard SMA', short: 'Standard', accent: '#0F766E' },
  enhanced: { label: 'Enhanced SMA', short: 'ESMA', accent: '#E39446' },
};
const ESMA_TEXT = '#C8711F';

const FONT_HEAD = "font-['Inter_Tight',sans-serif]";
const FONT_DISPLAY = "font-['Inter_Tight',sans-serif]";
const FONT_MONO = "font-['IBM_Plex_Mono',monospace] tabular-nums";

const ALGOS: AlgoTab[] = ['standard', 'enhanced'];

export default function OpgAnalyzer({ onNavigate }: { onNavigate: Navigate }) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [originalPreview, setOriginalPreview] = useState<string | null>(null);
  const [imageDims, setImageDims] = useState<string | null>(null);
  const [processing, setProcessing] = useState<Record<AlgoTab, boolean>>({
    standard: false,
    enhanced: false,
  });
  const [runningLabel, setRunningLabel] = useState<string>('');
  const [resultsView, setResultsView] = useState<'findings' | 'metrics'>('findings');
  const [fullscreenImage, setFullscreenImage] = useState<string | null>(null);
  const [showSimulation, setShowSimulation] = useState<boolean>(false);
  const [crosshair, setCrosshair] = useState<CrossPos>(null);
  const [isExporting, setIsExporting] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  const [results, setResults] = useState<Record<AlgoTab, AnalyzeResult | null>>({
    standard: null,
    enhanced: null,
  });

  const isProcessing = processing.standard || processing.enhanced;
  const bothDone = results.standard?.status === 'success' && results.enhanced?.status === 'success';
  const anyDone = results.standard?.status === 'success' || results.enhanced?.status === 'success';

  const loadFile = (file: File) => {
    if (!/^image\/(png|jpeg)$/.test(file.type)) {
      alert('PNG o JPEG lang ang puwedeng i-upload.');
      return;
    }
    const url = URL.createObjectURL(file);
    setSelectedFile(file);
    setOriginalPreview(url);
    setImageDims(null);
    setResults({ standard: null, enhanced: null });
    setRunningLabel('');
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

  const runAlgorithm = async (algo: AlgoTab, file: File): Promise<AnalyzeResult> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('d', '4');
    formData.append('N', '30');
    formData.append('T', '100');

    const endpoint =
      algo === 'standard'
        ? `${API_BASE}/analyze/standard/`
        : `${API_BASE}/analyze/enhanced/`;

    const yoloFormData = new FormData();
    yoloFormData.append('file', file);
    yoloFormData.append('conf', '0.25');
    yoloFormData.append('iou', '0.35');
    yoloFormData.append('use_esma', 'true');
    yoloFormData.append('sma_algorithm', algo === 'standard' ? 'standard' : 'enhanced_v2');
    yoloFormData.append(
      'weights_path',
      algo === 'standard'
        ? 'runs_yolo/train_standard_v2/weights/best.pt'
        : 'runs_yolo/train_v2_mendeley/weights/best.pt'
    );

    const [metricsRes, yoloRes] = await Promise.all([
      axios.post<AnalyzeResult>(endpoint, formData, {
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

  // Each algorithm fills its own screen as soon as it finishes.
  const handleRun = async (algos: AlgoTab[]) => {
    if (!selectedFile) return;
    const file = selectedFile;
    setProcessing((prev) => ({ ...prev, ...Object.fromEntries(algos.map((a) => [a, true])) }));
    setRunningLabel(algos.length > 1 ? 'Running Standard and ESMA…' : `Running ${TAB_META[algos[0]].label}…`);
    let failed = false;
    await Promise.all(
      algos.map(async (algo) => {
        try {
          const result = await runAlgorithm(algo, file);
          setResults((prev) => ({ ...prev, [algo]: result }));
        } catch (error) {
          console.error('Error analyzing image:', error);
          failed = true;
        } finally {
          setProcessing((prev) => ({ ...prev, [algo]: false }));
        }
      })
    );
    if (failed) {
      setRunningLabel('');
      alert('May error sa pag-connect sa backend. Siguraduhing tumatakbo ang FastAPI server.');
    } else {
      setRunningLabel(algos.length > 1 ? 'Both finished' : `${TAB_META[algos[0]].label} finished`);
    }
  };

  const fileBase = selectedFile ? selectedFile.name.replace(/\.[^.]+$/, '') : 'opg';

  const downloadFindingsImage = (algo: AlgoTab, image: string) => {
    triggerDownload(image, `${fileBase}_${algo === 'enhanced' ? 'esma' : 'standard'}_findings.png`);
  };

  const downloadReportPdf = async () => {
    if (!originalPreview) return;
    setIsExporting(true);
    try {
      // loaded on demand so jsPDF doesn't slow down the first page load
      const { downloadPdfReport } = await import('./report');
      await downloadPdfReport(
        {
          fileName: selectedFile?.name ?? 'opg',
          imageDims,
          originalUrl: originalPreview,
          standard: results.standard,
          enhanced: results.enhanced,
          quadrantLabels: QUADRANT_LABELS,
        },
        `${fileBase}_report.pdf`
      );
    } catch (error) {
      console.error('Error creating PDF report:', error);
      alert('Hindi nagawa ang PDF report. Subukan ulit.');
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="galaxy-bg min-h-screen text-ink font-['Figtree',sans-serif] pb-16">
      <Navbar
        onNavigate={onNavigate}
        action={
          <button onClick={() => setShowSimulation(true)} className={NAV_BUTTON}>
            Algorithm simulation
          </button>
        }
      />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-5 flex flex-col gap-4">
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
            Standard vs Enhanced SMA
          </h1>
          <p className="text-[14px] text-muted mt-1">
            Upload an orthopantomogram (OPG), run the algorithms, and compare both results side by side.
          </p>
        </div>

        {/* Toolbar: Original thumbnail + file + run buttons; the whole card is a drop zone */}
        <section
          onDragOver={(e) => {
            e.preventDefault();
            if (!isProcessing) setIsDragging(true);
          }}
          onDragLeave={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget as Node)) setIsDragging(false);
          }}
          onDrop={handleDrop}
          className={`${CARD} p-3 sm:p-4 flex flex-col md:flex-row md:items-center gap-4 transition-shadow ${
            isDragging ? 'ring-2 ring-[#0F766E] ring-offset-2 ring-offset-page' : ''
          }`}
        >
          <div className="relative w-full md:w-[250px] shrink-0">
            {originalPreview ? (
              <>
                <Stage src={originalPreview} alt="Original OPG" pos={crosshair} onPos={setCrosshair} />
                <span className="pointer-events-none absolute top-2 left-2 text-[10px] font-semibold tracking-[0.11em] uppercase bg-black/55 text-white px-2 py-0.5 rounded-full">
                  Original
                </span>
                <button
                  onClick={() => setFullscreenImage(originalPreview)}
                  aria-label="Expand original OPG"
                  className="absolute top-1.5 right-1.5 w-8 h-8 rounded-lg bg-black/55 text-white inline-flex items-center justify-center hover:bg-black/75"
                >
                  <ExpandIcon />
                </button>
              </>
            ) : (
              <label
                className="neu-inset aspect-[2/1] rounded-xl flex flex-col
                           items-center justify-center gap-1.5 text-[12.5px] font-semibold text-muted cursor-pointer
                           hover:text-link transition-colors"
              >
                <UploadIcon />
                Drop OPG here
                <input type="file" accept="image/png, image/jpeg" onChange={handleFileChange} disabled={isProcessing} className="sr-only" />
              </label>
            )}
          </div>

          <div className="min-w-0 flex-1">
            <p className="text-[11px] font-semibold tracking-[0.11em] uppercase text-muted">Original OPG</p>
            <p className="text-[15px] font-semibold truncate mt-0.5">
              {selectedFile ? selectedFile.name : 'No OPG selected'}
            </p>
            <p className="text-xs text-muted">
              {selectedFile ? imageDims ?? 'Loading…' : 'PNG or JPEG panoramic image — drag it here or choose a file'}
            </p>
            <label className="mt-2 inline-flex text-[13px] font-semibold text-link hover:underline cursor-pointer">
              {selectedFile ? 'Change OPG' : 'Choose OPG'}
              <input type="file" accept="image/png, image/jpeg" onChange={handleFileChange} disabled={isProcessing} className="sr-only" />
            </label>
          </div>

          <div className="flex flex-col gap-1.5 md:items-end shrink-0">
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => handleRun(['standard'])}
                disabled={!selectedFile || isProcessing}
                className={`${BTN} text-white`}
                style={{ backgroundColor: TAB_META.standard.accent }}
              >
                Standard
              </button>
              <button
                onClick={() => handleRun(['enhanced'])}
                disabled={!selectedFile || isProcessing}
                className={`${BTN} text-white`}
                style={{ backgroundColor: TAB_META.enhanced.accent }}
              >
                ESMA
              </button>
              <button
                onClick={() => handleRun(ALGOS)}
                disabled={!selectedFile || isProcessing}
                className={`${BTN} text-btn-fg bg-btn`}
              >
                Run both
              </button>
            </div>
            <p className="text-[12.5px] text-muted min-h-[1.1rem]" aria-live="polite">{runningLabel}</p>
          </div>
        </section>

        {/* Two result screens: Standard SMA and ESMA */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {ALGOS.map((algo) => {
            const result = results[algo];
            const busy = processing[algo];
            const image = !busy && result?.status === 'success' ? result.annotated_image ?? null : null;
            const regions = result?.status === 'success' ? result.detected_regions ?? [] : [];
            return (
              <Frame
                key={algo}
                title={TAB_META[algo].label}
                accent={TAB_META[algo].accent}
                right={image ? `${regions.length} findings` : undefined}
                onExpand={image ? () => setFullscreenImage(image) : undefined}
                onDownload={image ? () => downloadFindingsImage(algo, image) : undefined}
                footer={
                  image && regions.length > 0 ? (
                    <div className="flex flex-wrap gap-1.5">
                      {regions.map((r, i) => (
                        <span
                          key={i}
                          className="neu-sm inline-flex items-center gap-1.5 text-[11.5px] font-semibold px-2.5 py-1 rounded-full"
                        >
                          <span
                            className="w-[7px] h-[7px] rounded-full"
                            style={{ backgroundColor: DIAGNOSIS_ACCENT[r.label] ?? DIAGNOSIS_DEFAULT_ACCENT }}
                          />
                          {r.quadrant} {r.label}
                        </span>
                      ))}
                    </div>
                  ) : undefined
                }
              >
                {busy ? (
                  <EmptyStage>
                    <span
                      className="animate-spin rounded-full h-7 w-7 border-2 border-[#0F766E]/15 mr-3"
                      style={{ borderTopColor: TAB_META[algo].accent }}
                    />
                    Computing Kapur's entropy…
                  </EmptyStage>
                ) : image ? (
                  <Stage src={image} alt={`${TAB_META[algo].label} findings`} pos={crosshair} onPos={setCrosshair} />
                ) : result?.status === 'error' ? (
                  <EmptyStage>
                    <span className="text-danger px-4 text-center">{result.message}</span>
                  </EmptyStage>
                ) : (
                  <EmptyStage>Awaiting analysis</EmptyStage>
                )}
              </Frame>
            );
          })}
        </div>

        {/* Results */}
        <section className={`${CARD} p-5 flex flex-col gap-4`}>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className={`text-[17px] font-semibold ${FONT_HEAD}`}>Results</h2>
            <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={downloadReportPdf}
              disabled={!anyDone || isProcessing || isExporting}
              className="neu-sm min-h-[38px] px-3.5 rounded-full text-[13px] font-semibold
                         inline-flex items-center gap-2 hover:text-link transition-colors
                         disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <DownloadIcon />
              {isExporting ? 'Creating PDF…' : 'Download report (PDF)'}
            </button>
            <div className="neu-inset inline-flex rounded-full p-1 gap-1" role="tablist">
              {(['findings', 'metrics'] as const).map((v) => (
                <button
                  key={v}
                  role="tab"
                  aria-selected={resultsView === v}
                  onClick={() => setResultsView(v)}
                  className={`min-h-[34px] px-3.5 rounded-full text-[13px] font-semibold capitalize transition-all ${
                    resultsView === v ? 'neu-sm text-ink' : 'text-muted hover:text-ink'
                  }`}
                >
                  {v}
                </button>
              ))}
            </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {ALGOS.map((algo) => (
              <ResultPanel key={algo} algo={algo} result={results[algo]} busy={processing[algo]} view={resultsView} />
            ))}
          </div>
        </section>

        {/* Standard vs Enhanced */}
        {bothDone && (
          <section className={`${CARD} p-5 flex flex-col gap-3.5`}>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className={`text-[17px] font-semibold ${FONT_HEAD}`}>Standard vs Enhanced</h2>
              <span className="text-[12.5px] text-muted">Highlighted value = better result</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-2.5">
              <CompareCard
                label="Kapur's entropy"
                a={results.standard!.kapur_entropy_fitness}
                b={results.enhanced!.kapur_entropy_fitness}
                digits={4}
              />
              <CompareCard label="PSNR (dB)" a={results.standard!.psnr} b={results.enhanced!.psnr} digits={4} />
              <CompareCard label="SSIM" a={results.standard!.ssim} b={results.enhanced!.ssim} digits={4} />
              <CompareCard
                label="Runtime (s)"
                a={results.standard!.runtime_sec}
                b={results.enhanced!.runtime_sec}
                digits={4}
                lowerIsBetter
              />
              <CompareCard
                label="Iterations"
                a={results.standard!.iterations_used}
                b={results.enhanced!.iterations_used}
                digits={0}
                lowerIsBetter
              />
            </div>
          </section>
        )}

        {/* How to read */}
        <section className={`${CARD} p-5`}>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-x-8 gap-y-4 text-[12.5px] text-muted">
            <div>
              <h3 className="text-[11.5px] font-semibold tracking-wider uppercase mb-2">Diagnosis colors</h3>
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
              <h3 className="text-[11.5px] font-semibold tracking-wider uppercase mb-2">Confidence %</h3>
              <p className="leading-relaxed">
                How certain the trained detection model is about that specific finding — not a measure of
                clinical certainty. Values are typically modest (25–45%) since this is an exploratory,
                non-validated feature.
              </p>
            </div>
            <div>
              <h3 className="text-[11.5px] font-semibold tracking-wider uppercase mb-2">Frames</h3>
              <ul className="space-y-1 leading-relaxed">
                <li><b className="text-ink">Original</b> — the uploaded OPG, unmodified.</li>
                <li><b className="text-ink">Standard SMA / ESMA</b> — segmentation output of each algorithm with flagged regions.</li>
                <li>Hover one frame and the crosshair appears on all three.</li>
              </ul>
            </div>
          </div>
          <div className="bg-warn-bg border border-warn-line text-warn-ink text-[12.5px] rounded-xl px-4 py-3 mt-5">
            <strong>⚠ Not a medical diagnosis.</strong> Research prototype for academic demonstration only.
            This output is NOT a clinical diagnosis. Findings are generated by a trained detection model
            and threshold-based segmentation, not a validated diagnostic tool. Please consult a licensed
            dentist for any actual diagnosis or treatment.
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

      {showSimulation && (
        <AlgorithmSimulation onClose={() => setShowSimulation(false)} selectedFile={selectedFile} />
      )}
    </div>
  );
}

const CARD = 'neu rounded-[22px]';
const BTN =
  'neu-btn min-h-[42px] px-[18px] rounded-full text-[13.5px] font-semibold inline-flex items-center ' +
  'disabled:opacity-40 disabled:cursor-not-allowed shrink-0';

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
  accent,
  right,
  onExpand,
  onDownload,
  footer,
  children,
}: {
  title: string;
  accent?: string;
  right?: string;
  onExpand?: () => void;
  onDownload?: () => void;
  footer?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <article className={`${CARD} overflow-hidden flex flex-col min-w-0`}>
      <div className="flex items-center justify-between gap-2 px-4 py-2.5 border-b border-line">
        <span className="flex items-center gap-2 text-[11px] font-semibold tracking-[0.11em] uppercase text-muted">
          {accent && <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: accent }} />}
          {title}
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
function Stage({
  src,
  alt,
  pos,
  onPos,
}: {
  src: string;
  alt: string;
  pos: CrossPos;
  onPos: (p: CrossPos) => void;
}) {
  return (
    <div
      className="relative rounded-xl overflow-hidden bg-[var(--stage)]"
      onPointerMove={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        onPos({ x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height });
      }}
      onPointerLeave={() => onPos(null)}
    >
      <img src={src} alt={alt} className="block w-full h-auto" />
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
}

function ResultPanel({
  algo,
  result,
  busy,
  view,
}: {
  algo: AlgoTab;
  result: AnalyzeResult | null;
  busy: boolean;
  view: 'findings' | 'metrics';
}) {
  const regions = result?.status === 'success' ? result.detected_regions ?? [] : [];
  return (
    <div className="flex flex-col gap-2.5 min-w-0">
      <h3 className="flex items-center gap-2 text-[14px] font-semibold">
        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: TAB_META[algo].accent }} />
        {TAB_META[algo].label}
      </h3>
      {busy ? (
        <p className="text-sm text-muted py-4">Running…</p>
      ) : result?.status !== 'success' ? (
        <p className="text-sm text-muted py-4">Awaiting analysis</p>
      ) : view === 'metrics' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-5">
          <MetricRow label="Thresholds" value={result.thresholds?.join(', ') ?? '-'} />
          <MetricRow label="Kapur's entropy" value={result.kapur_entropy_fitness?.toFixed(4) ?? '-'} />
          <MetricRow label="PSNR (dB)" value={result.psnr != null ? result.psnr.toFixed(4) : '∞'} />
          <MetricRow label="SSIM" value={result.ssim?.toFixed(6) ?? '-'} />
          <MetricRow label="Runtime (s)" value={result.runtime_sec?.toFixed(4) ?? '-'} />
          <MetricRow
            label="Iterations"
            value={result.iterations_used != null ? `${result.iterations_used} / 100` : '-'}
          />
        </div>
      ) : regions.length === 0 ? (
        <p className="text-sm text-muted">No regions were flagged in this image.</p>
      ) : (
        <ul className="grid grid-cols-1 gap-2">
          {regions.map((region, idx) => {
            const accent = DIAGNOSIS_ACCENT[region.label] ?? DIAGNOSIS_DEFAULT_ACCENT;
            const area = region.area_px != null
              ? `${Math.round(region.area_px).toLocaleString()} px`
              : region.bbox ? `${region.bbox[2]} × ${region.bbox[3]} px box` : null;
            return (
              <li
                key={idx}
                className="neu-inset flex items-center gap-3 px-3 py-2.5 rounded-xl"
              >
                <span
                  className={`w-[38px] py-[3px] rounded-md text-center text-[13px] font-medium text-white shrink-0 ${FONT_MONO}`}
                  style={{ backgroundColor: accent }}
                >
                  {region.quadrant}
                </span>
                <div className="min-w-0">
                  <p className="text-[13.5px] font-semibold truncate">{region.label}</p>
                  <p className="text-xs text-muted">
                    {QUADRANT_LABELS[region.quadrant]}
                    {area != null && ` · ${area}`}
                  </p>
                </div>
                {region.confidence != null && (
                  <span className={`ml-auto text-[12.5px] text-muted ${FONT_MONO}`}>
                    {Math.round(region.confidence * 100)}%
                  </span>
                )}
              </li>
            );
          })}
        </ul>
      )}
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

function CompareCard({
  label,
  a,
  b,
  digits,
  lowerIsBetter = false,
}: {
  label: string;
  a?: number | null;
  b?: number | null;
  digits: number;
  lowerIsBetter?: boolean;
}) {
  if (a == null || b == null) return null;
  const bWins = lowerIsBetter ? b < a : b > a;
  const diff = b - a;
  const rel = a !== 0 ? (diff / a) * 100 : 0;
  const sign = diff > 0 ? '+' : diff < 0 ? '−' : '±';
  const relText = `${sign}${Math.abs(rel).toFixed(Math.abs(rel) < 1 ? 2 : 0)}%`;

  return (
    <div className="neu-sm flex flex-col gap-2 p-3.5 rounded-xl">
      <div>
        <p className="text-[13px] font-semibold">{label}</p>
        <p className="text-[11px] text-muted">{lowerIsBetter ? 'lower is better' : 'higher is better'}</p>
      </div>
      <div className="flex items-center gap-2 text-[12.5px] text-muted">
        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: TAB_META.standard.accent }} />
        Standard
        <span className={`ml-auto text-sm font-medium text-ink ${FONT_MONO}`}>{a.toFixed(digits)}</span>
      </div>
      <div className="flex items-center gap-2 text-[12.5px] text-muted">
        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: TAB_META.enhanced.accent }} />
        ESMA
        <span
          className={`ml-auto text-sm ${FONT_MONO} ${bWins ? 'font-semibold' : 'font-medium text-ink'}`}
          style={bWins ? { color: ESMA_TEXT } : undefined}
        >
          {b.toFixed(digits)}
        </span>
      </div>
      <span
        className={`self-start mt-auto text-[11.5px] font-medium px-2 py-0.5 rounded-full neu-inset ${FONT_MONO}`}
        style={{ color: bWins ? ESMA_TEXT : 'var(--muted)' }}
      >
        {sign}
        {Math.abs(diff).toFixed(digits)} ({relText})
      </span>
    </div>
  );
}

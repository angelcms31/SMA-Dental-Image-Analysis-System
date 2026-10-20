import React, { useState, type ChangeEvent } from 'react';
import axios from 'axios';
import AlgorithmSimulation from './AlgorithmSimulation';
import { API_BASE } from './api';

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
  standard: { label: 'Standard SMA', short: 'Standard', accent: '#1F7AA3' },
  enhanced: { label: 'Enhanced SMA', short: 'ESMA', accent: '#E39446' },
};
const ESMA_TEXT = '#C8711F';

const FONT_HEAD = "font-['Outfit',sans-serif]";
const FONT_MONO = "font-['IBM_Plex_Mono',monospace] tabular-nums";

export default function OpgAnalyzer() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [originalPreview, setOriginalPreview] = useState<string | null>(null);
  const [imageDims, setImageDims] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [runningLabel, setRunningLabel] = useState<string>('');
  const [activeTab, setActiveTab] = useState<AlgoTab>('standard');
  const [resultsView, setResultsView] = useState<'findings' | 'metrics'>('findings');
  const [fullscreenImage, setFullscreenImage] = useState<string | null>(null);
  const [showSimulation, setShowSimulation] = useState<boolean>(false);
  const [crosshair, setCrosshair] = useState<CrossPos>(null);

  const [results, setResults] = useState<Record<AlgoTab, AnalyzeResult | null>>({
    standard: null,
    enhanced: null,
  });

  const currentResult = results[activeTab];
  const meta = TAB_META[activeTab];
  const bothDone = results.standard?.status === 'success' && results.enhanced?.status === 'success';
  const anyDone = results.standard?.status === 'success' || results.enhanced?.status === 'success';
  const regions = currentResult?.status === 'success' ? currentResult.detected_regions ?? [] : [];

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      const url = URL.createObjectURL(file);
      setSelectedFile(file);
      setOriginalPreview(url);
      setImageDims(null);
      setResults({ standard: null, enhanced: null });
      const probe = new Image();
      probe.onload = () => setImageDims(`${probe.naturalWidth} × ${probe.naturalHeight}`);
      probe.src = url;
    }
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

  const handleRunSingle = async (algo: AlgoTab) => {
    if (!selectedFile) return;
    setIsProcessing(true);
    setRunningLabel(`Running ${TAB_META[algo].label}…`);
    setActiveTab(algo);
    try {
      const result = await runAlgorithm(algo, selectedFile);
      setResults((prev) => ({ ...prev, [algo]: result }));
      setRunningLabel(
        result.status === 'success'
          ? `${TAB_META[algo].label} finished${result.iterations_used != null ? ` in ${result.iterations_used} iterations` : ''}`
          : ''
      );
    } catch (error) {
      console.error('Error analyzing image:', error);
      setRunningLabel('');
      alert('May error sa pag-connect sa backend. Siguraduhing tumatakbo ang FastAPI server.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRunBoth = async () => {
    if (!selectedFile) return;
    setIsProcessing(true);
    setRunningLabel('Running Standard and ESMA…');
    try {
      const [standard, enhanced] = await Promise.all([
        runAlgorithm('standard', selectedFile),
        runAlgorithm('enhanced', selectedFile),
      ]);
      setResults({ standard, enhanced });
      setRunningLabel('Both finished');
    } catch (error) {
      console.error('Error analyzing image:', error);
      setRunningLabel('');
      alert('May error sa pag-connect sa backend. Siguraduhing tumatakbo ang FastAPI server.');
    } finally {
      setIsProcessing(false);
    }
  };

  const findingsImage =
    currentResult?.status === 'success' ? currentResult.annotated_image ?? null : null;

  return (
    <div className="min-h-screen bg-[#F3F7F9] text-[#0F2738] font-['Figtree',sans-serif] pb-16">
      {/* Header */}
      <header className="bg-white border-b border-[#DCE6EC] px-4 sm:px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <span className="w-10 h-10 rounded-xl bg-[#EAF1F5] text-[#1F7AA3] flex items-center justify-center shrink-0">
              <ToothIcon />
            </span>
            <div>
              <h1 className={`text-xl font-bold leading-tight tracking-tight ${FONT_HEAD}`}>
                Standard vs Enhanced SMA
              </h1>
              <p className="text-[13px] text-[#4F6979] mt-0.5">Dental OPG segmentation comparison</p>
            </div>
          </div>
          <button
            onClick={() => setShowSimulation(true)}
            className="min-h-[42px] px-4 rounded-full border border-[#0F2738]/80 text-[13px] font-semibold
                       hover:bg-[#0F2738] hover:text-white transition-colors"
          >
            View algorithm simulation
          </button>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 sm:px-6 mt-5 flex flex-col gap-4">
        {/* Controls */}
        <section className={`${CARD} p-3 flex flex-wrap items-center gap-2.5`}>
          <div className="flex items-center gap-2.5 min-w-0 flex-1 basis-60">
            <div className="w-[72px] h-[35px] rounded-md bg-[#0C1A24] overflow-hidden shrink-0">
              {originalPreview && (
                <img src={originalPreview} alt="" className="w-full h-full object-cover" />
              )}
            </div>
            <div className="min-w-0">
              <p className="text-[13.5px] font-semibold truncate">
                {selectedFile ? selectedFile.name : 'No X-ray selected'}
              </p>
              <p className="text-xs text-[#4F6979]">
                {selectedFile ? imageDims ?? 'Loading…' : 'PNG or JPEG panoramic image'}
              </p>
            </div>
          </div>

          <label className={`${BTN} border border-[#DCE6EC] bg-white cursor-pointer hover:bg-[#EAF1F5]`}>
            Choose X-ray
            <input type="file" accept="image/png, image/jpeg" onChange={handleFileChange} className="sr-only" />
          </label>
          <button
            onClick={() => handleRunSingle('standard')}
            disabled={!selectedFile || isProcessing}
            className={`${BTN} text-white`}
            style={{ backgroundColor: TAB_META.standard.accent }}
          >
            Standard
          </button>
          <button
            onClick={() => handleRunSingle('enhanced')}
            disabled={!selectedFile || isProcessing}
            className={`${BTN} text-white`}
            style={{ backgroundColor: TAB_META.enhanced.accent }}
          >
            ESMA
          </button>
          <button
            onClick={handleRunBoth}
            disabled={!selectedFile || isProcessing}
            className={`${BTN} text-white bg-[#0F2738]`}
          >
            Both
          </button>

          {runningLabel && (
            <span className="text-[12.5px] text-[#4F6979]" aria-live="polite">{runningLabel}</span>
          )}

          {anyDone && (
            <div className="inline-flex bg-[#EAF1F5] rounded-full p-[3px] gap-0.5 sm:ml-auto">
              {(Object.keys(TAB_META) as AlgoTab[]).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  aria-pressed={activeTab === tab}
                  className={`min-h-[36px] px-3.5 rounded-full text-[13px] font-semibold transition-all ${
                    activeTab === tab ? 'bg-white text-[#0F2738] shadow-sm' : 'text-[#4F6979] hover:text-[#0F2738]'
                  }`}
                >
                  {TAB_META[tab].short}
                </button>
              ))}
            </div>
          )}
        </section>

        {/* Two frames */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Frame
            title="Original"
            onExpand={originalPreview ? () => setFullscreenImage(originalPreview) : undefined}
          >
            {originalPreview ? (
              <Stage src={originalPreview} alt="Original X-ray" pos={crosshair} onPos={setCrosshair} />
            ) : (
              <EmptyStage>No image uploaded</EmptyStage>
            )}
          </Frame>

          <Frame
            title={`Findings — ${meta.short}`}
            right={findingsImage && !isProcessing ? `${regions.length} findings` : undefined}
            onExpand={findingsImage && !isProcessing ? () => setFullscreenImage(findingsImage) : undefined}
            footer={
              !isProcessing && regions.length > 0 ? (
                <div className="flex flex-wrap gap-1.5">
                  {regions.map((r, i) => (
                    <span
                      key={i}
                      className="inline-flex items-center gap-1.5 text-[11.5px] font-semibold px-2.5 py-1 rounded-full bg-[#EAF1F5]"
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
            {isProcessing ? (
              <EmptyStage>
                <span
                  className="animate-spin rounded-full h-7 w-7 border-2 border-white/20 mr-3"
                  style={{ borderTopColor: meta.accent }}
                />
                Computing Kapur's entropy…
              </EmptyStage>
            ) : findingsImage ? (
              <Stage src={findingsImage} alt="Annotated findings" pos={crosshair} onPos={setCrosshair} />
            ) : currentResult?.status === 'error' ? (
              <EmptyStage>
                <span className="text-[#FF9B85] px-4 text-center">{currentResult.message}</span>
              </EmptyStage>
            ) : (
              <EmptyStage>Awaiting analysis</EmptyStage>
            )}
          </Frame>
        </div>

        {/* Results */}
        <section className={`${CARD} p-5 flex flex-col gap-3.5`}>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className={`text-[17px] font-semibold ${FONT_HEAD}`}>Results · {meta.label}</h2>
            <div className="inline-flex bg-[#EAF1F5] rounded-full p-[3px] gap-0.5" role="tablist">
              {(['findings', 'metrics'] as const).map((v) => (
                <button
                  key={v}
                  role="tab"
                  aria-selected={resultsView === v}
                  onClick={() => setResultsView(v)}
                  className={`min-h-[34px] px-3.5 rounded-full text-[13px] font-semibold capitalize transition-all ${
                    resultsView === v ? 'bg-white text-[#0F2738] shadow-sm' : 'text-[#4F6979] hover:text-[#0F2738]'
                  }`}
                >
                  {v}
                </button>
              ))}
            </div>
          </div>

          {currentResult?.status !== 'success' ? (
            <p className="text-sm text-[#4F6979] py-4">Awaiting analysis</p>
          ) : resultsView === 'metrics' ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-5">
              <MetricRow label="Thresholds" value={currentResult.thresholds?.join(', ') ?? '-'} />
              <MetricRow label="Kapur's entropy" value={currentResult.kapur_entropy_fitness?.toFixed(4) ?? '-'} />
              <MetricRow label="PSNR (dB)" value={currentResult.psnr != null ? currentResult.psnr.toFixed(4) : '∞'} />
              <MetricRow label="SSIM" value={currentResult.ssim?.toFixed(6) ?? '-'} />
              <MetricRow label="Runtime (s)" value={currentResult.runtime_sec?.toFixed(4) ?? '-'} />
              <MetricRow
                label="Iterations"
                value={currentResult.iterations_used != null ? `${currentResult.iterations_used} / 100` : '-'}
              />
            </div>
          ) : regions.length === 0 ? (
            <p className="text-sm text-[#4F6979]">No regions were flagged in this image.</p>
          ) : (
            <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {regions.map((region, idx) => {
                const accent = DIAGNOSIS_ACCENT[region.label] ?? DIAGNOSIS_DEFAULT_ACCENT;
                const area = region.area_px != null
                  ? `${Math.round(region.area_px).toLocaleString()} px`
                  : region.bbox ? `${region.bbox[2]} × ${region.bbox[3]} px box` : null;
                return (
                  <li
                    key={idx}
                    className="flex items-center gap-3 px-3 py-2.5 rounded-xl border border-[#DCE6EC]"
                  >
                    <span
                      className={`w-[38px] py-[3px] rounded-md text-center text-[13px] font-medium text-white shrink-0 ${FONT_MONO}`}
                      style={{ backgroundColor: accent }}
                    >
                      {region.quadrant}
                    </span>
                    <div className="min-w-0">
                      <p className="text-[13.5px] font-semibold truncate">{region.label}</p>
                      <p className="text-xs text-[#4F6979]">
                        {QUADRANT_LABELS[region.quadrant]}
                        {area != null && ` · ${area}`}
                      </p>
                    </div>
                    {region.confidence != null && (
                      <span className={`ml-auto text-[12.5px] text-[#4F6979] ${FONT_MONO}`}>
                        {Math.round(region.confidence * 100)}%
                      </span>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        {/* Standard vs Enhanced */}
        {bothDone && (
          <section className={`${CARD} p-5 flex flex-col gap-3.5`}>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className={`text-[17px] font-semibold ${FONT_HEAD}`}>Standard vs Enhanced</h2>
              <span className="text-[12.5px] text-[#4F6979]">Highlighted value = better result</span>
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
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-x-8 gap-y-4 text-[12.5px] text-[#4F6979]">
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
                      <span className="text-[#0F2738] font-semibold">{d}</span> — {DIAGNOSIS_INFO[d]}
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
                <li><b className="text-[#0F2738]">Original</b> — the uploaded X-ray, unmodified.</li>
                <li><b className="text-[#0F2738]">Findings</b> — segmentation output with flagged regions.</li>
                <li>Hover one frame and the crosshair appears on both.</li>
              </ul>
            </div>
          </div>
          <div className="bg-[#FFF8EC] border border-[#F1DDB8] text-[#6D4A12] text-[12.5px] rounded-xl px-4 py-3 mt-5">
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

const CARD = 'bg-white border border-[#DCE6EC] rounded-[22px] shadow-[0_12px_32px_rgba(15,39,56,0.07)]';
const BTN =
  'min-h-[42px] px-[18px] rounded-full text-[13.5px] font-semibold inline-flex items-center ' +
  'disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0';

function ToothIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M7 3c-2.5 0-4 2-4 4.5 0 3 1.5 4.5 2 7.5.4 2.6 1 6 2.6 6 1.4 0 1.6-3 2.4-5 .4-1 1.6-1 2 0 .8 2 1 5 2.4 5 1.6 0 2.2-3.4 2.6-6 .5-3 2-4.5 2-7.5C21 5 19.5 3 17 3c-2 0-3 1-5 1S9 3 7 3z" />
    </svg>
  );
}

function Frame({
  title,
  right,
  onExpand,
  footer,
  children,
}: {
  title: string;
  right?: string;
  onExpand?: () => void;
  footer?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <article className={`${CARD} overflow-hidden flex flex-col min-w-0`}>
      <div className="flex items-center justify-between gap-2 px-4 py-2.5 border-b border-[#DCE6EC]">
        <span className="text-[11px] font-semibold tracking-[0.11em] uppercase text-[#4F6979]">{title}</span>
        <span className="flex items-center gap-2.5 text-xs text-[#4F6979]">
          {right}
          <button
            onClick={onExpand}
            disabled={!onExpand}
            aria-label={`Expand ${title}`}
            className="w-9 h-9 rounded-lg border border-[#DCE6EC] inline-flex items-center justify-center
                       hover:bg-[#EAF1F5] disabled:opacity-35 disabled:hover:bg-transparent"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
              strokeLinecap="round" aria-hidden="true">
              <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />
            </svg>
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
    <div className="aspect-[2/1] rounded-xl bg-[#0C1A24] flex items-center justify-center text-sm text-[#8FA3AF]">
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
      className="relative rounded-xl overflow-hidden bg-[#0C1A24]"
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

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2 py-2.5 border-b border-[#DCE6EC]">
      <span className="text-[12.5px] text-[#4F6979]">{label}</span>
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
    <div className="flex flex-col gap-2 p-3.5 rounded-xl border border-[#DCE6EC]">
      <div>
        <p className="text-[13px] font-semibold">{label}</p>
        <p className="text-[11px] text-[#4F6979]">{lowerIsBetter ? 'lower is better' : 'higher is better'}</p>
      </div>
      <div className="flex items-center gap-2 text-[12.5px] text-[#4F6979]">
        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: TAB_META.standard.accent }} />
        Standard
        <span className={`ml-auto text-sm font-medium text-[#0F2738] ${FONT_MONO}`}>{a.toFixed(digits)}</span>
      </div>
      <div className="flex items-center gap-2 text-[12.5px] text-[#4F6979]">
        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: TAB_META.enhanced.accent }} />
        ESMA
        <span
          className={`ml-auto text-sm ${FONT_MONO} ${bWins ? 'font-semibold' : 'font-medium text-[#0F2738]'}`}
          style={bWins ? { color: ESMA_TEXT } : undefined}
        >
          {b.toFixed(digits)}
        </span>
      </div>
      <span
        className={`self-start mt-auto text-[11.5px] font-medium px-2 py-0.5 rounded-full bg-[#EAF1F5] ${FONT_MONO}`}
        style={{ color: bWins ? ESMA_TEXT : '#4F6979' }}
      >
        {sign}
        {Math.abs(diff).toFixed(digits)} ({relText})
      </span>
    </div>
  );
}

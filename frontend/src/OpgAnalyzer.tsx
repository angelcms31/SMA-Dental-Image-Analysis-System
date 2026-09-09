import React, { useState, ChangeEvent } from 'react';
import axios from 'axios';

type AlgoTab = 'standard' | 'enhanced';

interface DetectedRegion {
  quadrant: 'Q1' | 'Q2' | 'Q3' | 'Q4';
  label: string;
  bbox: [number, number, number, number];
  area_px: number;
  mean_intensity: number;
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
  convergence_curve?: number[];
  segmented_image?: string;
  annotated_image?: string;
  detected_regions?: DetectedRegion[];
  quadrant_summary?: Record<'Q1' | 'Q2' | 'Q3' | 'Q4', number>;
  disclaimer?: string;
}

const QUADRANT_LABELS: Record<string, string> = {
  Q1: 'Upper Left',
  Q2: 'Upper Right',
  Q3: 'Lower Right',
  Q4: 'Lower Left',
};

// Palette: Harvest Gold / Calico / Hampton / Sea Nymph / Smalt Blue
const DIAGNOSIS_ACCENT: Record<string, string> = {
  Caries: '#E1A36F',
  'Deep Caries': '#B8541F',
  Impacted: '#6F9F9C',
  'Periapical Lesion': '#577E89',
};
const DIAGNOSIS_DEFAULT_ACCENT = '#B0A480';

const DIAGNOSIS_INFO: Record<string, string> = {
  Caries: 'Tooth decay (cavity) affecting the enamel or dentin.',
  'Deep Caries': 'Decay extending close to or into the pulp -- more advanced than Caries.',
  Impacted: 'A tooth that has not fully erupted, often blocked by another tooth or bone.',
  'Periapical Lesion': 'Infection or inflammation at the root tip, usually from advanced decay or trauma.',
};

const TAB_META: Record<AlgoTab, { label: string; short: string; accent: string }> = {
  standard: { label: 'Standard SMA', short: 'Standard', accent: '#577E89' },
  enhanced: { label: 'Enhanced SMA', short: 'Enhanced', accent: '#E1A36F' },
};

export default function OpgAnalyzer() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [originalPreview, setOriginalPreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<AlgoTab>('standard');
  const [resultsView, setResultsView] = useState<'findings' | 'metrics'>('findings');
  const [fullscreenImage, setFullscreenImage] = useState<string | null>(null);

  const [results, setResults] = useState<Record<AlgoTab, AnalyzeResult | null>>({
    standard: null,
    enhanced: null,
  });

  const currentResult = results[activeTab];
  const meta = TAB_META[activeTab];
  const bothDone = results.standard?.status === 'success' && results.enhanced?.status === 'success';

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setOriginalPreview(URL.createObjectURL(file));
      setResults({ standard: null, enhanced: null });
    }
  };

  // Runs one algorithm end-to-end (core metrics + YOLO detector, merged).
  // Returns the result rather than setting state, so handleRunBoth can
  // await both algorithms in parallel from a single button.
  const runAlgorithm = async (algo: AlgoTab, file: File): Promise<AnalyzeResult> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('d', '4');
    formData.append('N', '30');
    formData.append('T', '100');

    const endpoint =
      algo === 'standard'
        ? 'http://localhost:8000/analyze/standard/'
        : 'http://localhost:8000/analyze/enhanced/';

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
      axios.post<AnalyzeResult>('http://localhost:8000/analyze/detect-teeth/', yoloFormData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      }),
    ]);

    return {
      ...metricsRes.data,
      annotated_image: yoloRes.data.status === 'success' ? yoloRes.data.annotated_image : metricsRes.data.annotated_image,
      detected_regions: yoloRes.data.status === 'success' ? yoloRes.data.detected_regions : metricsRes.data.detected_regions,
    };
  };

  // Runs a single algorithm and switches display to it immediately.
  const handleRunSingle = async (algo: AlgoTab) => {
    if (!selectedFile) return;
    setIsProcessing(true);
    setActiveTab(algo);
    try {
      const result = await runAlgorithm(algo, selectedFile);
      setResults((prev) => ({ ...prev, [algo]: result }));
    } catch (error) {
      console.error('Error analyzing image:', error);
      alert('May error sa pag-connect sa backend. Siguraduhing tumatakbo ang FastAPI server.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Runs Standard and Enhanced together. The tab switcher below then
  // changes which already-computed result is shown.
  const handleRunBoth = async () => {
    if (!selectedFile) return;
    setIsProcessing(true);
    try {
      const [standard, enhanced] = await Promise.all([
        runAlgorithm('standard', selectedFile),
        runAlgorithm('enhanced', selectedFile),
      ]);
      setResults({ standard, enhanced });
    } catch (error) {
      console.error('Error analyzing image:', error);
      alert('May error sa pag-connect sa backend. Siguraduhing tumatakbo ang FastAPI server.');
    } finally {
      setIsProcessing(false);
    }
  };

  const anyDone = results.standard?.status === 'success' || results.enhanced?.status === 'success';

  return (
    <div className="min-h-screen bg-[#F5F6F8] font-['Inter',sans-serif] pb-16">
      {/* Header -- compact, single row */}
      <div
        className="w-full px-6 py-4"
        style={{ background: 'linear-gradient(135deg, #3A5661 0%, #577E89 55%, #6F9F9C 100%)' }}
      >
        <div className="max-w-6xl mx-auto flex items-center justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-lg font-semibold text-white font-['Space_Grotesk',sans-serif] leading-tight">
              Standard vs Enhanced Slime Mould Algorithm
            </h1>
            <p className="text-white/60 text-xs mt-0.5">Dental OPG segmentation comparison</p>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 mt-6">
        {/* Upload card -- file picker + three run buttons (Standard /
            Enhanced / Both). Each single-algorithm button also switches
            the active tab to that algorithm; results differ per tab
            since each holds its own computed result. */}
        <div className="bg-white rounded-2xl shadow-[0_8px_30px_rgb(0,0,0,0.06)] border border-black/[0.04] p-5 mb-6">
          <div className="flex flex-wrap items-center gap-3">
            <input
              type="file"
              accept="image/png, image/jpeg"
              onChange={handleFileChange}
              className="text-sm text-slate-500
                         file:mr-3 file:py-2 file:px-4
                         file:rounded-full file:border-0
                         file:text-sm file:font-medium
                         file:bg-[#577E89] file:text-white
                         hover:file:bg-[#3A5661] cursor-pointer transition-colors"
            />

            <button
              onClick={() => handleRunSingle('standard')}
              disabled={!selectedFile || isProcessing}
              className="text-white px-5 py-2 rounded-full font-medium text-sm
                         disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
              style={{ backgroundColor: TAB_META.standard.accent }}
            >
              Standard
            </button>
            <button
              onClick={() => handleRunSingle('enhanced')}
              disabled={!selectedFile || isProcessing}
              className="text-white px-5 py-2 rounded-full font-medium text-sm
                         disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
              style={{ backgroundColor: TAB_META.enhanced.accent }}
            >
              ESMA
            </button>
            <button
              onClick={handleRunBoth}
              disabled={!selectedFile || isProcessing}
              className="text-white px-5 py-2 rounded-full font-medium text-sm
                         disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
              style={{ backgroundColor: '#3A5661' }}
            >
              Both
            </button>

            {isProcessing && <span className="text-xs text-slate-400">Running…</span>}

            {anyDone && (
              <div className="inline-flex bg-slate-100 rounded-full p-1 gap-1 shrink-0 ml-auto">
                {(Object.keys(TAB_META) as AlgoTab[]).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all ${
                      activeTab === tab ? 'text-white shadow-sm' : 'text-slate-500 hover:text-slate-700'
                    }`}
                    style={activeTab === tab ? { backgroundColor: TAB_META[tab].accent } : undefined}
                  >
                    {TAB_META[tab].short}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Two-column canvas: Original | Findings, wide aspect ratio
            matching panoramic X-rays for a bigger, clearer image */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div className="rounded-2xl overflow-hidden bg-[#14181F]">
            <div className="px-4 py-3 border-b border-white/10">
              <span className="text-white/50 text-xs font-medium tracking-wide">ORIGINAL</span>
            </div>
            <div className="aspect-[2/1] flex items-center justify-center p-2 relative group">
                {originalPreview ? (
                  <button
                    onClick={() => setFullscreenImage(originalPreview)}
                    className="relative w-full h-full flex items-center justify-center cursor-zoom-in"
                  >
                    <img
                      src={originalPreview}
                      alt="Original X-Ray"
                      className="max-w-full max-h-full object-contain rounded-lg"
                    />
                    <span className="absolute inset-0 flex items-center justify-center bg-black/0 group-hover:bg-black/30 transition-colors rounded-lg">
                      <span className="opacity-0 group-hover:opacity-100 text-white text-xs font-medium bg-black/60 px-3 py-1.5 rounded-full transition-opacity">
                        ⤢ Expand
                      </span>
                    </span>
                  </button>
                ) : (
                  <span className="text-white/30 text-sm">No image uploaded</span>
                )}
            </div>
          </div>

          <div className="rounded-2xl overflow-hidden bg-[#14181F]">
            <div className="px-4 py-3 border-b border-white/10 flex items-center justify-between">
              <span className="text-white/50 text-xs font-medium tracking-wide">
                FINDINGS — {meta.short.toUpperCase()}
              </span>
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: meta.accent }} />
            </div>
            <div className="aspect-[2/1] flex items-center justify-center p-2 relative group">
                {isProcessing ? (
                  <div className="flex flex-col items-center">
                    <div
                      className="animate-spin rounded-full h-8 w-8 border-2 border-white/20 mb-3"
                      style={{ borderTopColor: meta.accent }}
                    />
                    <span className="text-white/50 text-sm">Computing Kapur's entropy…</span>
                  </div>
                ) : currentResult?.status === 'success' && currentResult.annotated_image ? (
                  <button
                    onClick={() => setFullscreenImage(currentResult.annotated_image!)}
                    className="relative w-full h-full flex items-center justify-center cursor-zoom-in"
                  >
                    <img
                      src={currentResult.annotated_image}
                      alt="Annotated Findings"
                      className="max-w-full max-h-full object-contain rounded-lg"
                    />
                    <span className="absolute inset-0 flex items-center justify-center bg-black/0 group-hover:bg-black/30 transition-colors rounded-lg">
                      <span className="opacity-0 group-hover:opacity-100 text-white text-xs font-medium bg-black/60 px-3 py-1.5 rounded-full transition-opacity">
                        ⤢ Expand
                      </span>
                    </span>
                  </button>
                ) : currentResult?.status === 'error' ? (
                  <span className="text-red-400 text-sm px-4 text-center">{currentResult.message}</span>
                ) : (
                  <span className="text-white/30 text-sm">Awaiting analysis</span>
                )}
            </div>
          </div>
        </div>

        {/* Results panel -- full width below the images, Findings shown
            first by default, laid out as a grid instead of a long list */}
        <div className="rounded-2xl overflow-hidden bg-[#14181F] mb-6">
          <div className="px-4 py-3 border-b border-white/10 flex items-center justify-between">
            <span className="text-white/50 text-xs font-medium tracking-wide">RESULTS</span>
            <div className="inline-flex bg-white/5 rounded-full p-0.5 gap-0.5">
              <button
                onClick={() => setResultsView('findings')}
                className={`px-2.5 py-1 rounded-full text-[11px] font-medium transition-all ${
                  resultsView === 'findings' ? 'bg-white/15 text-white' : 'text-white/40 hover:text-white/70'
                }`}
              >
                Findings
              </button>
              <button
                onClick={() => setResultsView('metrics')}
                className={`px-2.5 py-1 rounded-full text-[11px] font-medium transition-all ${
                  resultsView === 'metrics' ? 'bg-white/15 text-white' : 'text-white/40 hover:text-white/70'
                }`}
              >
                Metrics
              </button>
            </div>
          </div>

          <div className="p-4">
            {currentResult?.status !== 'success' ? (
              <div className="py-8 flex items-center justify-center">
                <span className="text-white/30 text-sm">Awaiting analysis</span>
              </div>
            ) : resultsView === 'metrics' ? (
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <ResultStat label="Thresholds" value={currentResult.thresholds?.join(', ') ?? '-'} />
                <ResultStat label="Kapur's Entropy" value={currentResult.kapur_entropy_fitness?.toFixed(4) ?? '-'} />
                <ResultStat label="PSNR" value={currentResult.psnr != null ? currentResult.psnr.toFixed(4) : '∞'} />
                <ResultStat label="SSIM" value={currentResult.ssim?.toFixed(6) ?? '-'} />
                <ResultStat label="Runtime (s)" value={currentResult.runtime_sec?.toFixed(4) ?? '-'} />
              </div>
            ) : !currentResult.detected_regions || currentResult.detected_regions.length === 0 ? (
              <p className="text-sm text-white/40">No regions were flagged in this image.</p>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {currentResult.detected_regions.map((region, idx) => (
                  <div
                    key={idx}
                    className="flex items-center gap-2.5 bg-white/[0.04] rounded-md px-2.5 py-1.5 border-l-[3px]"
                    style={{ borderLeftColor: DIAGNOSIS_ACCENT[region.label] ?? DIAGNOSIS_DEFAULT_ACCENT }}
                  >
                    <div className="min-w-0">
                      <p className="text-xs font-semibold text-white/90 truncate">
                        {region.label}
                        {region.confidence != null && (
                          <span className="ml-1.5 font-normal text-white/40">
                            {Math.round(region.confidence * 100)}%
                          </span>
                        )}
                      </p>
                      <p className="text-[11px] text-white/40">
                        {QUADRANT_LABELS[region.quadrant]} · {region.area_px}px
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Fullscreen image modal */}
        {fullscreenImage && (
          <div
            onClick={() => setFullscreenImage(null)}
            className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-8 cursor-zoom-out"
          >
            <button
              onClick={() => setFullscreenImage(null)}
              className="absolute top-5 right-6 text-white/70 hover:text-white text-2xl leading-none"
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

        {/* Side-by-side comparison once both have been run */}
        {bothDone && (
          <div className="bg-white rounded-2xl shadow-[0_8px_30px_rgb(0,0,0,0.06)] border border-black/[0.04] p-5 md:p-6 mb-6">
            <h3 className="text-sm font-semibold text-slate-800 mb-4">Standard vs Enhanced</h3>
            <div className="space-y-4">
              <ComparisonBar
                label="Kapur's Entropy"
                a={results.standard!.kapur_entropy_fitness}
                b={results.enhanced!.kapur_entropy_fitness}
              />
              <ComparisonBar label="PSNR" a={results.standard!.psnr} b={results.enhanced!.psnr} />
              <ComparisonBar label="SSIM" a={results.standard!.ssim} b={results.enhanced!.ssim} />
              <ComparisonBar
                label="Runtime (s)"
                a={results.standard!.runtime_sec}
                b={results.enhanced!.runtime_sec}
                lowerIsBetter
              />
            </div>
            <div className="flex items-center gap-4 mt-5 pt-4 border-t border-slate-100 text-xs text-slate-400">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: TAB_META.standard.accent }} />
                Standard SMA
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: TAB_META.enhanced.accent }} />
                Enhanced SMA
              </span>
            </div>
          </div>
        )}

        {/* Legend -- includes the permanent, always-visible disclaimer at the bottom */}
        <div className="bg-white rounded-2xl shadow-[0_8px_30px_rgb(0,0,0,0.06)] border border-black/[0.04] p-5 mb-6">
          <h3 className="text-xs font-semibold text-slate-500 tracking-wide mb-3">HOW TO READ THIS</h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-x-8 gap-y-4">
            <div>
              <p className="text-xs font-medium text-slate-600 mb-2">Diagnosis colors</p>
              <div className="space-y-1.5">
                {(Object.keys(DIAGNOSIS_ACCENT) as Array<keyof typeof DIAGNOSIS_ACCENT>).map((d) => (
                  <div key={d} className="flex items-start gap-1.5 text-xs">
                    <span
                      className="w-2.5 h-2.5 rounded-full mt-0.5 shrink-0"
                      style={{ backgroundColor: DIAGNOSIS_ACCENT[d] }}
                    />
                    <span>
                      <span className="text-slate-700 font-medium">{d}</span>
                      <span className="text-slate-400"> — {DIAGNOSIS_INFO[d]}</span>
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <p className="text-xs font-medium text-slate-600 mb-2">Confidence %</p>
              <p className="text-xs text-slate-500 leading-relaxed">
                How certain the trained detection model is about that specific finding —
                not a measure of clinical certainty. Values are typically modest (25–45%)
                since this is an exploratory, non-validated feature.
              </p>
            </div>
            <div>
              <p className="text-xs font-medium text-slate-600 mb-2">Panels &amp; tabs</p>
              <ul className="text-xs text-slate-500 space-y-1 leading-relaxed">
                <li><strong className="text-slate-700">Original</strong> — the uploaded X-ray, unmodified.</li>
                <li><strong className="text-slate-700">Findings</strong> — segmentation output with flagged regions.</li>
                <li><strong className="text-slate-700">Results → Findings</strong> — list of flagged regions.</li>
                <li><strong className="text-slate-700">Results → Metrics</strong> — entropy, PSNR, SSIM, runtime.</li>
              </ul>
            </div>
          </div>

          <div className="bg-amber-50 border border-amber-200 text-amber-900 text-xs rounded-xl px-4 py-3 mt-5">
            <strong>⚠ Not a medical diagnosis.</strong> Research prototype for academic demonstration only.
            This output is NOT a clinical diagnosis. Findings are generated by a trained detection model
            and threshold-based segmentation, not a validated diagnostic tool. Please consult a licensed
            dentist for any actual diagnosis or treatment.
          </div>
        </div>

      </div>
    </div>
  );
}

function ResultStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between border-b border-white/5 pb-2.5">
      <span className="text-xs text-white/40">{label}</span>
      <span className="text-sm font-semibold text-white font-['Space_Grotesk',sans-serif]">{value}</span>
    </div>
  );
}

function ComparisonBar({
  label,
  a,
  b,
  lowerIsBetter = false,
}: {
  label: string;
  a?: number | null;
  b?: number | null;
  lowerIsBetter?: boolean;
}) {
  if (a == null || b == null) return null;
  const max = Math.max(a, b) || 1;
  const aPct = (a / max) * 100;
  const bPct = (b / max) * 100;
  const bWins = lowerIsBetter ? b < a : b > a;

  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="text-xs font-medium text-slate-600">{label}</span>
        <span className="text-[11px] text-slate-400">{lowerIsBetter ? 'lower is better' : 'higher is better'}</span>
      </div>
      <div className="flex items-center gap-2 mb-1">
        <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all"
            style={{ width: `${aPct}%`, backgroundColor: TAB_META.standard.accent }}
          />
        </div>
        <span className="text-xs text-slate-500 w-16 text-right tabular-nums">{a.toFixed(4)}</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all"
            style={{ width: `${bPct}%`, backgroundColor: TAB_META.enhanced.accent }}
          />
        </div>
        <span
          className="text-xs w-16 text-right tabular-nums font-medium"
          style={{ color: bWins ? TAB_META.enhanced.accent : '#94A3B8' }}
        >
          {b.toFixed(4)}
        </span>
      </div>
    </div>
  );
}
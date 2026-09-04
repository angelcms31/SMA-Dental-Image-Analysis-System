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

// Colored by DIAGNOSIS TYPE (not quadrant) so a finding's category is
// identifiable at a glance -- kept consistent with the backend's box
// colors (predict_yolo.py / overlay.py DIAGNOSIS_COLORS).
const DIAGNOSIS_ACCENT: Record<string, string> = {
  Caries: '#E8A33D',
  'Deep Caries': '#E8604C',
  Impacted: '#8B5FBF',
  'Periapical Lesion': '#4C8FE8',
};
const DIAGNOSIS_DEFAULT_ACCENT = '#94A3B8';

const DIAGNOSIS_INFO: Record<string, string> = {
  Caries: 'Tooth decay (cavity) affecting the enamel or dentin.',
  'Deep Caries': 'Decay extending close to or into the pulp -- more advanced than Caries.',
  Impacted: 'A tooth that has not fully erupted, often blocked by another tooth or bone.',
  'Periapical Lesion': 'Infection or inflammation at the root tip, usually from advanced decay or trauma.',
};

const TAB_META: Record<AlgoTab, { label: string; short: string; accent: string }> = {
  standard: { label: 'Standard SMA', short: 'Standard', accent: '#1B6E8C' },
  enhanced: { label: 'Enhanced SMA (ESMA)', short: 'Enhanced', accent: '#E8A33D' },
};

export default function OpgAnalyzer() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [originalPreview, setOriginalPreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<AlgoTab>('standard');
  const [resultsView, setResultsView] = useState<'metrics' | 'findings'>('metrics');
  const [fullscreenImage, setFullscreenImage] = useState<string | null>(null);

  const [results, setResults] = useState<Record<AlgoTab, AnalyzeResult | null>>({
    standard: null,
    enhanced: null,
  });

  const currentResult = results[activeTab];
  const meta = TAB_META[activeTab];

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setOriginalPreview(URL.createObjectURL(file));
      setResults({ standard: null, enhanced: null });
    }
  };

  const handleAnalyze = async (algo: AlgoTab) => {
    if (!selectedFile) return;
    setIsProcessing(true);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('d', '4');
    formData.append('N', '30');
    formData.append('T', '100');

    const endpoint =
      algo === 'standard'
        ? 'http://localhost:8000/analyze/standard/'
        : 'http://localhost:8000/analyze/enhanced/';

    // The trained YOLO detector (clean per-tooth boxes) uses SEPARATE
    // weights from the core SMA/ESMA comparison -- one model per
    // preprocessing algorithm. We call both endpoints and merge:
    // real entropy/PSNR/SSIM/runtime numbers from the SMA/ESMA endpoint,
    // but the clean trained-detector overlay for the visual.
    const yoloFormData = new FormData();
    yoloFormData.append('file', selectedFile);
    yoloFormData.append('conf', '0.25');
    yoloFormData.append('iou', '0.35');
    yoloFormData.append('use_esma', 'true');
    yoloFormData.append('sma_algorithm', algo);
    yoloFormData.append(
      'weights_path',
      algo === 'standard'
        ? 'runs_yolo/train_standard_v2/weights/best.pt'
        : 'runs_yolo/train_enhanced_v2/weights/best.pt'
    );

    try {
      const [metricsRes, yoloRes] = await Promise.all([
        axios.post<AnalyzeResult>(endpoint, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        }),
        axios.post<AnalyzeResult>('http://localhost:8000/analyze/detect-teeth/', yoloFormData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        }),
      ]);

      const merged: AnalyzeResult = {
        ...metricsRes.data,
        // prefer the YOLO detector's cleaner overlay + findings for display,
        // but fall back to the SMA/ESMA endpoint's own overlay if the YOLO
        // call failed for some reason (e.g. weights not found yet)
        annotated_image: yoloRes.data.status === 'success' ? yoloRes.data.annotated_image : metricsRes.data.annotated_image,
        detected_regions: yoloRes.data.status === 'success' ? yoloRes.data.detected_regions : metricsRes.data.detected_regions,
      };
      setResults((prev) => ({ ...prev, [algo]: merged }));
    } catch (error) {
      console.error('Error analyzing image:', error);
      alert('May error sa pag-connect sa backend. Siguraduhing tumatakbo ang FastAPI server.');
    } finally {
      setIsProcessing(false);
    }
  };

  const bothDone = results.standard?.status === 'success' && results.enhanced?.status === 'success';

  return (
    <div className="min-h-screen bg-[#F5F6F8] font-['Inter',sans-serif] pb-16">
      {/* Header */}
      <div
        className="w-full px-6 py-10 md:py-14"
        style={{ background: 'linear-gradient(135deg, #14324A 0%, #1B6E8C 55%, #1E8C82 100%)' }}
      >
        <div className="max-w-5xl mx-auto">
          <p className="text-white/60 text-xs font-medium tracking-wide mb-2">DENTAL OPG SEGMENTATION</p>
          <h1 className="text-3xl md:text-[2.15rem] font-semibold text-white font-['Space_Grotesk',sans-serif] leading-tight">
            Standard vs Enhanced Slime Mould Algorithm
          </h1>
          <p className="text-white/70 mt-2 text-sm md:text-base max-w-xl">
            Upload a panoramic X-ray to compare threshold segmentation quality between
            the original and proposed algorithms.
          </p>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-6 -mt-6">
        {/* Upload card -- compact single-row layout */}
        <div className="bg-white rounded-2xl shadow-[0_8px_30px_rgb(0,0,0,0.06)] border border-black/[0.04] p-5 mb-6">
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-3">
              <input
                type="file"
                accept="image/png, image/jpeg"
                onChange={handleFileChange}
                className="text-sm text-slate-500
                           file:mr-3 file:py-2 file:px-4
                           file:rounded-full file:border-0
                           file:text-sm file:font-medium
                           file:bg-[#1B6E8C] file:text-white
                           hover:file:bg-[#155A73] cursor-pointer transition-colors"
              />
            </div>

            <div className="inline-flex bg-slate-100 rounded-full p-1 gap-1 shrink-0">
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

            <button
              onClick={() => handleAnalyze(activeTab)}
              disabled={!selectedFile || isProcessing}
              className="text-white px-6 py-2 rounded-full font-medium text-sm
                         disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
              style={{ backgroundColor: meta.accent }}
            >
              {isProcessing ? `Running…` : `Run ${meta.label}`}
            </button>
          </div>
        </div>

        {/* Non-clinical disclaimer */}
        {currentResult?.status === 'success' && currentResult.disclaimer && (
          <div className="mb-6 bg-amber-50 border border-amber-200 text-amber-900 text-xs rounded-xl px-4 py-3">
            <strong>⚠ Not a medical diagnosis.</strong> {currentResult.disclaimer}
          </div>
        )}

        {/* Two-column canvas: Original + Findings stacked on the left,
            Results spanning the full height on the right */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6 items-stretch">
          <div className="flex flex-col gap-4">
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

          {/* Results panel -- toggles between Metrics and Findings-list views,
              similar to a preview/code style switch. Spans the full height
              of the left column via items-stretch on the parent grid. */}
          <div className="rounded-2xl overflow-hidden bg-[#14181F] flex flex-col">
            <div className="px-4 py-3 border-b border-white/10 flex items-center justify-between shrink-0">
              <span className="text-white/50 text-xs font-medium tracking-wide">RESULTS</span>
              <div className="inline-flex bg-white/5 rounded-full p-0.5 gap-0.5">
                <button
                  onClick={() => setResultsView('metrics')}
                  className={`px-2.5 py-1 rounded-full text-[11px] font-medium transition-all ${
                    resultsView === 'metrics' ? 'bg-white/15 text-white' : 'text-white/40 hover:text-white/70'
                  }`}
                >
                  Metrics
                </button>
                <button
                  onClick={() => setResultsView('findings')}
                  className={`px-2.5 py-1 rounded-full text-[11px] font-medium transition-all ${
                    resultsView === 'findings' ? 'bg-white/15 text-white' : 'text-white/40 hover:text-white/70'
                  }`}
                >
                  Findings
                </button>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-4">
              {currentResult?.status !== 'success' ? (
                <div className="h-full flex items-center justify-center">
                  <span className="text-white/30 text-sm">Awaiting analysis</span>
                </div>
              ) : resultsView === 'metrics' ? (
                <div className="space-y-3">
                  <ResultStat label="Thresholds" value={currentResult.thresholds?.join(', ') ?? '-'} />
                  <ResultStat label="Kapur's Entropy" value={currentResult.kapur_entropy_fitness?.toFixed(4) ?? '-'} />
                  <ResultStat label="PSNR" value={currentResult.psnr != null ? currentResult.psnr.toFixed(4) : '∞'} />
                  <ResultStat label="SSIM" value={currentResult.ssim?.toFixed(6) ?? '-'} />
                  <ResultStat label="Runtime (s)" value={currentResult.runtime_sec?.toFixed(4) ?? '-'} />
                </div>
              ) : !currentResult.detected_regions || currentResult.detected_regions.length === 0 ? (
                <p className="text-sm text-white/40">No regions were flagged in this image.</p>
              ) : (
                <div className="space-y-1.5">
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
        </div>

        {/* Fullscreen image modal -- click Original or Findings to expand */}
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

        {/* Legend / quick guide -- explains the diagnosis color coding and
            how to read the panels, so first-time viewers (panel, dentist,
            defense audience) don't have to ask */}
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
                since this is an exploratory, non-validated feature. The region's quadrant
                (e.g. Upper Left) is shown as text under each finding.
              </p>
            </div>
            <div>
              <p className="text-xs font-medium text-slate-600 mb-2">Panels &amp; tabs</p>
              <ul className="text-xs text-slate-500 space-y-1 leading-relaxed">
                <li><strong className="text-slate-700">Original</strong> — the uploaded X-ray, unmodified.</li>
                <li><strong className="text-slate-700">Findings</strong> — segmentation output with flagged regions.</li>
                <li><strong className="text-slate-700">Results → Metrics</strong> — entropy, PSNR, SSIM, runtime.</li>
                <li><strong className="text-slate-700">Results → Findings</strong> — list of flagged regions.</li>
                <li><strong className="text-slate-700">Standard / Enhanced</strong> — which SMA variant produced the thresholds.</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Side-by-side comparison once both have been run */}
        {bothDone && (
          <div className="bg-white rounded-2xl shadow-[0_8px_30px_rgb(0,0,0,0.06)] border border-black/[0.04] p-5 md:p-6">
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
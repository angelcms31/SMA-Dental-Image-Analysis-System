import React, { useState, ChangeEvent } from 'react';
import axios from 'axios';

type AlgoTab = 'standard' | 'enhanced';

interface DetectedRegion {
  quadrant: 'Q1' | 'Q2' | 'Q3' | 'Q4';
  label: string;
  bbox: [number, number, number, number];
  area_px: number;
  mean_intensity: number;
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

const QUADRANT_DOT_COLOR: Record<string, string> = {
  Q1: 'bg-green-600',
  Q2: 'bg-red-600',
  Q3: 'bg-blue-600',
  Q4: 'bg-yellow-500',
};

export default function OpgAnalyzer() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [originalPreview, setOriginalPreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<AlgoTab>('standard');

  // Hiwalay na resulta per tab, para hindi mawala yung Standard result
  // kapag lumipat ka papuntang Enhanced (kailangan mo silang dalawa
  // magkasabay para sa comparative analysis mo).
  const [results, setResults] = useState<Record<AlgoTab, AnalyzeResult | null>>({
    standard: null,
    enhanced: null,
  });

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
    // default params -- pwede mong ilagay bilang inputs sa UI later kung
    // gusto mo i-vary yung d/N/T per run for the ablation study
    formData.append('d', '4');
    formData.append('N', '30');
    formData.append('T', '100');

    const endpoint =
      algo === 'standard'
        ? 'http://localhost:8000/analyze/standard/'
        : 'http://localhost:8000/analyze/enhanced/';

    try {
      const response = await axios.post<AnalyzeResult>(endpoint, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResults((prev) => ({ ...prev, [algo]: response.data }));
    } catch (error) {
      console.error('Error analyzing image:', error);
      alert('May error sa pag-connect sa backend. Siguraduhing tumatakbo ang FastAPI server.');
    } finally {
      setIsProcessing(false);
    }
  };

  const currentResult = results[activeTab];

  return (
    <div className="min-h-screen bg-gray-50 p-8 flex flex-col items-center font-sans">
      <h1 className="text-3xl font-bold text-slate-800 mb-2">
        Automated Dental OPG Analysis
      </h1>
      <p className="text-slate-500 mb-8">
        Standard SMA vs Enhanced Slime Mould Algorithm (ESMA) — Comparative Analysis
      </p>

      <div className="bg-white p-6 rounded-lg shadow-md w-full max-w-4xl border border-gray-100">

        {/* Upload Input */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Upload Patient OPG Image (Grayscale X-Ray)
          </label>
          <input
            type="file"
            accept="image/png, image/jpeg"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-500
                       file:mr-4 file:py-2 file:px-4
                       file:rounded-md file:border-0
                       file:text-sm file:font-semibold
                       file:bg-blue-50 file:text-blue-700
                       hover:file:bg-blue-100 cursor-pointer"
          />
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-200 mb-6">
          <button
            onClick={() => setActiveTab('standard')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'standard'
                ? 'border-blue-600 text-blue-700'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            Standard SMA
          </button>
          <button
            onClick={() => setActiveTab('enhanced')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'enhanced'
                ? 'border-blue-600 text-blue-700'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            Enhanced SMA (ESMA)
          </button>
        </div>

        {/* Analyze Button */}
        <button
          onClick={() => handleAnalyze(activeTab)}
          disabled={!selectedFile || isProcessing}
          className="w-full md:w-auto bg-blue-600 text-white px-8 py-2.5 rounded-md font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isProcessing
            ? `${activeTab === 'standard' ? 'Standard SMA' : 'ESMA'} is Computing Thresholds...`
            : `Run ${activeTab === 'standard' ? 'Standard SMA' : 'ESMA'} Segmentation`}
        </button>

        {/* Non-clinical disclaimer -- always visible once a result exists.
            Required because patients, not just the thesis panel, may see
            this screen. Do not remove this without replacing it with
            something equally visible. */}
        {currentResult?.status === 'success' && currentResult.disclaimer && (
          <div className="mb-4 bg-amber-50 border border-amber-300 text-amber-900 text-xs rounded-md px-3 py-2">
            <strong>⚠ Not a medical diagnosis.</strong> {currentResult.disclaimer}
          </div>
        )}

        {/* Image Preview Grid */}
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-8">

          <div className="flex flex-col">
            <h3 className="text-lg font-semibold text-gray-800 mb-2 border-b pb-2">Original OPG</h3>
            <div className="bg-gray-100 min-h-[300px] flex items-center justify-center rounded border border-gray-200 overflow-hidden">
              {originalPreview ? (
                <img src={originalPreview} alt="Original X-Ray" className="w-full h-auto object-contain" />
              ) : (
                <span className="text-gray-400 text-sm">No image uploaded</span>
              )}
            </div>
          </div>

          <div className="flex flex-col">
            <h3 className="text-lg font-semibold text-gray-800 mb-2 border-b pb-2">
              Findings Overview ({activeTab === 'standard' ? 'Standard SMA' : 'ESMA'})
            </h3>
            <div className="bg-gray-100 min-h-[300px] flex items-center justify-center rounded border border-gray-200 overflow-hidden">
              {isProcessing ? (
                <div className="flex flex-col items-center">
                  <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600 mb-3"></div>
                  <span className="text-gray-500 text-sm">Running Kapur's Entropy...</span>
                </div>
              ) : currentResult?.status === 'success' && currentResult.annotated_image ? (
                <img src={currentResult.annotated_image} alt="Annotated Findings" className="w-full h-auto object-contain" />
              ) : currentResult?.status === 'error' ? (
                <span className="text-red-500 text-sm px-4 text-center">{currentResult.message}</span>
              ) : (
                <span className="text-gray-400 text-sm">Awaiting analysis</span>
              )}
            </div>
          </div>

        </div>

        {/* Patient-friendly findings summary */}
        {currentResult?.status === 'success' && currentResult.detected_regions && (
          <div className="mt-6 border-t pt-4">
            <h3 className="text-lg font-semibold text-gray-800 mb-3">What was flagged</h3>
            {currentResult.detected_regions.length === 0 ? (
              <p className="text-sm text-gray-500">No regions were flagged in this image.</p>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {currentResult.detected_regions.map((region, idx) => (
                  <div key={idx} className="flex items-center gap-3 bg-gray-50 border border-gray-100 rounded-md px-3 py-2">
                    <span className={`w-3 h-3 rounded-full flex-shrink-0 ${QUADRANT_DOT_COLOR[region.quadrant]}`} />
                    <div>
                      <p className="text-sm font-semibold text-slate-800">{region.label}</p>
                      <p className="text-xs text-gray-500">
                        {QUADRANT_LABELS[region.quadrant]} · area {region.area_px}px
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Accuracy metrics -- para sa Chapter 4 comparative analysis */}
        {currentResult?.status === 'success' && (
          <div className="mt-6 border-t pt-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Performance Metrics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <MetricBox label="Thresholds" value={currentResult.thresholds?.join(', ') ?? '-'} />
              <MetricBox label="Kapur's Entropy" value={currentResult.kapur_entropy_fitness?.toFixed(4) ?? '-'} />
              <MetricBox label="PSNR" value={currentResult.psnr != null ? currentResult.psnr.toFixed(4) : '∞'} />
              <MetricBox label="SSIM" value={currentResult.ssim?.toFixed(6) ?? '-'} />
              <MetricBox label="Runtime (s)" value={currentResult.runtime_sec?.toFixed(4) ?? '-'} />
            </div>
          </div>
        )}

        {/* Side-by-side comparison once both tabs have been run */}
        {results.standard?.status === 'success' && results.enhanced?.status === 'success' && (
          <div className="mt-6 border-t pt-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Standard vs Enhanced — Comparison</h3>
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="text-left border-b">
                  <th className="py-1 pr-4">Metric</th>
                  <th className="py-1 pr-4">Standard SMA</th>
                  <th className="py-1 pr-4">Enhanced SMA</th>
                </tr>
              </thead>
              <tbody>
                <ComparisonRow
                  label="Kapur's Entropy"
                  a={results.standard.kapur_entropy_fitness}
                  b={results.enhanced.kapur_entropy_fitness}
                />
                <ComparisonRow label="PSNR" a={results.standard.psnr} b={results.enhanced.psnr} />
                <ComparisonRow label="SSIM" a={results.standard.ssim} b={results.enhanced.ssim} />
                <ComparisonRow label="Runtime (s)" a={results.standard.runtime_sec} b={results.enhanced.runtime_sec} />
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function MetricBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-50 rounded-md px-3 py-2 border border-gray-100">
      <p className="text-xs text-gray-500">{label}</p>
      <p className="text-sm font-semibold text-slate-800 break-words">{value}</p>
    </div>
  );
}

function ComparisonRow({ label, a, b }: { label: string; a?: number | null; b?: number | null }) {
  return (
    <tr className="border-b last:border-0">
      <td className="py-1 pr-4 font-medium text-gray-700">{label}</td>
      <td className="py-1 pr-4">{a != null ? a.toFixed(4) : '-'}</td>
      <td className="py-1 pr-4">{b != null ? b.toFixed(4) : '-'}</td>
    </tr>
  );
}
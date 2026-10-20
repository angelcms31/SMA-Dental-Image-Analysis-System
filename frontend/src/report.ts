// Builds the downloadable PDF report (original OPG, findings images,
// metrics and findings tables) entirely in the browser with jsPDF.
import { jsPDF } from 'jspdf';

interface ReportRegion {
  quadrant: string;
  label: string;
  confidence?: number;
}

export interface ReportResult {
  status: 'success' | 'error';
  thresholds?: number[];
  kapur_entropy_fitness?: number;
  psnr?: number | null;
  ssim?: number;
  runtime_sec?: number;
  iterations_used?: number;
  annotated_image?: string;
  detected_regions?: ReportRegion[];
}

interface ReportInput {
  fileName: string;
  imageDims: string | null;
  originalUrl: string;
  standard: ReportResult | null;
  enhanced: ReportResult | null;
  quadrantLabels: Record<string, string>;
}

const PAGE_W = 210;
const PAGE_H = 297;
const MARGIN = 15;
const CONTENT_W = PAGE_W - MARGIN * 2;
const BOTTOM = PAGE_H - 18;

const INK: [number, number, number] = [15, 39, 56];
const MUTED: [number, number, number] = [79, 105, 121];
const LINE: [number, number, number] = [220, 230, 236];
const BLUE: [number, number, number] = [31, 122, 163];
const ORANGE: [number, number, number] = [200, 113, 31];

// Re-encode as JPEG (and cap the width) so a 3 MB PNG X-ray doesn't
// turn into a huge PDF.
function loadAsJpeg(src: string, maxWidth = 1800): Promise<{ data: string; w: number; h: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, maxWidth / img.naturalWidth);
      const w = Math.round(img.naturalWidth * scale);
      const h = Math.round(img.naturalHeight * scale);
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d')!;
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, w, h);
      ctx.drawImage(img, 0, 0, w, h);
      resolve({ data: canvas.toDataURL('image/jpeg', 0.88), w, h });
    };
    img.onerror = () => reject(new Error('Could not load image for the report'));
    img.src = src;
  });
}

const fmt = (v: number | null | undefined, digits: number) => (v == null ? '-' : v.toFixed(digits));

export async function downloadPdfReport(input: ReportInput, outName: string): Promise<void> {
  const { fileName, imageDims, originalUrl, standard, enhanced, quadrantLabels } = input;
  const doc = new jsPDF({ unit: 'mm', format: 'a4' });
  let y = MARGIN;

  const ensureSpace = (need: number) => {
    if (y + need > BOTTOM) {
      doc.addPage();
      y = MARGIN;
    }
  };

  const sectionTitle = (text: string, color: [number, number, number] = INK) => {
    ensureSpace(12);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11.5);
    doc.setTextColor(...color);
    doc.text(text, MARGIN, y + 4);
    y += 8;
  };

  const addImage = async (src: string, caption: string, color?: [number, number, number]) => {
    const img = await loadAsJpeg(src);
    const h = (CONTENT_W * img.h) / img.w;
    ensureSpace(h + 12);
    sectionTitle(caption, color);
    doc.setFillColor(12, 26, 36);
    doc.roundedRect(MARGIN, y, CONTENT_W, h, 2, 2, 'F');
    doc.addImage(img.data, 'JPEG', MARGIN, y, CONTENT_W, h);
    y += h + 7;
  };

  // ---- header ----
  doc.setFillColor(...BLUE);
  doc.rect(0, 0, PAGE_W, 4, 'F');
  y += 2;
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(17);
  doc.setTextColor(...INK);
  doc.text('ESMA Dental Analyzer', MARGIN, y + 5);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9.5);
  doc.setTextColor(...MUTED);
  doc.text('Analysis report - Standard SMA vs Enhanced SMA (ESMA)', MARGIN, y + 11);
  doc.text(new Date().toLocaleString(), PAGE_W - MARGIN, y + 5, { align: 'right' });
  y += 17;

  const info: [string, string][] = [
    ['Image', fileName],
    ['Image size', imageDims ? `${imageDims} px` : '-'],
    ['Parameters', 'd = 4 threshold levels, N = 30 agents, T = 100 iterations'],
  ];
  doc.setFontSize(9.5);
  info.forEach(([k, v]) => {
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...MUTED);
    doc.text(k, MARGIN, y);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(...INK);
    doc.text(v, MARGIN + 28, y);
    y += 5.5;
  });
  y += 2;
  doc.setDrawColor(...LINE);
  doc.line(MARGIN, y, PAGE_W - MARGIN, y);
  y += 6;

  // ---- images ----
  await addImage(originalUrl, 'Original OPG');
  if (standard?.status === 'success' && standard.annotated_image) {
    await addImage(standard.annotated_image, 'Findings - Standard SMA', BLUE);
  }
  if (enhanced?.status === 'success' && enhanced.annotated_image) {
    await addImage(enhanced.annotated_image, 'Findings - Enhanced SMA (ESMA)', ORANGE);
  }

  // ---- metrics table ----
  const std = standard?.status === 'success' ? standard : null;
  const esma = enhanced?.status === 'success' ? enhanced : null;
  const metricRows: { label: string; a: string; b: string; better?: 'a' | 'b' }[] = [];
  const pick = (a?: number | null, b?: number | null, lower = false): 'a' | 'b' | undefined => {
    if (a == null || b == null || a === b) return undefined;
    return (lower ? b < a : b > a) ? 'b' : 'a';
  };
  metricRows.push({ label: 'Thresholds', a: std?.thresholds?.join(', ') ?? '-', b: esma?.thresholds?.join(', ') ?? '-' });
  metricRows.push({ label: "Kapur's entropy", a: fmt(std?.kapur_entropy_fitness, 4), b: fmt(esma?.kapur_entropy_fitness, 4),
    better: pick(std?.kapur_entropy_fitness, esma?.kapur_entropy_fitness) });
  metricRows.push({ label: 'PSNR (dB)', a: fmt(std?.psnr, 4), b: fmt(esma?.psnr, 4), better: pick(std?.psnr, esma?.psnr) });
  metricRows.push({ label: 'SSIM', a: fmt(std?.ssim, 4), b: fmt(esma?.ssim, 4), better: pick(std?.ssim, esma?.ssim) });
  metricRows.push({ label: 'Runtime (s)', a: fmt(std?.runtime_sec, 4), b: fmt(esma?.runtime_sec, 4),
    better: pick(std?.runtime_sec, esma?.runtime_sec, true) });
  metricRows.push({ label: 'Iterations used', a: std?.iterations_used?.toString() ?? '-', b: esma?.iterations_used?.toString() ?? '-',
    better: pick(std?.iterations_used, esma?.iterations_used, true) });
  metricRows.push({ label: 'Findings', a: std ? String(std.detected_regions?.length ?? 0) : '-',
    b: esma ? String(esma.detected_regions?.length ?? 0) : '-' });

  const ROW = 7;
  ensureSpace(14 + ROW * (metricRows.length + 1));
  sectionTitle('Segmentation metrics');
  const mCols = [MARGIN, MARGIN + 60, MARGIN + 120];
  const drawHeader = (cols: number[], labels: string[]) => {
    doc.setFillColor(234, 241, 245);
    doc.rect(MARGIN, y, CONTENT_W, ROW, 'F');
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(9);
    doc.setTextColor(...MUTED);
    labels.forEach((l, i) => doc.text(l, cols[i] + 2, y + 4.8));
    y += ROW;
  };
  drawHeader(mCols, ['Metric', 'Standard SMA', 'ESMA']);
  metricRows.forEach((r) => {
    doc.setFontSize(9.5);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(...INK);
    doc.text(r.label, mCols[0] + 2, y + 4.8);
    doc.setFont('helvetica', r.better === 'a' ? 'bold' : 'normal');
    doc.text(r.a, mCols[1] + 2, y + 4.8);
    doc.setFont('helvetica', r.better === 'b' ? 'bold' : 'normal');
    if (r.better === 'b') doc.setTextColor(...ORANGE);
    doc.text(r.b, mCols[2] + 2, y + 4.8);
    doc.setDrawColor(...LINE);
    doc.line(MARGIN, y + ROW, PAGE_W - MARGIN, y + ROW);
    y += ROW;
  });
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8);
  doc.setTextColor(...MUTED);
  doc.text('Bold = better value. Higher is better for entropy, PSNR and SSIM; lower is better for runtime and iterations.',
    MARGIN, y + 4.5);
  y += 10;

  // ---- findings table ----
  const findingRows: string[][] = [];
  ([['Standard SMA', std], ['ESMA', esma]] as const).forEach(([name, r]) => {
    (r?.detected_regions ?? []).forEach((d) => {
      findingRows.push([
        name,
        d.quadrant,
        quadrantLabels[d.quadrant] ?? '',
        d.label,
        d.confidence != null ? `${Math.round(d.confidence * 100)}%` : '-',
      ]);
    });
  });
  ensureSpace(14 + ROW * 2);
  sectionTitle('Findings');
  const fCols = [MARGIN, MARGIN + 34, MARGIN + 56, MARGIN + 90, MARGIN + 150];
  drawHeader(fCols, ['Algorithm', 'Quadrant', 'Side (patient)', 'Diagnosis', 'Confidence']);
  if (findingRows.length === 0) {
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9.5);
    doc.setTextColor(...MUTED);
    doc.text('No regions were flagged.', MARGIN + 2, y + 4.8);
    y += ROW;
  }
  findingRows.forEach((row) => {
    if (y + ROW > BOTTOM) {
      doc.addPage();
      y = MARGIN;
      drawHeader(fCols, ['Algorithm', 'Quadrant', 'Side (patient)', 'Diagnosis', 'Confidence']);
    }
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9.5);
    doc.setTextColor(...INK);
    row.forEach((cell, i) => doc.text(cell, fCols[i] + 2, y + 4.8));
    doc.setDrawColor(...LINE);
    doc.line(MARGIN, y + ROW, PAGE_W - MARGIN, y + ROW);
    y += ROW;
  });
  doc.setFontSize(8);
  doc.setTextColor(...MUTED);
  doc.text('Confidence is the detector\'s certainty for that box, not clinical certainty.', MARGIN, y + 4.5);
  y += 10;

  // ---- disclaimer ----
  const disclaimer =
    'NOT A MEDICAL DIAGNOSIS. Research prototype for academic demonstration only. Findings are generated by a ' +
    'trained detection model and threshold-based segmentation, not a validated diagnostic tool. Please consult a ' +
    'licensed dentist for any actual diagnosis or treatment.';
  const lines = doc.splitTextToSize(disclaimer, CONTENT_W - 8);
  const boxH = lines.length * 4.2 + 6;
  ensureSpace(boxH);
  doc.setFillColor(255, 248, 236);
  doc.setDrawColor(241, 221, 184);
  doc.roundedRect(MARGIN, y, CONTENT_W, boxH, 2, 2, 'FD');
  doc.setFontSize(8.5);
  doc.setTextColor(109, 74, 18);
  doc.text(lines, MARGIN + 4, y + 5);

  // ---- footer on every page ----
  const pages = doc.getNumberOfPages();
  for (let p = 1; p <= pages; p++) {
    doc.setPage(p);
    doc.setFontSize(8);
    doc.setTextColor(...MUTED);
    doc.text(`ESMA Dental Analyzer - ${fileName}`, MARGIN, PAGE_H - 9);
    doc.text(`Page ${p} of ${pages}`, PAGE_W - MARGIN, PAGE_H - 9, { align: 'right' });
  }

  doc.save(outName);
}

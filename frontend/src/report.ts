// Builds the downloadable PDF report (original OPG, highlighted image,
// findings table and technical details) entirely in the browser with jsPDF.
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
  result: ReportResult;
  quadrantLabels: Record<string, string>;
}

const PAGE_W = 210;
const PAGE_H = 297;
const MARGIN = 15;
const CONTENT_W = PAGE_W - MARGIN * 2;
const BOTTOM = PAGE_H - 18;

const INK: [number, number, number] = [31, 41, 55];
const MUTED: [number, number, number] = [78, 97, 91];
const LINE: [number, number, number] = [211, 236, 221];
const TEAL: [number, number, number] = [15, 118, 110];

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
  const { fileName, imageDims, originalUrl, result, quadrantLabels } = input;
  const ok = result.status === 'success' ? result : null;
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
  doc.setFillColor(...TEAL);
  doc.rect(0, 0, PAGE_W, 4, 'F');
  y += 2;
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(17);
  doc.setTextColor(...INK);
  doc.text('ESMA Dental Analyzer', MARGIN, y + 5);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9.5);
  doc.setTextColor(...MUTED);
  doc.text('OPG analysis report', MARGIN, y + 11);
  doc.text(new Date().toLocaleString(), PAGE_W - MARGIN, y + 5, { align: 'right' });
  y += 17;

  const info: [string, string][] = [
    ['Image', fileName],
    ['Image size', imageDims ? `${imageDims} px` : '-'],
    ['Areas flagged', ok ? String(ok.detected_regions?.length ?? 0) : '-'],
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
  await addImage(originalUrl, 'Your OPG');
  if (ok?.annotated_image) {
    await addImage(ok.annotated_image, 'Highlighted areas', TEAL);
  }

  const ROW = 7;
  const drawHeader = (cols: number[], labels: string[]) => {
    doc.setFillColor(234, 241, 245);
    doc.rect(MARGIN, y, CONTENT_W, ROW, 'F');
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(9);
    doc.setTextColor(...MUTED);
    labels.forEach((l, i) => doc.text(l, cols[i] + 2, y + 4.8));
    y += ROW;
  };
  const drawRows = (cols: number[], headers: string[], rows: string[][]) => {
    rows.forEach((row) => {
      if (y + ROW > BOTTOM) {
        doc.addPage();
        y = MARGIN;
        drawHeader(cols, headers);
      }
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(9.5);
      doc.setTextColor(...INK);
      row.forEach((cell, i) => doc.text(cell, cols[i] + 2, y + 4.8));
      doc.setDrawColor(...LINE);
      doc.line(MARGIN, y + ROW, PAGE_W - MARGIN, y + ROW);
      y += ROW;
    });
  };

  // ---- findings table ----
  const findingRows = (ok?.detected_regions ?? []).map((d) => [
    d.label,
    quadrantLabels[d.quadrant] ?? '',
    d.quadrant,
    d.confidence != null ? `${Math.round(d.confidence * 100)}%` : '-',
  ]);
  const fCols = [MARGIN, MARGIN + 50, MARGIN + 100, MARGIN + 135];
  const fHeaders = ['Finding', 'Location (patient side)', 'Quadrant', 'Confidence'];
  ensureSpace(14 + ROW * 2);
  sectionTitle('What was found');
  drawHeader(fCols, fHeaders);
  if (findingRows.length === 0) {
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9.5);
    doc.setTextColor(...MUTED);
    doc.text('No areas were flagged.', MARGIN + 2, y + 4.8);
    y += ROW;
  }
  drawRows(fCols, fHeaders, findingRows);
  doc.setFontSize(8);
  doc.setTextColor(...MUTED);
  doc.text("Confidence is how sure the computer model is about each area, not a medical certainty.", MARGIN, y + 4.5);
  y += 10;

  // ---- technical details ----
  const techRows: string[][] = [
    ['Brightness cut-offs (thresholds)', ok?.thresholds?.join(', ') ?? '-'],
    ["Segmentation quality (Kapur's entropy)", fmt(ok?.kapur_entropy_fitness, 4)],
    ['Image fidelity (PSNR, dB)', fmt(ok?.psnr, 4)],
    ['Structure kept (SSIM)', fmt(ok?.ssim, 4)],
    ['Processing time (s)', fmt(ok?.runtime_sec, 4)],
    ['Steps used (iterations)', ok?.iterations_used != null ? `${ok.iterations_used} / 100` : '-'],
    ['Settings', 'd = 4 threshold levels, N = 30 agents, T = 100 iterations'],
  ];
  const tCols = [MARGIN, MARGIN + 80];
  const tHeaders = ['Technical details', 'Value'];
  ensureSpace(14 + ROW * (techRows.length + 1));
  sectionTitle('Technical details (for researchers)');
  drawHeader(tCols, tHeaders);
  drawRows(tCols, tHeaders, techRows);
  y += 6;

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

// Shared page footer: the "not a medical diagnosis" warning plus the
// app's tagline. Used on every page so the warning is always in the same place.
export default function Footer() {
  return (
    <footer className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-14 sm:mt-20">
      <div className="border-t border-line-strong pt-8 pb-8 text-muted">
        <div className="flex items-start gap-4 rounded-2xl bg-warn-bg border border-warn-line px-5 py-4">
          <span className="w-9 h-9 rounded-xl border border-warn-line text-warn-ink flex items-center justify-center shrink-0">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
              strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M12 3L2 20h20L12 3zM12 10v4M12 17.5v.01" />
            </svg>
          </span>
          <div className="min-w-0">
            <p className="text-[15px] font-bold text-warn-ink">Not a medical diagnosis</p>
            <p className="mt-1 text-[14px] leading-relaxed text-warn-ink opacity-90">
              This is a research prototype for academic demonstration only. Findings come from a trained detection
              model and threshold-based segmentation, not a validated diagnostic tool. The highlighted areas may be
              wrong or incomplete. Please consult a licensed dentist for any actual diagnosis or treatment.
            </p>
          </div>
        </div>

        <div className="mt-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1.5 text-[13px]">
          <span>
            <span className="font-bold text-ink">ESMA Dental</span>
            <span className="mx-2 opacity-50">·</span>
            Slime mould–inspired dental image analysis
          </span>
          <span>Research prototype · Not for clinical use</span>
        </div>
      </div>
    </footer>
  );
}

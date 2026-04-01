import { TooltipProvider } from "@/components/ui/tooltip";
import { Separator } from "@/components/ui/separator";
import { StatusBar } from "@/components/StatusBar";
import { DropZone } from "@/components/DropZone";
import { ProcessingOverlay } from "@/components/ProcessingOverlay";
import { ResultPanel } from "@/components/ResultPanel";
import { useDenoise } from "@/hooks/useDenoise";

export default function App() {
  const { status, result, error, psnr, ssim, processingTime, denoise, reset } = useDenoise();
  const isDone  = status === "completed";
  const isError = status === "error";

  const handleDownload = () => {
    if (!result) return;
    const link = document.createElement('a');
    link.href = result;
    link.download = 'denoised.png';
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  return (
    <TooltipProvider delayDuration={300}>
      <div className="min-h-dvh flex flex-col bg-[radial-gradient(circle_at_top,_rgba(94,234,212,0.14),transparent_24%),radial-gradient(circle_at_bottom_right,_rgba(94,234,212,0.08),transparent_18%),var(--bg)]">

        {/* Header */}
        <header className="border-b border-[var(--border)] bg-[var(--surface)]/90 backdrop-blur-xl sticky top-0 z-20 shadow-sm">
          <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="w-7 h-7 rounded bg-[var(--accent-dim)] border border-[var(--accent)]/30 flex items-center justify-center">
                <svg viewBox="0 0 16 16" width="14" height="14" fill="none">
                  <rect x="1" y="1" width="6" height="6" rx="1" fill="var(--accent)" opacity=".4"/>
                  <rect x="9" y="1" width="6" height="6" rx="1" fill="var(--accent)" opacity=".7"/>
                  <rect x="1" y="9" width="6" height="6" rx="1" fill="var(--accent)" opacity=".7"/>
                  <rect x="9" y="9" width="6" height="6" rx="1" fill="var(--accent)"/>
                </svg>
              </div>
              <span className="font-display font-bold text-sm tracking-wide text-[var(--text)]">
                DENOISING<span className="text-[var(--accent)]">.</span>STUDIO
              </span>
            </div>
            <StatusBar />
          </div>
        </header>

        {/* Main */}
        <main className="flex-1 flex items-center justify-center px-4 py-10 sm:py-12">
          <div className="w-full max-w-6xl mx-auto grid grid-cols-1 gap-10 lg:grid-cols-[1.15fr_1.85fr] lg:items-start">

            {/* Left — upload */}
            <aside className="space-y-8 lg:space-y-10">
              <div className="space-y-3">
                <h1 className="font-display font-semibold text-3xl sm:text-4xl leading-tight text-[var(--text)]">
                  Image<br />
                  <span className="text-[var(--accent)]">Denoising</span>
                </h1>
                <p className="text-sm sm:text-base text-[var(--muted)] font-mono leading-relaxed max-w-md">
                  SwinIR transformer model.
                  Upload a noisy image to restore it.
                </p>
              </div>

              <Separator />

              <div className="relative">
                <DropZone
                  onFile={denoise}
                  disabled={status === "uploading" || status === "processing"}
                />
                <ProcessingOverlay status={status} />
              </div>

              {/* Model info */}
              <div className="rounded-[22px] border border-[var(--border)] bg-[var(--surface)] p-5 space-y-4">
                <p className="text-[11px] font-mono text-[var(--muted)] uppercase tracking-[0.3em]">Model info</p>
                <div className="space-y-2 text-xs font-mono text-[var(--muted)]">
                  {[
                    ["Model",     "SwinIR"],
                    ["Task",      "Color denoising"],
                    ["Embed dim", "180"],
                    ["Depth",     "6 × RSTB"],
                    ["Window",    "8 × 8"],
                    ["Input",     "PNG / JPG / WEBP"],
                    ["Output",    "PNG (lossless)"],
                  ].map(([k, v]) => (
                    <div key={k} className="flex justify-between gap-2">
                      <span>{k}</span>
                      <span className="text-[var(--text)]">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </aside>

            {/* Right — result */}
            <section className="min-h-[420px]">
              {(isDone || isError) ? (
                <div className="rounded-xl border border-[var(--border)] bg-[var(--surface)] p-6">
                  <ResultPanel
                    result={result}
                    processingTime={processingTime}
                    psnr={psnr}
                    ssim={ssim}
                    onDownload={handleDownload}
                  />
                </div>
              ) : (
                <div className="h-full min-h-[420px] rounded-[24px] border border-dashed border-[var(--border)] flex flex-col items-center justify-center gap-4 text-center px-8 py-10 bg-[var(--surface)]/70">
                  <div className="w-14 h-14 rounded-xl border border-[var(--border-hi)] bg-[var(--surface)] flex items-center justify-center">
                    <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="var(--muted)" strokeWidth="1.5">
                      <rect x="3" y="3" width="18" height="18" rx="2"/>
                      <path d="M3 9h18M9 21V9"/>
                    </svg>
                  </div>
                  <p className="font-display font-semibold text-[var(--muted)] text-sm">
                    {status === "processing" || status === "uploading" ? "Processing…" : "Result will appear here"}
                  </p>
                  <p className="text-xs text-[var(--muted)]/60 font-mono max-w-[220px] leading-relaxed">
                    Drop an image on the left to start denoising
                  </p>
                </div>
              )}
            </section>
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t border-[var(--border)] py-5">
          <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-3">
            <p className="text-[11px] font-mono text-[var(--muted)]">Image Denoising Studio · SwinIR on DIV2K</p>
            <p className="text-[11px] font-mono text-[var(--muted)]">POST /api/denoise · GET /api/health</p>
          </div>
        </footer>

      </div>
    </TooltipProvider>
  );
}

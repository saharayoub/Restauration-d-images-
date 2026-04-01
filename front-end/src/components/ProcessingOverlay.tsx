import { Loader2, Zap } from "lucide-react";
import type { AppStatus } from "@/types/api";

interface ProcessingOverlayProps {
  status: AppStatus;
}

const LABELS: Partial<Record<AppStatus, string>> = {
  uploading:  "Sending image…",
  processing: "Running SwinIR on GPU…",
};

export function ProcessingOverlay({ status }: ProcessingOverlayProps) {
  if (status !== "uploading" && status !== "processing") return null;

  return (
    <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-5 rounded-[24px] bg-[var(--bg)]/92 backdrop-blur-xl">
      {/* Pulsing ring */}
      <div className="relative flex items-center justify-center">
        <span className="absolute w-16 h-16 rounded-full border border-[var(--accent)]/20 animate-ping" />
        <span className="absolute w-12 h-12 rounded-full border border-[var(--accent)]/30 animate-ping [animation-delay:150ms]" />
        <div className="relative w-10 h-10 rounded-full bg-[var(--accent-dim)] border border-[var(--accent)]/40 flex items-center justify-center">
          {status === "processing"
            ? <Zap size={18} className="text-[var(--accent)]" />
            : <Loader2 size={18} className="text-[var(--accent)] animate-spin" />
          }
        </div>
      </div>

      <div className="text-center space-y-1">
        <p className="font-display font-semibold text-lg text-[var(--text)]">
          {LABELS[status]}
        </p>
        {status === "processing" && (
          <p className="text-xs text-[var(--muted)] font-mono">
            SwinIR · 180-dim · 6 RSTB layers
          </p>
        )}
      </div>

      {/* Scanning line */}
      {status === "processing" && (
        <div className="w-48 h-px bg-gradient-to-r from-transparent via-[var(--accent)] to-transparent"
             style={{ animation: "scan 1.8s ease-in-out infinite" }} />
      )}

      <style>{`
        @keyframes scan {
          0%, 100% { opacity: 0; transform: scaleX(0.3); }
          50%       { opacity: 1; transform: scaleX(1); }
        }
      `}</style>
    </div>
  );
}

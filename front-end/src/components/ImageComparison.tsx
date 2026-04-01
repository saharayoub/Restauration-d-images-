import { useCallback, useRef, useState } from "react";
import { cn } from "@/lib/utils";

interface ImageComparisonProps {
  original: string;
  denoised: string;
  className?: string;
}

export function ImageComparison({ original, denoised, className }: ImageComparisonProps) {
  const [split, setSplit] = useState(50);
  const [dragging, setDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const updateSplit = useCallback((clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const pct = Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100));
    setSplit(pct);
  }, []);

  const onMouseMove  = useCallback((e: React.MouseEvent)  => { if (dragging) updateSplit(e.clientX); }, [dragging, updateSplit]);
  const onTouchMove  = useCallback((e: React.TouchEvent)  => { updateSplit(e.touches[0].clientX); }, [updateSplit]);

  return (
    <div
      ref={containerRef}
      className={cn("relative overflow-hidden rounded-xl select-none cursor-col-resize group", className)}
      onMouseMove={onMouseMove}
      onMouseUp={() => setDragging(false)}
      onMouseLeave={() => setDragging(false)}
      onTouchMove={onTouchMove}
      onTouchEnd={() => setDragging(false)}
    >
      {/* Denoised (bottom layer — full width) */}
      <img
        src={denoised}
        alt="Denoised"
        className="block w-full h-full object-contain"
        draggable={false}
      />

      {/* Original (top layer — clipped) */}
      <div
        className="absolute inset-0 overflow-hidden"
        style={{ width: `${split}%` }}
      >
        <img
          src={original}
          alt="Original"
          className="block h-full object-contain"
          style={{ width: containerRef.current?.offsetWidth ?? "100%" }}
          draggable={false}
        />
      </div>

      {/* Divider line */}
      <div
        className="absolute inset-y-0 w-0.5 bg-[var(--accent)] shadow-[0_0_12px_var(--accent)]"
        style={{ left: `${split}%`, transform: "translateX(-50%)" }}
      >
        {/* Handle */}
        <div
          className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-[var(--accent)] shadow-[0_0_20px_var(--accent)] flex items-center justify-center cursor-grab active:cursor-grabbing"
          onMouseDown={(e) => { e.preventDefault(); setDragging(true); }}
          onTouchStart={() => setDragging(true)}
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M4 7h6M4 7L2 5M4 7L2 9M10 7l2-2M10 7l2 2" stroke="#0a0a0b" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        </div>
      </div>

      {/* Labels */}
      <span className="absolute top-3 left-3 px-2 py-0.5 rounded text-[11px] font-mono bg-[var(--bg)]/80 text-[var(--muted)] border border-[var(--border)]">
        ORIGINAL
      </span>
      <span className="absolute top-3 right-3 px-2 py-0.5 rounded text-[11px] font-mono bg-[var(--accent-dim)]/80 text-[var(--accent)] border border-[var(--accent)]/30">
        DENOISED
      </span>
    </div>
  );
}

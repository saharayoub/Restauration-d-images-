import { useCallback, useState } from "react";
import { Upload, ImageIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface DropZoneProps {
  onFile: (file: File) => void;
  disabled?: boolean;
}

const ACCEPTED = ["image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"];

export function DropZone({ onFile, disabled }: DropZoneProps) {
  const [dragging, setDragging] = useState(false);
  const [dragError, setDragError] = useState(false);

  const handle = useCallback(
    (file: File) => {
      if (!ACCEPTED.includes(file.type)) {
        setDragError(true);
        setTimeout(() => setDragError(false), 1500);
        return;
      }
      onFile(file);
    },
    [onFile]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handle(file);
    },
    [handle]
  );

  const onInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handle(file);
      e.target.value = "";
    },
    [handle]
  );

  return (
    <label
      className={cn(
        "relative flex flex-col items-center justify-center gap-5",
        "w-full min-h-[360px] rounded-[24px] border-2 border-dashed cursor-pointer bg-[var(--surface)]/80",
        "transition-all duration-300 group",
        disabled
          ? "opacity-40 cursor-not-allowed border-[var(--border)]"
          : dragging
          ? "border-[var(--accent)] bg-[var(--accent-dim)]"
          : dragError
          ? "border-[var(--danger)] bg-[var(--danger)]/5"
          : "border-[var(--border-hi)] hover:border-[var(--accent)]/60 hover:bg-[var(--surface)]"
      )}
      onDragEnter={(e) => { e.preventDefault(); if (!disabled) setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDragOver={(e) => e.preventDefault()}
      onDrop={disabled ? undefined : onDrop}
    >
      {/* Grid lines decoration */}
      <div className="absolute inset-0 rounded-xl overflow-hidden opacity-[0.03] pointer-events-none"
           style={{ backgroundImage: "linear-gradient(var(--text) 1px, transparent 1px), linear-gradient(90deg, var(--text) 1px, transparent 1px)", backgroundSize: "40px 40px" }} />

      <input
        type="file"
        accept={ACCEPTED.join(",")}
        className="sr-only"
        onChange={onInputChange}
        disabled={disabled}
      />

      <div className={cn(
        "flex items-center justify-center w-16 h-16 rounded-xl border transition-colors duration-300 bg-[var(--surface)]",
        dragging
          ? "border-[var(--accent)] bg-[var(--accent-dim)]"
          : dragError
          ? "border-[var(--danger)] bg-[var(--danger)]/10"
          : "border-[var(--border-hi)] bg-[var(--border)] group-hover:border-[var(--accent)]/50"
      )}>
        {dragError
          ? <span className="text-[var(--danger)] text-xs font-mono">✕ invalid</span>
          : dragging
          ? <Upload className="text-[var(--accent)]" size={22} />
          : <ImageIcon className="text-[var(--muted)] group-hover:text-[var(--text)] transition-colors" size={22} />
        }
      </div>

      <div className="text-center space-y-1.5 px-6">
        <p className="font-display font-semibold text-base text-[var(--text)]">
          {dragging ? "Release to upload" : "Drop your image here"}
        </p>
        <p className="text-sm text-[var(--muted)] font-mono">
          or <span className="text-[var(--accent)] underline underline-offset-2">click to browse</span>
        </p>
        <p className="text-[11px] text-[var(--muted)]">PNG · JPG · WEBP · BMP</p>
      </div>
    </label>
  );
}

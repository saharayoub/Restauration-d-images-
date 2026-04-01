import * as React from "react";
import { cn } from "@/lib/utils";

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "success" | "warn" | "danger" | "muted";
}

export function Badge({ className, variant = "default", ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 px-2.5 py-0.5 text-[11px] font-medium rounded-full border font-mono tracking-wide",
        {
          default: "border-[var(--border-hi)] text-[var(--muted)] bg-[var(--surface)]",
          success: "border-[var(--accent)]/30 text-[var(--accent)] bg-[var(--accent)]/10",
          warn:    "border-[var(--warn)]/30 text-[var(--warn)] bg-[var(--warn)]/10",
          danger:  "border-[var(--danger)]/30 text-[var(--danger)] bg-[var(--danger)]/10",
          muted:   "border-transparent text-[var(--muted)] bg-[var(--border)]",
        }[variant],
        className
      )}
      {...props}
    />
  );
}

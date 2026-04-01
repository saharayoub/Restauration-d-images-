import { Cpu, Zap, AlertCircle, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { useHealth } from "@/hooks/useHealth";

export function StatusBar() {
  const { health, loading } = useHealth();

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-[var(--muted)]">
        <Loader2 size={13} className="animate-spin" />
        <span className="text-xs font-mono">connecting…</span>
      </div>
    );
  }

  if (!health) {
    return (
      <div className="flex items-center gap-2">
        <AlertCircle size={13} className="text-[var(--danger)]" />
        <span className="text-xs font-mono text-[var(--danger)]">server offline</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2.5 flex-wrap">
      <Badge variant="success">
        <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent)] animate-pulse" />
        online
      </Badge>

      {health.cuda_available ? (
        <Badge variant="success">
          <Zap size={10} />
          {health.gpu_name ?? "GPU"}
        </Badge>
      ) : (
        <Badge variant="warn">
          <Cpu size={10} />
          CPU mode
        </Badge>
      )}

      <Badge variant="muted">
        {health.device}
      </Badge>
    </div>
  );
}

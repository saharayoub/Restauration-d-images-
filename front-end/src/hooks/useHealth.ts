import { useEffect, useState } from "react";
import type { HealthResponse } from "@/types/api";

export function useHealth() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function check() {
      try {
        const res = await fetch("/api/health");
        if (!cancelled && res.ok) setHealth(await res.json());
      } catch {
        // server not ready yet — silently ignore
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    check();
    const id = setInterval(check, 15_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  return { health, loading };
}

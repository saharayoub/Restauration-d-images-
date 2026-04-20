import React from 'react';
import { Download, Clock, BarChart3, Activity } from 'lucide-react';
import { Button } from './ui/button';
import { Separator } from './ui/separator';

interface ResultPanelProps {
  result: string | null;
  processingTime?: number;
  psnr?: number;
  ssim?: number;
  onDownload: () => void;
}

export const ResultPanel: React.FC<ResultPanelProps> = ({
  result,
  processingTime,
  psnr,
  ssim,
  onDownload
}) => {
  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-zinc-500">
        <div className="w-16 h-16 rounded-full bg-zinc-900 flex items-center justify-center mb-4">
          <BarChart3 className="w-8 h-8 text-zinc-600" />
        </div>
        <p className="text-sm font-mono">Process an image to see results</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-zinc-100 font-syne">Denoised Result</h3>
        <Button
          onClick={onDownload}
          variant="outline"
          size="sm"
          className="border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/10 hover:text-emerald-300"
        >
          <Download className="w-4 h-4 mr-2" />
          Download
        </Button>
      </div>

      <div className="relative flex-1 bg-zinc-950 rounded-lg overflow-hidden border border-zinc-800">
        <img
          src={result}
          alt="Denoised result"
          className="w-full h-full object-contain"
        />
      </div>

      <div className="mt-4 space-y-3">
        <Separator className="bg-zinc-800" />
        
        <div className="flex items-center justify-between">
          <span className="text-xs text-zinc-500 font-mono uppercase tracking-wider">Metrics</span>
        </div>

        <div className="grid grid-cols-3 gap-2">
          {processingTime !== undefined && (
            <div className="bg-zinc-900/50 rounded-md p-2 border border-zinc-800">
              <div className="flex items-center gap-1.5 mb-1">
                <Clock className="w-3 h-3 text-zinc-500" />
                <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-mono">Time</span>
              </div>
              <span className="text-sm font-mono text-zinc-200">{processingTime.toFixed(2)}s</span>
            </div>
          )}

          {psnr !== undefined && (
            <div className="bg-zinc-900/50 rounded-md p-2 border border-zinc-800">
              <div className="flex items-center gap-1.5 mb-1">
                <Activity className="w-3 h-3 text-emerald-500" />
                <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-mono">PSNR</span>
              </div>
              <span className="text-sm font-mono text-emerald-400">{psnr.toFixed(2)} dB</span>
            </div>
          )}

          {ssim !== undefined && (
            <div className="bg-zinc-900/50 rounded-md p-2 border border-zinc-800">
              <div className="flex items-center gap-1.5 mb-1">
                <BarChart3 className="w-3 h-3 text-emerald-500" />
                <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-mono">SSIM</span>
              </div>
              <span className="text-sm font-mono text-emerald-400">{ssim.toFixed(4)}</span>
            </div>
          )}
        </div>

        <div className="text-[10px] text-zinc-600 font-mono leading-relaxed">
          <p>PSNR: Peak Signal-to-Noise Ratio (higher is better, &gt;30dB good)</p>
          <p>SSIM: Structural Similarity Index (0-1, closer to 1 is better)</p>
        </div>
      </div>
    </div>
  );
};
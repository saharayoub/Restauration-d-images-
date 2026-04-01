import { useState, useCallback } from 'react';
import type { DenoiseResponse, AppStatus } from '../types/api';

interface UseDenoiseReturn {
  status: AppStatus;
  result: string | null;
  error: string | null;
  psnr: number | undefined;
  ssim: number | undefined;
  processingTime: number | undefined;
  denoise: (file: File) => Promise<void>;
  reset: () => void;
}

export const useDenoise = (): UseDenoiseReturn => {
  const [status, setStatus] = useState<AppStatus>('idle');
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [psnr, setPsnr] = useState<number | undefined>(undefined);
  const [ssim, setSsim] = useState<number | undefined>(undefined);
  const [processingTime, setProcessingTime] = useState<number | undefined>(undefined);

  const denoise = useCallback(async (file: File) => {
    setStatus('uploading');
    setError(null);

    const formData = new FormData();
    formData.append('image', file);

    try {
      setStatus('processing');
      const response = await fetch('/api/denoise', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data: DenoiseResponse = await response.json();
      
      setResult(data.denoised);
      setPsnr(data.psnr);
      setSsim(data.ssim);
      setProcessingTime(data.processing_time);
      setStatus('completed');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      setStatus('error');
    }
  }, []);

  const reset = useCallback(() => {
    setStatus('idle');
    setResult(null);
    setError(null);
    setPsnr(undefined);
    setSsim(undefined);
    setProcessingTime(undefined);
  }, []);

  return {
    status,
    result,
    error,
    psnr,
    ssim,
    processingTime,
    denoise,
    reset,
  };
};
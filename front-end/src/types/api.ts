export interface HealthResponse {
  status: string;
  device: string;
  cuda_available: boolean;
  gpu_name: string | null;
}

export interface DenoiseResponse {
  denoised: string;
  psnr: number;
  ssim: number;
  processing_time: number;
}

export type AppStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'error';
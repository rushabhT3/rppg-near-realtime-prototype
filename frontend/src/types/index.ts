export interface ChunkData {
  type: 'chunk_update';
  chunk_index: number;
  start_time: number;
  end_time: number;
  bpm: number | null;
  sqi: number | null;
  respiratory_rate: number | null;
}

export interface FinalResult {
  type: 'final_result';
  overall_bpm: number | null;
  overall_sqi: number | null;
  overall_respiratory_rate: number | null;
  total_processing_time_sec: number;
  average_latency_ms: number;
  video_duration_sec: number;
}

export type AppStatus = 'idle' | 'uploading' | 'processing' | 'done' | 'error';

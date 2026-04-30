import { useState, useRef, useCallback } from 'react';
import type { ChunkData, FinalResult, AppStatus } from '../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

export const useRPPGSession = () => {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [chunks, setChunks] = useState<ChunkData[]>([]);
  const [finalResult, setFinalResult] = useState<FinalResult | null>(null);
  const [status, setStatus] = useState<AppStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  
  const wsRef = useRef<WebSocket | null>(null);

  const connectWebSocket = useCallback((id: string) => {
    setStatus('processing');
    const ws = new WebSocket(`${WS_URL}/ws/process/${id}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'chunk_update') {
        setChunks(prev => [...prev, data]);
        const chunkIndex = data.chunk_index;
        const totalExpected = 12; // Based on 60s video
        setProgress(Math.min(((chunkIndex + 1) / totalExpected) * 100, 98));
      } else if (data.type === 'final_result') {
        setFinalResult(data);
        setStatus('done');
        setProgress(100);
        ws.close();
      } else if (data.error) {
        setError(data.error);
        setStatus('error');
        ws.close();
      }
    };

    ws.onerror = () => {
      setError('WebSocket connection failed');
      setStatus('error');
    };
  }, []);

  const uploadVideo = async (file: File) => {
    setStatus('uploading');
    setError(null);
    setChunks([]);
    setFinalResult(null);
    setProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');
      
      const data = await response.json();
      setSessionId(data.session_id);
      connectWebSocket(data.session_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setStatus('error');
    }
  };

  return {
    sessionId,
    chunks,
    finalResult,
    status,
    error,
    progress,
    uploadVideo
  };
};

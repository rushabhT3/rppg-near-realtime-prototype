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

  const connectWebSocket = useCallback((id: string, objectName?: string) => {
    setStatus('processing');
    setError(null);
    setProgress(0);
    
    // Clear any existing WebSocket
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    const wsUrl = objectName
      ? `${WS_URL}/ws/process/${id}?object_name=${encodeURIComponent(objectName)}`
      : `${WS_URL}/ws/process/${id}`;
    const ws = new WebSocket(wsUrl);
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
        
        // Keep results visible, reset only input state
        setTimeout(() => {
          setStatus('idle');
          setProgress(0);
          setError(null);
        }, 3000);
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
    setProgress(0);
    // Clear previous results
    setChunks([]);
    setFinalResult(null);
    
    // Clear any existing WebSocket
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    try {
      // Try GCS signed URL upload first (for Cloud Run)
      const urlResponse = await fetch(
        `${API_URL}/api/upload-url?filename=${encodeURIComponent(file.name)}`
      );

      if (urlResponse.ok) {
        const { session_id, upload_url, object_name } = await urlResponse.json();

        // Upload directly to GCS via signed URL
        const uploadResponse = await fetch(upload_url, {
          method: 'PUT',
          body: file,
          headers: { 'Content-Type': file.type || 'video/mp4' },
        });

        if (uploadResponse.ok) {
          setSessionId(session_id);
          connectWebSocket(session_id, object_name);
          return;
        }
        // GCS upload failed, fall through to direct upload
      }

      // Fallback: direct upload (for local dev or GCS unavailable)
      const formData = new FormData();
      formData.append('file', file);

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

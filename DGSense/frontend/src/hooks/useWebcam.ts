import { useState, useRef, useCallback, useEffect } from 'react';

export interface UseWebcamReturn {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isStreaming: boolean;
  permissionState: 'prompt' | 'granted' | 'denied' | 'error';
  errorMessage: string | null;
  startCamera: () => Promise<boolean>;
  stopCamera: () => void;
  availableDevices: MediaDeviceInfo[];
  selectedDeviceId: string;
  setSelectedDeviceId: (id: string) => void;
}

export function useWebcam(): UseWebcamReturn {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [permissionState, setPermissionState] = useState<'prompt' | 'granted' | 'denied' | 'error'>('prompt');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [availableDevices, setAvailableDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');

  // Enumerate video devices
  const updateDevices = useCallback(async () => {
    try {
      if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevs = devices.filter((d) => d.kind === 'videoinput');
        setAvailableDevices(videoDevs);
        if (videoDevs.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(videoDevs[0].deviceId);
        }
      }
    } catch (err) {
      console.warn('Could not enumerate media devices:', err);
    }
  }, [selectedDeviceId]);

  useEffect(() => {
    updateDevices();
  }, [updateDevices]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop();
      });
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
  }, []);

  const startCamera = useCallback(async (): Promise<boolean> => {
    stopCamera();
    setErrorMessage(null);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setPermissionState('error');
      setErrorMessage('Camera access is not supported by this browser.');
      return false;
    }

    try {
      const constraints: MediaStreamConstraints = {
        video: selectedDeviceId
          ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } }
          : { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await new Promise((resolve) => {
          if (videoRef.current) {
            videoRef.current.onloadedmetadata = () => {
              videoRef.current?.play().then(resolve).catch(resolve);
            };
          }
        });
      }

      setIsStreaming(true);
      setPermissionState('granted');
      updateDevices();
      return true;
    } catch (err: any) {
      console.error('Camera access error:', err);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setPermissionState('denied');
        setErrorMessage('Camera permission was denied. Please allow camera access in browser settings.');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setPermissionState('error');
        setErrorMessage('No camera device detected on your system.');
      } else {
        setPermissionState('error');
        setErrorMessage(err.message || 'Failed to start camera preview.');
      }
      setIsStreaming(false);
      return false;
    }
  }, [selectedDeviceId, stopCamera, updateDevices]);

  // Cleanup all media tracks on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return {
    videoRef,
    isStreaming,
    permissionState,
    errorMessage,
    startCamera,
    stopCamera,
    availableDevices,
    selectedDeviceId,
    setSelectedDeviceId,
  };
}

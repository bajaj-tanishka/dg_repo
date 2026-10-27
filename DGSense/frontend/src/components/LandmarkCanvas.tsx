import React, { useEffect, useRef } from 'react';
import { HandLandmarkerResult } from '@mediapipe/tasks-vision';

interface LandmarkCanvasProps {
  landmarksResult: HandLandmarkerResult | null;
  videoWidth: number;
  videoHeight: number;
}

const HAND_CONNECTIONS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm base
  [5, 9], [9, 13], [13, 17]
];

export const LandmarkCanvas: React.FC<LandmarkCanvasProps> = ({
  landmarksResult,
  videoWidth,
  videoHeight,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear previous frame
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!landmarksResult || !landmarksResult.landmarks) return;

    landmarksResult.landmarks.forEach((hand, handIndex) => {
      const isRight = landmarksResult.handednesses?.[handIndex]?.[0]?.categoryName === 'Right';
      const strokeColor = isRight ? '#38bdf8' : '#a855f7';
      const jointColor = isRight ? '#0284c7' : '#7e22ce';

      // Draw skeleton lines
      ctx.lineWidth = 3;
      ctx.strokeStyle = strokeColor;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      HAND_CONNECTIONS.forEach(([startIdx, endIdx]) => {
        const start = hand[startIdx];
        const end = hand[endIdx];
        if (start && end) {
          ctx.beginPath();
          ctx.moveTo(start.x * canvas.width, start.y * canvas.height);
          ctx.lineTo(end.x * canvas.width, end.y * canvas.height);
          ctx.stroke();
        }
      });

      // Draw joint markers
      hand.forEach((lm, idx) => {
        const x = lm.x * canvas.width;
        const y = lm.y * canvas.height;
        const radius = idx === 0 || idx === 4 || idx === 8 || idx === 12 || idx === 16 || idx === 20 ? 5 : 3.5;

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fillStyle = jointColor;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
      });
    });
  }, [landmarksResult, videoWidth, videoHeight]);

  return (
    <canvas
      ref={canvasRef}
      className="landmark-canvas"
      width={videoWidth || 640}
      height={videoHeight || 480}
    />
  );
};

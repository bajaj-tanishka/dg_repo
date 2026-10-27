/**
 * MediaPipe Tasks Vision client-side landmark extraction.
 * Extracts hand landmarks and formats them into the 144-dimensional dgsense-v1 schema.
 */

import { FilesetResolver, HandLandmarker, HandLandmarkerResult } from '@mediapipe/tasks-vision';

let handLandmarker: HandLandmarker | null = null;
let isInitializing = false;

export interface ExtractedFrameFeatures {
  vector: number[]; // 144 floats
  leftHandDetected: boolean;
  rightHandDetected: boolean;
  landmarksRaw: HandLandmarkerResult | null;
}

export async function initMediaPipe(): Promise<HandLandmarker> {
  if (handLandmarker) return handLandmarker;
  if (isInitializing) {
    // Wait for in-flight initialization
    while (isInitializing) {
      await new Promise((r) => setTimeout(r, 100));
      if (handLandmarker) return handLandmarker;
    }
  }

  isInitializing = true;
  try {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    );

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    return handLandmarker;
  } catch (error) {
    console.warn('[MediaPipe] GPU delegate failed or network error, falling back to CPU delegate:', error);
    try {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
          delegate: 'CPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.5,
      });
      return handLandmarker;
    } catch (fallbackError) {
      console.error('[MediaPipe] Failed to initialize hand landmarker:', fallbackError);
      throw fallbackError;
    }
  } finally {
    isInitializing = false;
  }
}

/**
 * Extracts 144 features from a video frame:
 * - Indices 0-62: Left hand (21 landmarks x, y, z)
 * - Indices 63-125: Right hand (21 landmarks x, y, z)
 * - Indices 126-143: Upper body keypoints (estimated shoulders, elbows, wrists)
 */
export function extract144Features(
  videoElement: HTMLVideoElement,
  timestampMs: number
): ExtractedFrameFeatures {
  const featureVector = new Array(144).fill(0.0);
  let leftHandDetected = false;
  let rightHandDetected = false;
  let result: HandLandmarkerResult | null = null;

  if (!handLandmarker) {
    return {
      vector: featureVector,
      leftHandDetected: false,
      rightHandDetected: false,
      landmarksRaw: null,
    };
  }

  try {
    result = handLandmarker.detectForVideo(videoElement, timestampMs);

    if (result && result.landmarks && result.landmarks.length > 0) {
      result.landmarks.forEach((hand, idx) => {
        const handedness = result?.handednesses?.[idx]?.[0]?.categoryName;
        // Note: In mirrored webcam view, "Left" can appear as user's right or vice versa
        const isLeft = handedness === 'Left';
        const isRight = handedness === 'Right';

        if (isLeft && !leftHandDetected) {
          leftHandDetected = true;
          // Fill indices 0 to 62
          hand.forEach((lm, lmIdx) => {
            const base = lmIdx * 3;
            featureVector[base] = lm.x;
            featureVector[base + 1] = lm.y;
            featureVector[base + 2] = lm.z;
          });
          // Also set wrist reference in upper body pose slots (left wrist: index 138-140)
          featureVector[138] = hand[0].x;
          featureVector[139] = hand[0].y;
          featureVector[140] = hand[0].z;
        } else if (isRight && !rightHandDetected) {
          rightHandDetected = true;
          // Fill indices 63 to 125
          hand.forEach((lm, lmIdx) => {
            const base = 63 + lmIdx * 3;
            featureVector[base] = lm.x;
            featureVector[base + 1] = lm.y;
            featureVector[base + 2] = lm.z;
          });
          // Also set wrist reference in upper body pose slots (right wrist: index 141-143)
          featureVector[141] = hand[0].x;
          featureVector[142] = hand[0].y;
          featureVector[143] = hand[0].z;
        } else if (!leftHandDetected) {
          // If handedness is not explicit, map first hand to right hand (dominant in most datasets)
          rightHandDetected = true;
          hand.forEach((lm, lmIdx) => {
            const base = 63 + lmIdx * 3;
            featureVector[base] = lm.x;
            featureVector[base + 1] = lm.y;
            featureVector[base + 2] = lm.z;
          });
        }
      });

      // Approximate upper body torso points if wrists are present
      // left_shoulder: 126-128, right_shoulder: 129-131, left_elbow: 132-134, right_elbow: 135-137
      if (leftHandDetected) {
        featureVector[126] = 0.35; // left shoulder x default
        featureVector[127] = 0.50; // left shoulder y default
        featureVector[128] = 0.0;
        featureVector[132] = 0.30; // left elbow x
        featureVector[133] = 0.65; // left elbow y
        featureVector[134] = 0.0;
      }
      if (rightHandDetected) {
        featureVector[129] = 0.65; // right shoulder x default
        featureVector[130] = 0.50; // right shoulder y default
        featureVector[131] = 0.0;
        featureVector[135] = 0.70; // right elbow x
        featureVector[136] = 0.65; // right elbow y
        featureVector[137] = 0.0;
      }
    }
  } catch (err) {
    // If detectForVideo fails during tab blur or frame skip, return zeros safely
    console.debug('[MediaPipe] detectForVideo frame skipped:', err);
  }

  return {
    vector: featureVector,
    leftHandDetected,
    rightHandDetected,
    landmarksRaw: result,
  };
}

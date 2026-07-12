/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Detection, Track, TrackerConfig } from '../types';

// Compute Intersection over Union (IoU) metric between two bounding boxes
export function computeIoU(
  boxA: [number, number, number, number],
  boxB: [number, number, number, number]
): number {
  const xA = Math.max(boxA[0], boxB[0]);
  const yA = Math.max(boxA[1], boxB[1]);
  const xB = Math.min(boxA[0] + boxA[2], boxB[0] + boxB[2]);
  const yB = Math.min(boxA[1] + boxA[3], boxB[1] + boxB[3]);

  // Compute intersection area
  const interWidth = Math.max(0, xB - xA);
  const interHeight = Math.max(0, yB - yA);
  const interArea = interWidth * interHeight;

  if (interArea === 0) return 0;

  // Compute union area
  const boxAArea = boxA[2] * boxA[3];
  const boxBArea = boxB[2] * boxB[3];
  const unionArea = boxAArea + boxBArea - interArea;

  return unionArea > 0 ? interArea / unionArea : 0;
}

// Generate an elegant glowing neon color based on an ID and a theme
export function getNeonColor(id: number, theme: 'purple' | 'cyan' | 'pink' | 'amber'): string {
  // Use a hash of ID to distribute starting hue smoothly
  const hash = (id * 157) % 360;
  
  switch (theme) {
    case 'purple':
      // Purple theme focuses on violet, indigo, and deep blue ranges
      const purpleHue = (260 + (id * 45) % 80) % 360;
      return `hsl(${purpleHue}, 100%, 65%)`;
    case 'cyan':
      // Cyan/teal theme
      const cyanHue = (170 + (id * 35) % 60) % 360;
      return `hsl(${cyanHue}, 100%, 60%)`;
    case 'pink':
      // Cyberpunk pink/neon magenta theme
      const pinkHue = (310 + (id * 25) % 70) % 360;
      return `hsl(${pinkHue}, 100%, 65%)`;
    case 'amber':
      // Sci-fi amber/orange theme
      const amberHue = (25 + (id * 30) % 55) % 360;
      return `hsl(${amberHue}, 100%, 55%)`;
    default:
      return `hsl(${hash}, 95%, 60%)`;
  }
}

export class SORTTracker {
  private nextId = 1;
  private tracks: Track[] = [];

  constructor() {}

  /**
   * Resets the tracker state and counter.
   */
  public reset(): void {
    this.tracks = [];
    this.nextId = 1;
  }

  /**
   * Gets the active track listing.
   */
  public getTracks(): Track[] {
    return this.tracks;
  }

  /**
   * Performs prediction and update step for a new frame, matching raw detections to existing tracks.
   * Uses an Intersection-over-Union greedy matching technique resembling SORT algorithms.
   */
  public update(detections: Detection[], config: TrackerConfig): Track[] {
    const updatedTracks: Track[] = [];

    // 1. Prediction: Project all tracks to their new locations based on estimated velocity
    const predictedTracks = this.tracks.map(track => {
      const { bbox, velocity } = track;
      // Constant velocity model projection:
      const predictedBbox: [number, number, number, number] = [
        bbox[0] + velocity[0],
        bbox[1] + velocity[1],
        bbox[2],
        bbox[3]
      ];
      return {
        ...track,
        predictedBbox
      };
    });

    // Match trackers to detections using greedy bipartite matching based on IoU cost
    const matchedDetections = new Set<number>();
    const matchedTracks = new Set<number>();

    // Prepare all possible pairs with their IoUs
    const associations: Array<{
      trackIdx: number;
      detIdx: number;
      iou: number;
    }> = [];

    predictedTracks.forEach((track, trackIdx) => {
      detections.forEach((det, detIdx) => {
        // Only consider matches of the same class type to preserve track taxonomy
        if (track.class === det.class) {
          const iou = computeIoU(track.predictedBbox || track.bbox, det.bbox);
          if (iou >= config.iouThreshold) {
            associations.push({ trackIdx, detIdx, iou });
          }
        }
      });
    });

    // Sort association pairs by maximum IoU
    associations.sort((a, b) => b.iou - a.iou);

    // Apply greedy matching assignment
    associations.forEach(({ trackIdx, detIdx }) => {
      if (!matchedTracks.has(trackIdx) && !matchedDetections.has(detIdx)) {
        matchedTracks.add(trackIdx);
        matchedDetections.add(detIdx);

        const track = predictedTracks[trackIdx];
        const det = detections[detIdx];

        // Compute current centers
        const oldCenter = [
          track.bbox[0] + track.bbox[2] / 2,
          track.bbox[1] + track.bbox[3] / 2
        ];
        const newCenter = [
          det.bbox[0] + det.bbox[2] / 2,
          det.bbox[1] + det.bbox[3] / 2
        ];

        // Velocity prediction with smoothing index (constant acceleration dampening multiplier)
        const alpha = 0.5;
        const rawDx = newCenter[0] - oldCenter[0];
        const rawDy = newCenter[1] - oldCenter[1];
        const newDx = track.velocity[0] * (1 - alpha) + rawDx * alpha;
        const newDy = track.velocity[1] * (1 - alpha) + rawDy * alpha;

        // Velocity vector speed
        const speed = Math.sqrt(rawDx * rawDx + rawDy * rawDy);

        // Update track coordinates and reset survival parameters
        const updatedHistory = [
          ...track.history,
          { x: newCenter[0], y: newCenter[1], timestamp: Date.now() }
        ].slice(-config.historyLength);

        const updatedTrack: Track = {
          ...track,
          bbox: det.bbox,
          score: det.score,
          history: updatedHistory,
          velocity: [newDx, newDy],
          speed: Number(speed.toFixed(1)),
          age: track.age + 1,
          missedFrames: 0,
          isActive: true
        };

        updatedTracks.push(updatedTrack);
      }
    });

    // 2. Handle Unmatched Tracks (survival & projection)
    predictedTracks.forEach((track, idx) => {
      if (!matchedTracks.has(idx)) {
        const missedFrames = track.missedFrames + 1;
        // If within persistence capability, project track via velocity
        if (missedFrames <= config.maxMissedFrames) {
          const predictedBbox = track.predictedBbox || track.bbox;
          
          updatedTracks.push({
            ...track,
            bbox: predictedBbox,
            missedFrames,
            isActive: false, // temporarily shaded
            // Dampen velocity when target is not observed
            velocity: [track.velocity[0] * 0.8, track.velocity[1] * 0.8]
          });
        }
      }
    });

    // 3. Handle Unmatched Detections (spawn new tracks)
    detections.forEach((det, idx) => {
      if (!matchedDetections.has(idx)) {
        const id = this.nextId++;
        const center = [
          det.bbox[0] + det.bbox[2] / 2,
          det.bbox[1] + det.bbox[3] / 2
        ];

        const newTrack: Track = {
          id,
          bbox: det.bbox,
          class: det.class,
          score: det.score,
          history: [{ x: center[0], y: center[1], timestamp: Date.now() }],
          velocity: [0, 0],
          color: getNeonColor(id, config.colorTheme),
          age: 1,
          missedFrames: 0,
          isActive: true,
          speed: 0
        };

        updatedTracks.push(newTrack);
      }
    });

    // Synchronize current tracks listing
    this.tracks = updatedTracks;

    // Filter minimum tracking age if configured
    return this.tracks.filter(tr => tr.age >= config.minTrackAge);
  }
}

/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Detection {
  bbox: [number, number, number, number]; // [x, y, w, h] from model
  class: string;
  score: number;
}

export interface TrackPoint {
  x: number;
  y: number;
  timestamp: number;
}

export interface Track {
  id: number;
  bbox: [number, number, number, number]; // [x, y, w, h]
  class: string;
  score: number;
  history: TrackPoint[];
  velocity: [number, number]; // [dx, dy] per frame
  color: string;
  age: number; // total active frames
  missedFrames: number; // consecutive missed frames
  isActive: boolean;
  predictedBbox?: [number, number, number, number];
  speed: number; // average pixels shifted per frame
}

export interface TrackerConfig {
  iouThreshold: number;
  confidenceThreshold: number;
  maxMissedFrames: number;
  historyLength: number;
  drawTrails: boolean;
  blurBackground: boolean;
  minTrackAge: number;
  colorTheme: 'purple' | 'cyan' | 'pink' | 'amber';
  filterClasses: string[];
}

export interface SystemEvent {
  id: string;
  timestamp: string;
  type: 'info' | 'success' | 'warning' | 'alert';
  message: string;
}

export interface ChartDataPoint {
  time: string;
  activeCount: number;
  person: number;
  vehicle: number;
  others: number;
}

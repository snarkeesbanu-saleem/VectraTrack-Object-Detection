/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Detection } from '../types';

export interface SimulatedObject {
  id: number;
  class: 'person' | 'car' | 'bicycle' | 'dog' | 'backpack';
  x: number;
  y: number;
  width: number;
  height: number;
  vx: number;
  vy: number;
  color: string;
  isOccluded: boolean;
}

export class TrackingSimulator {
  private objects: SimulatedObject[] = [];
  private width: number;
  private height: number;
  private nextId = 100; // distinct range for simulator elements

  // Structural occlusions to test SORT cohesion through visual obstacles
  public occlusions: Array<{ x: number; y: number; w: number; h: number; opacity: number }> = [];

  constructor(width: number = 640, height: number = 480) {
    this.width = width;
    this.height = height;
    this.initializeDefaultObstacles();
  }

  private initializeDefaultObstacles() {
    // Vertical columns representing concrete pillars or tall dividers
    this.occlusions = [
      { x: 180, y: 0, w: 80, h: this.height, opacity: 0.75 },
      { x: 420, y: 0, w: 90, h: this.height, opacity: 0.75 }
    ];
  }

  public setDimensions(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.initializeDefaultObstacles();
  }

  /**
   * Resets and spawns synthetic objects with randomized bounding boxes, velocities, and labels
   */
  public populate(count: number = 5) {
    const classes: Array<SimulatedObject['class']> = ['person', 'car', 'bicycle', 'dog', 'backpack'];
    const colors = ['#a855f7', '#06b6d4', '#f43f5e', '#f59e0b', '#10b981'];
    
    this.objects = [];
    for (let i = 0; i < count; i++) {
      const cls = classes[i % classes.length];
      const w = cls === 'car' ? 85 : cls === 'person' ? 45 : 35;
      const h = cls === 'car' ? 55 : cls === 'person' ? 90 : 35;
      
      this.objects.push({
        id: this.nextId++,
        class: cls,
        x: Math.random() * (this.width - w - 10) + 5,
        y: Math.random() * (this.height - h - 10) + 5,
        width: w,
        height: h,
        vx: (Math.random() * 2 + 1) * (Math.random() > 0.5 ? 1 : -1),
        vy: (Math.random() * 1.5 + 0.5) * (Math.random() > 0.5 ? 1 : -1),
        color: colors[i % colors.length],
        isOccluded: false
      });
    }
  }

  /**
   * Updates coordinates of objects, handles collisions with walls, and determines if they are occluded.
   * Returns a list of "detections" that simulates what a neural network would output.
   * Detections are withheld if the object is highly occluded, simulating model failure!
   */
  public step(noiseFactor: number = 0.05): {
    actualObjects: SimulatedObject[];
    simulatedDetections: Detection[];
  } {
    const simulatedDetections: Detection[] = [];

    this.objects.forEach(obj => {
      // 1. Physics update: step position
      obj.x += obj.vx;
      obj.y += obj.vy;

      // Wall bounce collision logic
      if (obj.x <= 0) {
        obj.x = 0;
        obj.vx *= -1;
      } else if (obj.x + obj.width >= this.width) {
        obj.x = this.width - obj.width;
        obj.vx *= -1;
      }

      if (obj.y <= 0) {
        obj.y = 0;
        obj.vy *= -1;
      } else if (obj.y + obj.height >= this.height) {
        obj.y = this.height - obj.height;
        obj.vy *= -1;
      }

      // 2. Compute Occlusion Matrix
      // Check if the object center overlaps with any of the occlusion areas
      const cx = obj.x + obj.width / 2;
      const cy = obj.y + obj.height / 2;
      
      let inOcclusionZone = false;
      this.occlusions.forEach(occ => {
        if (cx >= occ.x && cx <= occ.x + occ.w && cy >= occ.y && cy <= occ.y + occ.h) {
          inOcclusionZone = true;
        }
      });

      obj.isOccluded = inOcclusionZone;

      // 3. Simulated model detection output
      // If object is NOT occluded, the neural net detects it with high probability (95%).
      // If object IS occluded, the detector fails to output it! This forces SORT to use velocity prediction.
      const isDetected = inOcclusionZone ? (Math.random() < 0.15) : (Math.random() < 0.98);

      if (isDetected) {
        // Add artificial neural network bounding-box noise/jitter to keep tracker validation grounded
        const jitterX = (Math.random() - 0.5) * noiseFactor * obj.width;
        const jitterY = (Math.random() - 0.5) * noiseFactor * obj.height;
        const jitterW = (Math.random() - 0.5) * noiseFactor * obj.width;
        const jitterH = (Math.random() - 0.5) * noiseFactor * obj.height;

        const jitteredBbox: [number, number, number, number] = [
          Math.max(0, Math.min(this.width, obj.x + jitterX)),
          Math.max(0, Math.min(this.height, obj.y + jitterY)),
          Math.max(10, Math.min(this.width, obj.width + jitterW)),
          Math.max(10, Math.min(this.height, obj.height + jitterH))
        ];

        simulatedDetections.push({
          bbox: jitteredBbox,
          class: obj.class,
          score: Math.min(0.99, Math.max(0.70, 0.92 + (Math.random() - 0.5) * 0.08))
        });
      }
    });

    return {
      actualObjects: this.objects,
      simulatedDetections
    };
  }

  /**
   * Explicitly updates simulated target density (adds or removes)
   */
  public updateObjectCount(newCount: number) {
    if (newCount === this.objects.length) return;
    if (newCount > this.objects.length) {
      const additional = newCount - this.objects.length;
      const classes: Array<SimulatedObject['class']> = ['person', 'car', 'bicycle', 'dog', 'backpack'];
      for (let i = 0; i < additional; i++) {
        const cls = classes[Math.floor(Math.random() * classes.length)];
        const w = cls === 'car' ? 85 : cls === 'person' ? 45 : 35;
        const h = cls === 'car' ? 55 : cls === 'person' ? 90 : 35;
        this.objects.push({
          id: this.nextId++,
          class: cls,
          x: Math.random() * (this.width - w - 10) + 5,
          y: Math.random() * (this.height - h - 10) + 5,
          width: w,
          height: h,
          vx: (Math.random() * 2 + 1) * (Math.random() > 0.5 ? 1 : -1),
          vy: (Math.random() * 1.5 + 0.5) * (Math.random() > 0.5 ? 1 : -1),
          color: '#' + Math.floor(Math.random()*16777215).toString(16),
          isOccluded: false
        });
      }
    } else {
      this.objects = this.objects.slice(0, newCount);
    }
  }
}

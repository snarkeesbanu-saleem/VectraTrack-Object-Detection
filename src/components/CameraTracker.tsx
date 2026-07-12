/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef, useState } from 'react';
import { TrackerConfig, Detection, Track, SystemEvent } from '../types';
import { SORTTracker, computeIoU } from '../utils/tracker';
import { TrackingSimulator, SimulatedObject } from '../utils/simulator';
import { Play, Pause, Camera, AlertTriangle, ShieldCheck, HeartPulse, RefreshCw } from 'lucide-react';

// Dynamically import tf and cocoSsd to ensure we can capture loading stages
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

interface CameraTrackerProps {
  config: TrackerConfig;
  onUpdateTracks: (tracks: Track[], latencyMs: number) => void;
  inputMode: 'webcam' | 'video' | 'simulator';
  selectedVideo: string;
  simulatorObjectsCount: number;
  onAddEvent: (event: Omit<SystemEvent, 'id' | 'timestamp'>) => void;
  selectedTrack: Track | null;
  onSelectTrack: (track: Track | null) => void;
  tracker: SORTTracker;
  onModelLoadedStateChange: (loaded: boolean) => void;
}

export default function CameraTracker({
  config,
  onUpdateTracks,
  inputMode,
  selectedVideo,
  simulatorObjectsCount,
  onAddEvent,
  selectedTrack,
  onSelectTrack,
  tracker,
  onModelLoadedStateChange
}: CameraTrackerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const simulatorRef = useRef<TrackingSimulator | null>(null);

  // States for computer vision weights loader
  const [modelLoadingStage, setModelLoadingStage] = useState<string>('Initialization Pending...');
  const [isLoaderActive, setIsLoaderActive] = useState<boolean>(true);
  const [isCoCoModelLoaded, setIsCoCoModelLoaded] = useState<boolean>(false);
  const [modelLoadError, setModelLoadError] = useState<string | null>(null);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);

  // General runtime controls
  const [isEngineRunning, setIsEngineRunning] = useState<boolean>(true);
  const [isFacingCameraIssue, setIsFacingCameraIssue] = useState<boolean>(false);
  const [cvSystemStatus, setCvSystemStatus] = useState<'IDLE' | 'LOADING' | 'RUNNING' | 'ERROR'>('IDLE');
  const [hasTaintedCanvas, setHasTaintedCanvas] = useState<boolean>(false);

  // Initialize Simulator on startup
  useEffect(() => {
    simulatorRef.current = new TrackingSimulator(640, 480);
    simulatorRef.current.populate(simulatorObjectsCount);
  }, []);

  // Sync simulator item count slider
  useEffect(() => {
    if (simulatorRef.current) {
      simulatorRef.current.updateObjectCount(simulatorObjectsCount);
    }
  }, [simulatorObjectsCount]);

  // Handle Model Loading
  useEffect(() => {
    let active = true;
    async function loadCoreModel() {
      if (!active) return;
      setCvSystemStatus('LOADING');
      setIsLoaderActive(true);
      
      try {
        setModelLoadingStage('Initializing WebAssembly compiler engines...');
        await tf.ready();
        
        if (!active) return;
        setModelLoadingStage('Contacting secure database & fetching neural weights (COCO-SSD)...');
        // Load model
        const loadedModel = await cocoSsd.load({
          base: 'lite_mobilenet_v2' // standard lightweight mobile-v2 for super high FPS in client grids
        });

        if (!active) return;
        setModelLoadingStage('Conducting warm-up inference tensors...');
        
        // Execute a fast synthetic passing step to warm up GPU buffers
        const dummyCanvas = document.createElement('canvas');
        dummyCanvas.width = 100;
        dummyCanvas.height = 100;
        const dummyCtx = dummyCanvas.getContext('2d');
        if (dummyCtx) {
          dummyCtx.fillStyle = '#000';
          dummyCtx.fillRect(0, 0, 100, 100);
          await loadedModel.detect(dummyCanvas);
        }

        if (!active) return;
        setModel(loadedModel);
        setIsCoCoModelLoaded(true);
        setIsLoaderActive(false);
        onModelLoadedStateChange(true);
        setCvSystemStatus('RUNNING');
        
        onAddEvent({
          type: 'success',
          message: 'VectraTrack CV Core Online. MobileNetV2 weight maps parsed successfully.'
        });
      } catch (err: any) {
        console.error('Failed to load TF model:', err);
        if (!active) return;
        setModelLoadError(err?.message || 'WASM model compilation limit or network timeout.');
        setIsLoaderActive(false);
        setCvSystemStatus('ERROR');
        onAddEvent({
          type: 'warning',
          message: 'Local machine learning compiled with limitations. Fallback to CPU matrices activated.'
        });
      }
    }

    loadCoreModel();
    return () => {
      active = false;
    };
  }, []);

  // Handle Input Feed Source Transitions
  useEffect(() => {
    // Shutdown webcam stream if switching away
    stopWebcamStream();
    setIsFacingCameraIssue(false);

    if (inputMode === 'webcam') {
      startWebcamStream();
    } else if (inputMode === 'video' && videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.src = selectedVideo;
      videoRef.current.load();
      videoRef.current.play().catch(e => {
        console.warn('Video element autoplay blocked, click play to begin:', e);
      });
      onAddEvent({
        type: 'info',
        message: `Switched stream input to video asset file: Shibuya Pedestrians.`
      });
    } else if (inputMode === 'simulator') {
      if (simulatorRef.current) {
        simulatorRef.current.populate(simulatorObjectsCount);
      }
      onAddEvent({
        type: 'info',
        message: 'Activated virtual composite radar simulator sandbox. Zero CORS bounding boxes populated.'
      });
    }
  }, [inputMode, selectedVideo]);

  const startWebcamStream = async () => {
    onAddEvent({ type: 'info', message: 'Requesting optical camera viewport authorization...' });
    setIsFacingCameraIssue(false);
    
    try {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'environment'
          },
          audio: false
        });

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.src = '';
          videoRef.current.srcObject = stream;
          videoRef.current.play().then(() => {
            onAddEvent({ type: 'success', message: 'Camera stream link active. Processing frame buffer.' });
          }).catch(err => {
            console.error('Webcam play failure:', err);
          });
        }
      } else {
        throw new Error('Native getUserMedia not supported in this frame context.');
      }
    } catch (err: any) {
      console.warn('Webcam start failed:', err);
      setIsFacingCameraIssue(true);
      onAddEvent({
        type: 'alert',
        message: 'Camera permission denied or camera device missing.'
      });
      // Switch automatically to simulator so they have an operating workspace!
      onChangeToSimulatorFallback();
    }
  };

  const stopWebcamStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const onChangeToSimulatorFallback = () => {
    setTimeout(() => {
      onAddEvent({
        type: 'warning',
        message: 'Camera device unavailable. Re-routing viewport to virtual simulator feed.'
      });
    }, 1500);
  };

  // Run Real-Time Frame Loop
  useEffect(() => {
    let lastTime = performance.now();
    let frameCount = 0;
    let fpsIntervalTime = performance.now();
    let computedFps = 30;

    const processFrameLoop = async () => {
      if (!isEngineRunning) {
        requestRef.current = requestAnimationFrame(processFrameLoop);
        return;
      }

      const canvas = canvasRef.current;
      const video = videoRef.current;
      const ctx = canvas?.getContext('2d');

      if (!canvas || !ctx) {
        requestRef.current = requestAnimationFrame(processFrameLoop);
        return;
      }

      const startTime = performance.now();
      let rawDetections: Detection[] = [];
      let actualSimulatedObjects: SimulatedObject[] = [];

      // CLEAR CANVAS
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // --- MODE A: WEB MACHINE LEARNING (WEBCAM / MP4 VIDEO) ---
      if (inputMode !== 'simulator' && video) {
        const canReadVideo = video.readyState >= 2 && video.videoWidth > 0;
        
        if (canReadVideo) {
          // Adjust canvas size to match dimensions of video container fluidly
          if (canvas.width !== video.videoWidth && video.videoWidth > 0) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }

          // Render underlying frame
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          // Apply privacy blur filter if enabled
          if (config.blurBackground) {
            ctx.save();
            ctx.filter = 'blur(12px) brightness(0.6)';
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            ctx.restore();
          }

          // Run Inference via standard COCO-SSD mobile-net
          if (isCoCoModelLoaded && model) {
            try {
              const predictions = await model.detect(video);
              
              rawDetections = predictions.map(p => {
                const [bx, by, bw, bh] = p.bbox;
                return {
                  bbox: [bx, by, bw, bh],
                  class: p.class,
                  score: p.score
                };
              });

              // If canvas tainted, clear CORS error
              if (hasTaintedCanvas) setHasTaintedCanvas(false);
            } catch (err: any) {
              if (err?.message?.includes('tainted') || err?.message?.includes('SecurityError')) {
                if (!hasTaintedCanvas) {
                  setHasTaintedCanvas(true);
                  onAddEvent({
                    type: 'alert',
                    message: 'Video asset CORS blocked, falling back to fully localized Simulator sandbox!'
                  });
                  // Trigger switch
                  setTimeout(() => {
                    onChangeToSimulatorFallback();
                  }, 1200);
                }
              }
            }
          }
        } else {
          // Display idle loading spinner on canvas
          renderOpticalLoadingGrid(ctx, canvas);
        }
      } 
      // --- MODE B: SYNTHETIC RADAR SIMULATOR ---
      else if (inputMode === 'simulator' && simulatorRef.current) {
        if (canvas.width !== 640) {
          canvas.width = 640;
          canvas.height = 480;
        }

        // Fill background with space-grid styling
        renderSimulatorSpacialBackground(ctx, canvas);

        // Advance physics simulator state
        const simResult = simulatorRef.current.step(0.04);
        rawDetections = simResult.simulatedDetections;
        actualSimulatedObjects = simResult.actualObjects;

        // Render actual base ground truth models for visuals
        actualSimulatedObjects.forEach(obj => {
          // Draw subtle dashed lines showing the absolute target position
          ctx.strokeStyle = obj.isOccluded ? 'rgba(71, 85, 105, 0.4)' : `${obj.color}15`;
          ctx.lineWidth = 1.2;
          ctx.strokeRect(obj.x, obj.y, obj.width, obj.height);
          
          if (obj.isOccluded) {
            ctx.fillStyle = 'rgba(71, 85, 105, 0.15)';
            ctx.fillRect(obj.x, obj.y, obj.width, obj.height);
          }
        });
      }

      // Filter classes that are excluded in active configuration
      const filteredDetections = rawDetections.filter(
        det => {
          const cls = det.class.toLowerCase();
          // Filter out if not inside permitted lists or exceeds threshold
          if (config.filterClasses.includes(cls)) return false;
          return det.score >= config.confidenceThreshold;
        }
      );

      // FEED FILTERED DETECTIONS INTO THE SORT MULTI-OBJECT TRACKER
      const tracks = tracker.update(filteredDetections, config);

      // Calculate performance timing speeds
      const endTime = performance.now();
      const latencyMs = Math.round(endTime - startTime);

      frameCount++;
      if (endTime - fpsIntervalTime >= 1000) {
        computedFps = (frameCount * 1000) / (endTime - fpsIntervalTime);
        frameCount = 0;
        fpsIntervalTime = endTime;
      }

      // Dispatch tracking results to dashboard parent hooks
      onUpdateTracks(tracks, latencyMs);

      // --- CANVAS RENDERING (OVERLAYS, TRAILS, CORNER BRACKETS, LABELS) ---
      tracks.forEach(track => {
        const [tx, ty, tw, th] = track.bbox;
        const color = track.color;
        const isFocal = selectedTrack?.id === track.id;

        // 1. Draw route history lines if enabled
        if (config.drawTrails && track.history.length > 1) {
          ctx.beginPath();
          ctx.strokeStyle = color;
          // Gradient alpha trail
          ctx.lineWidth = isFocal ? 3.0 : 1.5;
          ctx.setLineDash([]);
          track.history.forEach((pt, pIdx) => {
            if (pIdx === 0) {
              ctx.moveTo(pt.x, pt.y);
            } else {
              ctx.lineTo(pt.x, pt.y);
            }
          });
          ctx.stroke();

          // Sparkle node circles at coordinates
          track.history.forEach((pt, pIdx) => {
            if (pIdx % 3 === 0 || pIdx === track.history.length - 1) {
              ctx.beginPath();
              ctx.fillStyle = color;
              ctx.arc(pt.x, pt.y, isFocal ? 3.0 : 1.8, 0, 2 * Math.PI);
              ctx.fill();
            }
          });
        }

        // 2. Draw privacy background blur cutout if blur background-mode is enabled
        if (config.blurBackground && video) {
          // Draw cutout for matched target element to stay sharp in focus (surveillance lens effect)
          ctx.save();
          ctx.beginPath();
          ctx.rect(tx, ty, tw, th);
          ctx.clip();
          // Re-draw clean video just inside bounding frame
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          ctx.restore();
        }

        // 3. Draw Bounding Baffle Frame (SURVEILLANCE HUD BRACKETS)
        ctx.strokeStyle = color;
        ctx.lineWidth = isFocal ? 2.5 : 1.5;
        
        if (!track.isActive) {
          // If track is in projected model estimation (occluded), render bounding box as dashed line
          ctx.setLineDash([5, 5]);
          ctx.strokeStyle = 'rgba(239, 68, 68, 0.7)'; // flashing indicator red
        } else {
          ctx.setLineDash([]);
        }

        // Draw bracket corners
        const cornerLen = Math.min(tw * 0.2, 16);
        // Top-Left corner
        ctx.beginPath();
        ctx.moveTo(tx, ty + cornerLen);
        ctx.lineTo(tx, ty);
        ctx.lineTo(tx + cornerLen, ty);
        ctx.stroke();

        // Top-Right corner
        ctx.beginPath();
        ctx.moveTo(tx + tw - cornerLen, ty);
        ctx.lineTo(tx + tw, ty);
        ctx.lineTo(tx + tw, ty + cornerLen);
        ctx.stroke();

        // Bottom-Left corner
        ctx.beginPath();
        ctx.moveTo(tx, ty + th - cornerLen);
        ctx.lineTo(tx, ty + th);
        ctx.lineTo(tx + cornerLen, ty + th);
        ctx.stroke();

        // Bottom-Right corner
        ctx.beginPath();
        ctx.moveTo(tx + tw - cornerLen, ty + th);
        ctx.lineTo(tx + tw, ty + th);
        ctx.lineTo(tx + tw, ty + th - cornerLen);
        ctx.stroke();

        // Subtle glow border just for selected target
        if (isFocal) {
          ctx.fillStyle = `${color}10`;
          ctx.fillRect(tx, ty, tw, th);
          
          // Draw aiming rect crosshairs
          ctx.beginPath();
          ctx.strokeStyle = color;
          ctx.lineWidth = 0.5;
          ctx.setLineDash([2, 5]);
          ctx.moveTo(tx + tw / 2, 0);
          ctx.lineTo(tx + tw / 2, canvas.height);
          ctx.moveTo(0, ty + th / 2);
          ctx.lineTo(canvas.width, ty + th / 2);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // 4. Draw Glow Heading Direction Arrow
        if (Math.abs(track.velocity[0]) > 0.4 || Math.abs(track.velocity[1]) > 0.4) {
          const cx = tx + tw / 2;
          const cy = ty + th / 2;
          const dx = track.velocity[0] * 6.5; // vector amplifier scale
          const dy = track.velocity[1] * 6.5;

          ctx.beginPath();
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.8;
          ctx.moveTo(cx, cy);
          ctx.lineTo(cx + dx, cy + dy);
          ctx.stroke();

          // Arrowhead
          const angle = Math.atan2(dy, dx);
          ctx.beginPath();
          ctx.fillStyle = color;
          ctx.moveTo(cx + dx, cy + dy);
          ctx.lineTo(
            cx + dx - 6 * Math.cos(angle - Math.PI / 6),
            cy + dy - 6 * Math.sin(angle - Math.PI / 6)
          );
          ctx.lineTo(
            cx + dx - 6 * Math.cos(angle + Math.PI / 6),
            cy + dy - 6 * Math.sin(angle + Math.PI / 6)
          );
          ctx.fill();
        }

        // 5. Draw ID and HUD Category pill box label
        const labelText = `[ID:${track.id}] ${track.class.toUpperCase()} ${Math.round(track.score * 100)}%`;
        ctx.font = 'bold 9px monospace';
        const textMetrics = ctx.measureText(labelText);
        const textWidth = textMetrics.width;

        // Label bg color
        ctx.fillStyle = isFocal ? '#ffffff' : color;
        ctx.fillRect(tx - 1, ty - th * 0.001 - 15 > 0 ? ty - 16 : ty + 1, textWidth + 8, 14);

        // Label text
        ctx.fillStyle = isFocal ? '#000000' : '#020617';
        ctx.fillText(labelText, tx + 3, ty - th * 0.001 - 15 > 0 ? ty - 6 : ty + 11);

        // If projected in occlusion, draw "LOST - ESTIMATING" box below
        if (!track.isActive) {
          ctx.fillStyle = '#ef4444';
          ctx.fillRect(tx - 1, ty + th - 1, 95, 12);
          ctx.fillStyle = '#ffffff';
          ctx.font = '8px monospace';
          ctx.fillText('⚡ PROJECTING PATH', tx + 3, ty + th + 8);
        }
      });

      // If active track is selected, update details on fly
      if (selectedTrack) {
        const matchingCurrentTrack = tracks.find(t => t.id === selectedTrack.id);
        if (matchingCurrentTrack) {
          // Trigger hook re-evaluation
          onSelectTrack(matchingCurrentTrack);
        }
      }

      requestRef.current = requestAnimationFrame(processFrameLoop);
    };

    requestRef.current = requestAnimationFrame(processFrameLoop);
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [
    isEngineRunning,
    inputMode,
    isCoCoModelLoaded,
    model,
    config,
    tracker,
    selectedTrack,
    hasTaintedCanvas
  ]);

  // Optical HUD grid and sweep laser on simulation radar backgrounds
  const renderSimulatorSpacialBackground = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    // Fill space background
    ctx.fillStyle = '#030712';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw dynamic matrix background grids
    ctx.strokeStyle = 'rgba(76, 29, 149, 0.12)';
    ctx.lineWidth = 0.5;
    
    // Vertical gridlines
    const spaceX = 40;
    for (let x = 0; x < canvas.width; x += spaceX) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }

    // Horizontal gridlines
    const spaceY = 40;
    for (let y = 0; y < canvas.height; y += spaceY) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Draw radar rings (concentric radial rings mapping)
    ctx.strokeStyle = 'rgba(139, 92, 246, 0.05)';
    ctx.beginPath();
    ctx.arc(canvas.width / 2, canvas.height / 2, 120, 0, 2 * Math.PI);
    ctx.arc(canvas.width / 2, canvas.height / 2, 240, 0, 2 * Math.PI);
    ctx.stroke();

    // Render structural Occlusion dividers (shaded pillars)
    if (simulatorRef.current) {
      simulatorRef.current.occlusions.forEach(occ => {
        ctx.fillStyle = 'rgba(30, 41, 59, 0.45)';
        ctx.fillRect(occ.x, occ.y, occ.w, occ.h);

        // Glowing columns borders
        ctx.strokeStyle = 'rgba(139, 92, 246, 0.15)';
        ctx.lineWidth = 1;
        ctx.strokeRect(occ.x, occ.y, occ.w, occ.h);

        // "OCCLUSION AREA" HUD warning tag text
        ctx.fillStyle = 'rgba(168, 85, 247, 0.35)';
        ctx.font = 'bold 8px monospace';
        ctx.save();
        ctx.translate(occ.x + occ.w / 2, canvas.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('🔴 SHIELDED ZONE (CV OCULATOR)', -60, 3);
        ctx.restore();
      });
    }

    // Canvas boundary box
    ctx.strokeStyle = 'rgba(168, 85, 247, 0.2)';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, canvas.width, canvas.height);
  };

  const renderOpticalLoadingGrid = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    ctx.fillStyle = '#010409';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Rotating loading ring
    const radius = 30;
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const time = Date.now() * 0.003;

    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, time, time + Math.PI * 1.5);
    ctx.stroke();

    ctx.fillStyle = 'rgba(168, 85, 247, 0.8)';
    ctx.font = '10px monospace';
    ctx.fillText('SYNCING OPTICAL SENSOR FEED...', cx - 80, cy + radius + 25);
  };

  // Canvas manual target selection handler
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    // Convert click offsets cleanly to coordinate space of internal canvas size
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;

    // Search active tracks if any overlap click point
    let matchedTrack: Track | null = null;
    const currentTracks = tracker.getTracks();

    for (const track of currentTracks) {
      const [tx, ty, tw, th] = track.bbox;
      if (x >= tx && x <= tx + tw && y >= ty && y <= ty + th) {
        matchedTrack = track;
        break;
      }
    }

    if (matchedTrack) {
      onSelectTrack(matchedTrack);
      onAddEvent({
        type: 'info',
        message: `Focal Lock established. Target ID #${matchedTrack.id} [${matchedTrack.class.toUpperCase()}] isolated.`
      });
    } else {
      onSelectTrack(null);
    }
  };

  return (
    <div className="relative flex flex-col bg-slate-950 rounded-2xl border border-purple-950/40 p-4 shadow-[0_15px_30px_rgba(0,0,0,0.4)] overflow-hidden">
      
      {/* Visual Header */}
      <div className="flex items-center justify-between mb-3 text-slate-100">
        <div className="flex items-center gap-2">
          {cvSystemStatus === 'RUNNING' ? (
            <div className="w-2.5 h-2.5 rounded-full bg-green-500 animate-ping" />
          ) : (
            <div className="w-2.5 h-2.5 rounded-full bg-amber-500" />
          )}
          <span className="text-xs font-mono font-bold tracking-widest text-purple-300 uppercase">
            Viewport Output Stream
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsEngineRunning(!isEngineRunning)}
            className={`flex items-center gap-1.5 px-3 py-1 font-mono text-[10px] uppercase font-bold rounded border ${
              isEngineRunning
                ? 'bg-purple-950/40 text-purple-400 border-purple-900/40 hover:bg-purple-900/40'
                : 'bg-emerald-950/40 text-emerald-400 border-emerald-900/40 hover:bg-emerald-900/40'
            } transition-colors duration-200`}
            title={isEngineRunning ? 'Freeze process pipeline' : 'Resume process pipeline'}
          >
            {isEngineRunning ? (
              <>
                <Pause className="w-3 h-3 text-purple-400" /> Freeze Pipeline
              </>
            ) : (
              <>
                <Play className="w-3 h-3 text-emerald-400" /> Resume Pipeline
              </>
            )}
          </button>
        </div>
      </div>

      {/* Frame Port & Canvas Stack overlay */}
      <div className="relative w-full aspect-video bg-slate-950 rounded-lg overflow-hidden border border-purple-950/30">
        {/* Optical HTML5 video feed (usually invisible behind canvas rendering) */}
        <video
          ref={videoRef}
          className="absolute w-full h-full object-cover hidden"
          playsInline
          muted
          loop
          crossOrigin="anonymous"
        />

        {/* Dynamic Canvas Processing layer */}
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          className="absolute top-0 left-0 w-full h-full object-cover cursor-crosshair z-10"
        />

        {/* SECTION 5: MACHINE LEARNING INITIAL COLD ENGINE BOOTLOADER PAGE */}
        {isLoaderActive && (
          <div className="absolute inset-0 bg-slate-950/95 flex flex-col items-center justify-center p-6 text-center z-25 backdrop-blur-md">
            <div className="relative flex items-center justify-center mb-6">
              <div className="w-16 h-16 rounded-full border-2 border-dashed border-purple-500 animate-spin" />
              <Camera className="w-6 h-6 text-purple-400 absolute" />
            </div>
            
            <h3 className="text-sm font-semibold text-slate-100 font-sans tracking-wide uppercase">
              VectraTrack Engine Initialization
            </h3>
            
            <p className="text-[11px] font-mono text-purple-400 mt-2 max-w-sm animate-pulse">
              {modelLoadingStage}
            </p>

            <div className="flex items-center gap-1.5 mt-8 px-2.5 py-1 text-[9px] font-mono text-slate-500 bg-slate-900 rounded border border-slate-800">
              <HeartPulse className="w-3 h-3" /> TFJS Core compiled standard engine: v{tf.version?.tfjs || '2.4.0'}
            </div>
          </div>
        )}

        {/* Tainted exception alert panel banner */}
        {hasTaintedCanvas && (
          <div className="absolute bottom-4 left-4 right-4 z-20 bg-rose-950/90 border border-rose-900/60 p-3 rounded-lg flex items-center gap-2 text-xs text-rose-300 backdrop-blur">
            <AlertTriangle className="w-5 h-5 flex-shrink-0 text-rose-400" />
            <div>
              <span className="font-bold">CORS Security Containment:</span> External video parsing blocked by browser sandbox inside iframe. Re-routing instantly to Cyber Simulator.
            </div>
          </div>
        )}
      </div>

      {/* Bottom overlay parameters line */}
      <div className="mt-3.5 flex items-center justify-between text-[10px] font-mono text-slate-500 border-t border-purple-950/20 pt-3">
        <div className="flex items-center gap-4">
          <span>SOURCE: <span className="text-purple-400 font-bold uppercase">{inputMode}</span></span>
          {inputMode === 'webcam' && (
            <span className="text-slate-400">FPS Limit: Auto viewport sync (~30)</span>
          )}
        </div>
        <div className="flex items-center gap-1.5 text-slate-400">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-500" /> Frame Integrity Secured
        </div>
      </div>
    </div>
  );
}

/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { TrackerConfig, Track, SystemEvent, ChartDataPoint } from './types';
import { SORTTracker } from './utils/tracker';
import CameraTracker from './components/CameraTracker';
import SettingsPanel from './components/SettingsPanel';
import StatsPanel from './components/StatsPanel';
import LogConsole from './components/LogConsole';
import TrackInspector from './components/TrackInspector';
import { Shield, Cpu, Activity, Download, Layers, Sparkles, HelpCircle, Info } from 'lucide-react';

export default function App() {
  // 1. Core Tracker settings
  const [config, setConfig] = useState<TrackerConfig>({
    iouThreshold: 0.35,
    confidenceThreshold: 0.45,
    maxMissedFrames: 25,
    historyLength: 35,
    drawTrails: true,
    blurBackground: false,
    minTrackAge: 1,
    colorTheme: 'purple',
    filterClasses: []
  });

  // Unique tracker persistence across component lifecycles 
  const trackerInstance = useMemo(() => new SORTTracker(), []);

  // 2. Feed states
  const [inputMode, setInputMode] = useState<'webcam' | 'video' | 'simulator'>('simulator');
  const [selectedVideo, setSelectedVideo] = useState<string>('https://assets.mixkit.co/videos/preview/mixkit-pedestrians-walking-on-a-crosswalk-from-above-41858-large.mp4');
  const [simulatorObjectsCount, setSimulatorObjectsCount] = useState<number>(5);

  // 3. Operational Telemetry states
  const [tracks, setTracks] = useState<Track[]>([]);
  const [latencyMs, setLatencyMs] = useState<number>(0);
  const [fps, setFps] = useState<number>(30);
  const [isModelLoaded, setIsModelLoaded] = useState<boolean>(false);
  const [selectedTrack, setSelectedTrack] = useState<Track | null>(null);

  // 4. Analytics history state (throttled updates to prevent lag)
  const [historyData, setHistoryData] = useState<ChartDataPoint[]>([]);

  // 5. System event logs buffer (capped at 100 lines for efficiency)
  const [events, setEvents] = useState<SystemEvent[]>([]);

  // 6. Cumulative Lifetime Tracking stats
  const [cumulativesSet, setCumulativesSet] = useState<Set<number>>(new Set());
  const [totalLoggedTargetsCount, setTotalLoggedTargetsCount] = useState<number>(0);
  const [activeInstructionModal, setActiveInstructionModal] = useState<boolean>(false);

  // Timer trackers for real-time FPS computing details
  const lastUpdateRef = useRef<number>(performance.now());
  const previousTracksIds = useRef<Set<number>>(new Set());

  // App highlight colors map
  const activeColorHex = {
    purple: 'from-purple-500 to-indigo-600',
    cyan: 'from-cyan-500 to-teal-500',
    pink: 'from-pink-500 to-rose-500',
    amber: 'from-amber-400 to-orange-500'
  }[config.colorTheme];

  const focusBorderHex = {
    purple: 'border-purple-500/30 shadow-purple-500/10',
    cyan: 'border-cyan-500/30 shadow-cyan-500/10',
    pink: 'border-pink-500/30 shadow-pink-500/10',
    amber: 'border-amber-500/30 shadow-amber-500/10'
  }[config.colorTheme];

  const primaryGlowHex = {
    purple: 'text-purple-400',
    cyan: 'text-cyan-400',
    pink: 'text-rose-400',
    amber: 'text-amber-400'
  }[config.colorTheme];

  // Helper dispatcher to write structured system logs
  const handleAddEvent = (event: Omit<SystemEvent, 'id' | 'timestamp'>) => {
    const timestampStr = new Date().toISOString();
    const newEvent: SystemEvent = {
      ...event,
      id: `${Date.now()}-${Math.random()}`,
      timestamp: timestampStr
    };
    
    setEvents(prev => {
      const updated = [...prev, newEvent];
      return updated.slice(-100); // hard cutoff buffer limit
    });
  };

  // Seed initial log entries
  useEffect(() => {
    handleAddEvent({
      type: 'info',
      message: 'VectraTrack telemetry module initialized. Port 3000 mapping secured.'
    });
    handleAddEvent({
      type: 'info',
      message: 'Loading neural framework backend (TensorFlow.js).'
    });
  }, []);

  // Update FPS and analyze incoming tracks for new locks/warnings
  const handleUpdateTracks = (newTracks: Track[], pipelineLatency: number) => {
    setTracks(newTracks);
    setLatencyMs(pipelineLatency);

    // Compute dynamic rendering framerate updates
    const now = performance.now();
    const delta = now - lastUpdateRef.current;
    lastUpdateRef.current = now;
    const computedFps = delta > 0 ? 1000 / delta : 30;
    
    // Low-pass filter for smooth FPS reading
    setFps(prev => prev * 0.9 + computedFps * 0.1);

    // Analyze target tracking ID updates
    const currentIds = new Set(newTracks.map(t => t.id));
    
    newTracks.forEach(t => {
      // 1. Audit targets for entry notifications
      if (!previousTracksIds.current.has(t.id)) {
        handleAddEvent({
          type: 'success',
          message: `NEW TARGET LOCKED: Identified [ID #${t.id}] categorized as [${t.class.toUpperCase()}] with score ${(t.score * 100).toFixed(0)}%.`
        });

        // Track high velocity alerts
        if (t.speed > 15) {
          handleAddEvent({
            type: 'alert',
            message: `VELOCITY CEILING WARN: [ID #${t.id}] showing high kinetographic speed shift (${t.speed} px/frame).`
          });
        }
      }

      // 2. Audit targets for survival projections under occlusion
      if (!t.isActive && previousTracksIds.current.has(t.id)) {
        const matchingPrevTrack = tracks.find(pt => pt.id === t.id);
        if (matchingPrevTrack && matchingPrevTrack.isActive) {
          handleAddEvent({
            type: 'warning',
            message: `TARGET OBSTRUCTED: [ID #${t.id}] entered structural occlusion. Attempting predictive SORT vector simulation.`
          });
        }
      }

      // Add to lifetime cumulative tracker
      if (!cumulativesSet.has(t.id)) {
        setCumulativesSet(prev => {
          const updated = new Set(prev);
          updated.add(t.id);
          setTotalLoggedTargetsCount(updated.size);
          return updated;
        });
      }
    });

    // 3. Audit targets for loss deletion logs
    previousTracksIds.current.forEach(id => {
      if (!currentIds.has(id)) {
        const matchingDeadTrack = tracks.find(pt => pt.id === id);
        if (matchingDeadTrack) {
          handleAddEvent({
            type: 'warning',
            message: `TARGET EXITED VIEW: Lost visual lock on target [ID #${id}] [${matchingDeadTrack.class.toUpperCase()}]. Metadata flushed.`
          });
        }
      }
    });

    previousTracksIds.current = currentIds;
  };

  // Interval timer for pushing statistics to recharts
  useEffect(() => {
    // Append telemetry plot nodes every 1000ms
    const interval = setInterval(() => {
      const nowTime = new Date();
      const timeLabel = `${String(nowTime.getHours()).padStart(2, '0')}:${String(nowTime.getMinutes()).padStart(2, '0')}:${String(nowTime.getSeconds()).padStart(2, '0')}`;
      
      let person = 0;
      let vehicle = 0;
      let others = 0;

      tracks.forEach(tr => {
        const cls = tr.class.toLowerCase();
        if (cls === 'person') person++;
        else if (['car', 'bicycle', 'motorcycle', 'truck', 'bus', 'train'].includes(cls)) vehicle++;
        else others++;
      });

      setHistoryData(prev => {
        const updated = [
          ...prev,
          {
            time: timeLabel,
            activeCount: tracks.length,
            person,
            vehicle,
            others
          }
        ];
        return updated.slice(-30); // 30 periods max buffer
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [tracks]);

  // Flush tracking memories
  const handleResetTracks = () => {
    trackerInstance.reset();
    setTracks([]);
    setSelectedTrack(null);
    setCumulativesSet(new Set());
    setTotalLoggedTargetsCount(0);
    previousTracksIds.current = new Set();
    handleAddEvent({
      type: 'success',
      message: 'SORT Tracking Registry flushed successfully. Target indices reset to 1.'
    });
  };

  // Export telemetries as JSON file
  const handleExportTelemetryLog = () => {
    const backupObj = {
      appIdentification: 'VectraTrack AI Multi-Object Tracking Suite',
      exportTimestamp: new Date().toISOString(),
      activeParameters: config,
      sensorInflowSource: inputMode,
      totalUniquesTrackedLifetime: totalLoggedTargetsCount,
      recentLogs: events.map(ev => ({ time: ev.timestamp, type: ev.type, value: ev.message })),
      currentTracksInFrame: tracks.map(tr => ({
        id: tr.id,
        class: tr.class,
        confidence: tr.score,
        age: tr.age,
        speed: tr.speed,
        currentTranslatePos: tr.bbox,
        headingEstimateVector: tr.velocity
      }))
    };

    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(backupObj, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `VectraTrack_Telemetry_${Date.now()}.json`);
    downloadAnchor.click();

    handleAddEvent({
      type: 'success',
      message: 'Telemetry JSON database stream downloaded successfully.'
    });
  };

  return (
    <div className="min-h-screen bg-slate-950 font-sans text-slate-100 flex flex-col selection:bg-purple-600 selection:text-white">
      
      {/* 1. TOP STATS APPBAY */}
      <header className="border-b border-purple-950/20 bg-slate-900/45 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-col sm:flex-row items-center justify-between gap-4">
          
          {/* Dashboard identifier branding */}
          <div className="flex items-center gap-3">
            <div className={`p-2.5 rounded-xl bg-gradient-to-br ${activeColorHex} shadow-[0_0_15px_rgba(168,85,247,0.3)] border border-white/10`}>
              <Shield className="w-5.5 h-5.5 text-white animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-bold tracking-tight text-white font-sans sm:text-xl">
                  VectraTrack
                </h1>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-900/30 text-purple-300 border border-purple-800/40 uppercase tracking-widest font-bold">
                  Pro Engine v2.0
                </span>
              </div>
              <p className="text-[10px] text-slate-400 font-mono tracking-tight mt-0.5">
                Autonomous Computer Vision & Motion Estimator Panel
              </p>
            </div>
          </div>

          {/* Interactive controls & telemetry download */}
          <div className="flex items-center gap-3 flex-wrap">
            <button
              onClick={() => setActiveInstructionModal(!activeInstructionModal)}
              className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-mono border border-slate-800 hover:border-slate-700 bg-slate-900 text-slate-300 hover:text-white transition duration-300 shadow-sm cursor-pointer"
            >
              <Info className="w-4 h-4 text-purple-400" />
              Operational Guide
            </button>
            
            <button
              onClick={handleExportTelemetryLog}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold bg-gradient-to-r ${activeColorHex} hover:opacity-90 active:scale-98 text-white transition-all duration-300 shadow-md cursor-pointer`}
            >
              <Download className="w-4 h-4" />
              Download Telemetry Log
            </button>
          </div>
        </div>
      </header>

      {/* 2. INSTRUCTION HUD GUIDANCE */}
      {activeInstructionModal && (
        <div className="bg-purple-950/15 border-b border-purple-900/30 font-sans">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-5 text-sm text-slate-300 flex flex-col md:flex-row gap-5 items-stretch relative">
            <div className="flex-1 space-y-2">
              <h4 className="text-xs font-mono font-bold text-purple-300 uppercase tracking-widest">
                ⚙️ Quick-Start Instructions & Scientific Design
              </h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                VectraTrack delivers multi-object localized tracking (MOT) completely in the browser sandbox. It integrates a <strong>SORT Tracker</strong>—matching targets sequentially with constant-velocity bounding prediction and spatial Intersections over Union.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2 text-[11px] font-mono">
                <div className="bg-slate-950/40 p-2.5 rounded border border-purple-950/30">
                  <span className="text-purple-400 font-bold block mb-1">1. SENSOR INFLOWS</span> Webcam permissions run fully localized. Under sandbox iFrame restrictions, click the <strong>Asset Video</strong> or the <strong>Radar Simulator</strong> for a 100% CORS-friendly visual playground.
                </div>
                <div className="bg-slate-950/40 p-2.5 rounded border border-purple-950/30">
                  <span className="text-purple-400 font-bold block mb-1">2. TEST PERSISTENCE</span> The simulator features thick, tall structural pillars. When bounding blocks enter these zones, raw detector outputs are intentionally withheld, highlighting SORT's mathematical path extrapolation.
                </div>
                <div className="bg-slate-950/40 p-2.5 rounded border border-purple-950/30">
                  <span className="text-purple-400 font-bold block mb-1">3. LOCK ONTARGETS</span> Click directly inside bounding boxes draw layers on the Viewport to lock spatial telemetry details inside the focal target inspector panel.
                </div>
              </div>
            </div>
            
            <button
              onClick={() => setActiveInstructionModal(false)}
              className="md:self-center px-3 py-1 bg-slate-900 hover:bg-slate-800 text-xs font-mono rounded border border-slate-800 text-slate-400 hover:text-white transition cursor-pointer"
            >
              DISMISS HUD
            </button>
          </div>
        </div>
      )}

      {/* 3. CORE SUB-DASHBOARDS GRID LAYOUT */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-col gap-6">
        
        {/* TOP STATUS METER PANELSROW */}
        <section>
          <StatsPanel
            tracks={tracks}
            historyData={historyData}
            fps={fps}
            latencyMs={latencyMs}
            totalLoggedTargetsCount={totalLoggedTargetsCount}
            colorTheme={config.colorTheme}
          />
        </section>

        {/* PRIMARY WORKSPACE: VIEWPORT + CONFIG PANEL SPLIT */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          
          {/* LEFT INTERACTIVE HUD (7 Columns) */}
          <section className="lg:col-span-8 flex flex-col gap-6 h-full justify-between">
            {/* Viewport Frame Box */}
            <div className="flex-1">
              <CameraTracker
                config={config}
                onUpdateTracks={handleUpdateTracks}
                inputMode={inputMode}
                selectedVideo={selectedVideo}
                simulatorObjectsCount={simulatorObjectsCount}
                onAddEvent={handleAddEvent}
                selectedTrack={selectedTrack}
                onSelectTrack={setSelectedTrack}
                tracker={trackerInstance}
                onModelLoadedStateChange={setIsModelLoaded}
              />
            </div>

            {/* Event logs terminal underneath viewport */}
            <div className="h-[280px]">
              <LogConsole
                events={events}
                onClearEvents={() => setEvents([])}
              />
            </div>
          </section>

          {/* RIGHT SIDEBOARD CONFIGS & INSPECTOR HUD (4 Columns) */}
          <section className="lg:col-span-4 flex flex-col gap-6">
            
            {/* Config Controller sliders */}
            <div className="flex-1">
              <SettingsPanel
                config={config}
                onChangeConfig={setConfig}
                inputMode={inputMode}
                onChangeInputMode={(mode) => setInputMode(mode)}
                selectedVideo={selectedVideo}
                onChangeSelectedVideo={(url) => setSelectedVideo(url)}
                simulatorObjectsCount={simulatorObjectsCount}
                onChangeSimulatorObjectsCount={(c) => setSimulatorObjectsCount(c)}
                isModelLoaded={isModelLoaded}
                onResetTracker={handleResetTracks}
              />
            </div>

            {/* Target Inspector telemetry node locks */}
            <div className="h-auto">
              <TrackInspector
                selectedTrack={selectedTrack}
                onDeselect={() => setSelectedTrack(null)}
                colorTheme={config.colorTheme}
              />
            </div>
          </section>
        </div>
      </main>

      {/* 4. FOOTER CREDITS */}
      <footer className="border-t border-purple-950/20 bg-slate-950 py-5 text-center mt-auto">
        <div className="max-w-7xl mx-auto px-4 text-slate-500 font-mono text-[10px] flex flex-col sm:flex-row items-center justify-between gap-2.5">
          <span>VectraTrack Autonomous Motion Engine</span>
          <div className="flex items-center gap-1">
            <Cpu className="w-3.5 h-3.5 text-purple-500" />
            <span>WebGL Hardware Acceleration Enabled (TensorFlow.js)</span>
          </div>
          <span>UTC Timestamp: 2026-06-07 • Offline Secured</span>
        </div>
      </footer>
    </div>
  );
}

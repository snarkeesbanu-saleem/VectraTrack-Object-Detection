/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { TrackerConfig } from '../types';
import { Sliders, Eye, RefreshCw, Layers, ShieldCheck } from 'lucide-react';

interface SettingsPanelProps {
  config: TrackerConfig;
  onChangeConfig: (newConfig: TrackerConfig) => void;
  inputMode: 'webcam' | 'video' | 'simulator';
  onChangeInputMode: (mode: 'webcam' | 'video' | 'simulator') => void;
  selectedVideo: string;
  onChangeSelectedVideo: (url: string) => void;
  simulatorObjectsCount: number;
  onChangeSimulatorObjectsCount: (count: number) => void;
  isModelLoaded: boolean;
  onResetTracker: () => void;
}

const SAMPLE_VIDEOS = [
  {
    name: '🏙️ Shibuya Pedestrians',
    url: 'https://assets.mixkit.co/videos/preview/mixkit-pedestrians-walking-on-a-crosswalk-from-above-41858-large.mp4'
  },
  {
    name: '🛣️ Highway Traffic',
    url: 'https://assets.mixkit.co/videos/preview/mixkit-traffic-on-a-highway-at-night-41584-large.mp4'
  },
  {
    name: '💼 Workspace Collaboration',
    url: 'https://assets.mixkit.co/videos/preview/mixkit-people-working-in-a-modern-office-41595-large.mp4'
  }
];

const AVAILABLE_CLASSES = {
  dynamic: ['person', 'car', 'bicycle', 'dog', 'backpack', 'sports ball', 'chair', 'cell phone'],
  labelMap: {
    person: '👤 Person',
    car: '🚗 Vehicle / Car',
    bicycle: '🚲 Bicycle',
    dog: '🐕 Animal / Dog',
    backpack: '🎒 Backpack',
    'sports ball': '⚽ Sports Ball',
    chair: '🪑 Furniture / Chair',
    'cell phone': '📱 Electronic / Phone'
  } as Record<string, string>
};

export default function SettingsPanel({
  config,
  onChangeConfig,
  inputMode,
  onChangeInputMode,
  selectedVideo,
  onChangeSelectedVideo,
  simulatorObjectsCount,
  onChangeSimulatorObjectsCount,
  isModelLoaded,
  onResetTracker
}: SettingsPanelProps) {
  const handleToggleClass = (className: string) => {
    const isFiltered = config.filterClasses.includes(className);
    let newFilter: string[];
    if (isFiltered) {
      newFilter = config.filterClasses.filter(c => c !== className);
    } else {
      newFilter = [...config.filterClasses, className];
    }
    onChangeConfig({ ...config, filterClasses: newFilter });
  };

  const currentThemeHex = {
    purple: '#a855f7',
    cyan: '#06b6d4',
    pink: '#f43f5e',
    amber: '#f29900'
  }[config.colorTheme];

  return (
    <div className="flex flex-col gap-6 bg-slate-950/60 backdrop-blur-md rounded-2xl border border-purple-950/40 p-5 shadow-[0_0_30px_rgba(168,85,247,0.05)] text-slate-100">
      {/* SECTION 1: SYSTEM INPUT CONTROLLER */}
      <div>
        <div className="flex items-center gap-2 mb-4 border-b border-purple-950/30 pb-2">
          <Layers className="w-5 h-5 text-purple-400" />
          <h2 className="text-sm font-semibold tracking-wider uppercase text-purple-300 font-sans">
            Sensor Feed Input Selection
          </h2>
        </div>

        <div className="grid grid-cols-3 gap-2 p-1 bg-slate-900/60 rounded-lg border border-slate-800/40">
          <button
            onClick={() => onChangeInputMode('webcam')}
            className={`py-2 text-xs font-medium rounded-md transition-all duration-300 ${
              inputMode === 'webcam'
                ? 'bg-purple-600 text-white shadow-[0_0_12px_rgba(168,85,247,0.4)]'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
            }`}
          >
            📹 Webcam Feed
          </button>
          <button
            onClick={() => onChangeInputMode('video')}
            className={`py-2 text-xs font-medium rounded-md transition-all duration-300 ${
              inputMode === 'video'
                ? 'bg-purple-600 text-white shadow-[0_0_12px_rgba(168,85,247,0.4)]'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
            }`}
          >
            🎞️ Asset Video
          </button>
          <button
            onClick={() => onChangeInputMode('simulator')}
            className={`py-2 text-xs font-medium rounded-md transition-all duration-300 ${
              inputMode === 'simulator'
                ? 'bg-purple-600 text-white shadow-[0_0_12px_rgba(168,85,247,0.4)]'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
            }`}
          >
            📡 Radar Simulator
          </button>
        </div>

        {/* Input Mode Specific Sub-Controls */}
        <div className="mt-4 p-3 bg-slate-900/30 rounded-lg border border-slate-900/50 min-h-[70px] flex flex-col justify-center">
          {inputMode === 'webcam' && (
            <div className="text-xs text-slate-400 leading-relaxed">
              <span className="text-purple-400 font-semibold font-mono">LIVE FEED STATUS:</span> Activating native user camera device. Requires hardware permissions. Overlap bounding boxes draw dynamically over frames.
            </div>
          )}

          {inputMode === 'video' && (
            <div className="flex flex-col gap-2">
              <label className="text-[10px] font-mono text-purple-400 uppercase tracking-tight">Select Demo Footage Asset</label>
              <select
                value={selectedVideo}
                onChange={(e) => onChangeSelectedVideo(e.target.value)}
                className="w-full bg-slate-950 text-xs px-2.5 py-1.5 rounded border border-slate-800 text-slate-200 focus:outline-none focus:border-purple-500 font-sans"
              >
                {SAMPLE_VIDEOS.map((vid, idx) => (
                  <option key={idx} value={vid.url}>
                    {vid.name}
                  </option>
                ))}
              </select>
            </div>
          )}

          {inputMode === 'simulator' && (
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <label className="text-[10px] font-mono text-purple-400 uppercase tracking-tight">Target Object Volume</label>
                <span className="text-xs font-mono font-bold text-cyan-400">{simulatorObjectsCount} Objects</span>
              </div>
              <input
                type="range"
                min="1"
                max="10"
                step="1"
                value={simulatorObjectsCount}
                onChange={(e) => onChangeSimulatorObjectsCount(Number(e.target.value))}
                className="w-full accent-cyan-500 cursor-pointer"
              />
              <div className="text-[10px] text-cyan-400/80 font-mono text-center">
                Built-in physics playground with experimental structural occlusion dividers.
              </div>
            </div>
          )}
        </div>
      </div>

      {/* SECTION 2: AI TRACKER MATHEMATICAL PARAMETERS */}
      <div>
        <div className="flex items-center gap-2 mb-4 border-b border-purple-950/30 pb-2">
          <Sliders className="w-5 h-5 text-purple-400" />
          <h2 className="text-sm font-semibold tracking-wider uppercase text-purple-300 font-sans">
            Tracking Algorithm Config (SORT)
          </h2>
        </div>

        <div className="flex flex-col gap-4">
          {/* IOU Threshold Slider */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">IoU Intersection Threshold</span>
              <span className="font-mono text-purple-400 font-bold">{config.iouThreshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.8"
              step="0.05"
              value={config.iouThreshold}
              onChange={(e) => onChangeConfig({ ...config, iouThreshold: Number(e.target.value) })}
              className="w-full accent-purple-500 cursor-pointer"
            />
            <span className="text-[10px] text-slate-500 italic">
              Minimum spatial overlap required to match objects between sequential frames.
            </span>
          </div>

          {/* Model Confidence Slider */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">Model Confidence Minimum</span>
              <span className="font-mono text-purple-400 font-bold">{(config.confidenceThreshold * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.30"
              max="0.85"
              step="0.05"
              value={config.confidenceThreshold}
              disabled={inputMode === 'simulator' ? false : !isModelLoaded}
              className="w-full accent-purple-500 cursor-pointer disabled:opacity-40"
              onChange={(e) => onChangeConfig({ ...config, confidenceThreshold: Number(e.target.value) })}
            />
            <span className="text-[10px] text-slate-500 italic">
              Filter incoming noisy object proposals before submitting to SORT tracker.
            </span>
          </div>

          {/* Target lost persistence limits */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">Lost Frame Retention</span>
              <span className="font-mono text-purple-400 font-bold">{config.maxMissedFrames} Frames</span>
            </div>
            <input
              type="range"
              min="5"
              max="45"
              step="5"
              value={config.maxMissedFrames}
              onChange={(e) => onChangeConfig({ ...config, maxMissedFrames: Number(e.target.value) })}
              className="w-full accent-purple-500 cursor-pointer"
            />
            <span className="text-[10px] text-slate-500 italic">
              Survival frames count during which tracker estimates missing targets via velocity projection.
            </span>
          </div>

          {/* Trail Length */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">Particle Trail Length</span>
              <span className="font-mono text-purple-400 font-bold">{config.historyLength} points</span>
            </div>
            <input
              type="range"
              min="10"
              max="60"
              step="5"
              value={config.historyLength}
              onChange={(e) => onChangeConfig({ ...config, historyLength: Number(e.target.value) })}
              className="w-full accent-purple-500 cursor-pointer"
            />
          </div>
        </div>
      </div>

      {/* SECTION 3: VISUAL PARAMETERS & INTERFACE MATRIX */}
      <div>
        <div className="flex items-center gap-2 mb-4 border-b border-purple-950/30 pb-2">
          <Eye className="w-5 h-5 text-purple-400" />
          <h2 className="text-sm font-semibold tracking-wider uppercase text-purple-300 font-sans">
            Visual & System Overlays
          </h2>
        </div>

        <div className="flex flex-col gap-3">
          {/* Theme highlight toggles */}
          <div className="flex flex-col gap-1.5">
            <label className="text-[10px] font-mono text-purple-400 uppercase tracking-tight">UI Neon Synergy Highlight</label>
            <div className="grid grid-cols-4 gap-2">
              {(['purple', 'cyan', 'pink', 'amber'] as const).map((thm) => {
                const colorsMap = {
                  purple: 'bg-purple-500 border-purple-400 box-shadow-[0_0_10px_purple]',
                  cyan: 'bg-cyan-500 border-cyan-400',
                  pink: 'bg-rose-500 border-rose-400',
                  amber: 'bg-amber-500 border-amber-400'
                };
                return (
                  <button
                    key={thm}
                    onClick={() => onChangeConfig({ ...config, colorTheme: thm })}
                    className={`h-8 rounded-lg border-2 capitalize text-[10px] font-mono tracking-tighter ${
                      config.colorTheme === thm ? 'border-white text-white font-bold' : 'border-slate-800 text-slate-400'
                    } ${colorsMap[thm]} transition-all duration-300 flex items-center justify-center`}
                  >
                    {thm}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 mt-1.5">
            <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={config.drawTrails}
                onChange={(e) => onChangeConfig({ ...config, drawTrails: e.target.checked })}
                className="w-4 h-4 rounded text-purple-600 focus:ring-purple-500 focus:ring-offset-slate-950 bg-slate-900 border-slate-800"
              />
              <span>Render Route Trails</span>
            </label>

            <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={config.blurBackground}
                onChange={(e) => onChangeConfig({ ...config, blurBackground: e.target.checked })}
                className="w-4 h-4 rounded text-purple-600 focus:ring-purple-500 focus:ring-offset-slate-950 bg-slate-900 border-slate-800"
              />
              <span>Privacy Blur Feed</span>
            </label>
          </div>
        </div>
      </div>

      {/* SECTION 4: CLASS FILTERS LIST */}
      <div>
        <div className="flex items-center gap-2 mb-4 border-b border-purple-950/30 pb-2">
          <ShieldCheck className="w-5 h-5 text-purple-400" />
          <h2 className="text-sm font-semibold tracking-wider uppercase text-purple-300 font-sans">
            Active Object Class Filter
          </h2>
        </div>

        <div className="grid grid-cols-2 gap-2 max-h-[140px] overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-purple-950">
          {AVAILABLE_CLASSES.dynamic.map((cls) => {
            const isFiltered = config.filterClasses.includes(cls);
            return (
              <button
                key={cls}
                onClick={() => handleToggleClass(cls)}
                className={`flex items-center gap-2 px-2.5 py-1.5 rounded-lg border text-left text-xs transition-all duration-300 ${
                  isFiltered
                    ? 'bg-slate-950 border-slate-900 text-slate-500 line-through'
                    : 'bg-slate-900/40 border-purple-950/40 text-slate-200 hover:border-purple-500/40'
                }`}
              >
                <div
                  className={`w-2 h-2 rounded-full ${isFiltered ? 'bg-slate-800' : 'bg-green-400 animate-pulse'}`}
                  style={{ backgroundColor: isFiltered ? undefined : currentThemeHex }}
                />
                <span className="truncate">{AVAILABLE_CLASSES.labelMap[cls] || cls}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* RESET TRACKS ENGINE BUTTON */}
      <button
        onClick={onResetTracker}
        className="w-full mt-2 py-3 bg-slate-900 hover:bg-slate-800/80 active:bg-slate-950 border border-purple-800/40 hover:border-purple-500/50 rounded-xl transition-all duration-300 text-xs font-semibold tracking-wider uppercase font-mono flex items-center justify-center gap-2 text-purple-200 shadow-md"
      >
        <RefreshCw className="w-4 h-4 text-purple-400" />
        Flush SORT Tracker Engine
      </button>
    </div>
  );
}
